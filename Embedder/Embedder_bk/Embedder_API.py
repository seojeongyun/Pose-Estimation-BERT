import os
import pprint

import torch
import yaml
import math
import random
import numpy as np

from torch import nn
from tqdm import tqdm
from pprint import pprint
from types import SimpleNamespace
from Embedder.Embedder_config import config
from collections import OrderedDict
from Embedder.data_loader import Video_Loader


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, SimpleNamespace):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

class Embedder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        #
        self.vocab = vocab
        #
        self.in_features = self.config.IN_FEAT
        self.out_features = self.config.OUT_FEAT
        self.num_layer = self.config.NUM_LAYER
        #
        self.use_embedding = self.config.USE_EMBEDDING
        #
        self.embedding = nn.Embedding(num_embeddings=self.config.NUM_JOINTS, embedding_dim=self.config.OUT_FEAT).to(self.config.DEVICE)
        self.load_state_dict_embedding()
        #
        self.layers = nn.ModuleList(self.make_layer())
        self.load_state_dict_linear()
        #
        if self.config.ACTIV == 'GELU':
            self.atfc = nn.GELU()
        else:
            self.atfc = nn.ReLU()

    def load_state_dict_embedding(self):
        if self.config.USE_EMBEDDING:
            if os.path.isfile(self.config.PRETRAINED_EMB_PATH):
                pretrained_emb_weight = torch.load(self.config.PRETRAINED_EMB_PATH)
                self.embedding.weight.data.copy_(pretrained_emb_weight['weight'])
                if config.BASIS_FREEZE:
                    self.embedding.weight.requires_grad = False

                self.embedding.to(self.config.DEVICE)
                pprint('Pretrained embedding weights loaded successfully.')

            else:
                raise ValueError("NOT EXIST PRETRAINED EMBEDDING WEIGHT PATH")

        else:
            pass

    def load_state_dict_linear(self):
        if os.path.isfile(self.config.PRETRAINED_PATH):
            pretrained_linear_weight = torch.load(self.config.PRETRAINED_PATH, map_location=self.config.DEVICE)
            #
            del pretrained_linear_weight['embedding.weight']
            #
            state_dict = OrderedDict()
            #
            for param in pretrained_linear_weight.keys():
                if param.startswith('layers.'):
                    state_dict[param[7:]] = pretrained_linear_weight[param]
                    if config.RELATIVE_FREEZE:
                        state_dict[param[7:]].requires_grad = False
                else:
                    state_dict[param] = pretrained_linear_weight[param]
                    if config.RELATIVE_FREEZE:
                        state_dict[param].requires_grad = False
            self.layers.load_state_dict(state_dict, strict=False)
            self.layers.to(self.config.DEVICE)
            pprint('Pretrained linear weights loaded successfully.')

        else:
            raise ValueError('NOT EXIST PRETRAINED LINEAR WEIGHT PATH')

    def make_layer(self):
        layers = []
        #
        if self.num_layer == 2:
            layers.append(nn.Linear(self.in_features, self.out_features//2, bias=True))
            layers.append(nn.Linear(self.out_features//2, self.out_features, bias=False))
        elif self.num_layer == 4:
            layers.append(nn.Linear(self.in_features, self.out_features//4, bias=True))
            layers.append(nn.Linear(self.out_features//4, self.out_features//2, bias=True))
            layers.append(nn.Linear(self.out_features//2, self.out_features//4, bias=True))
            layers.append(nn.Linear(self.out_features//4, self.out_features, bias=False))
        elif self.num_layer == 6:
            layers.append(nn.Linear(self.in_features, self.out_features//8, bias=True))
            layers.append(nn.Linear(self.out_features//8, self.out_features//4, bias=True))
            layers.append(nn.Linear(self.out_features//4, self.out_features//2, bias=True))
            layers.append(nn.Linear(self.out_features//2, self.out_features//4, bias=True))
            layers.append(nn.Linear(self.out_features//4, self.out_features//2, bias=True))
            layers.append(nn.Linear(self.out_features//2, self.out_features, bias=False))
        return layers

    def encode_joint_info(self, videos, exercise_names, view_idx):
        BS = len(videos)
        #
        for video_idx in range(BS):
            exercise_name = exercise_names[video_idx]
            # if frame len of current video is smaller than max_frame
            if int(list(videos[video_idx].keys())[-1]) != self.config.MAX_FRAMES-1:
                for i in range(int(list(videos[video_idx].keys())[-1])+1, self.config.MAX_FRAMES):
                    videos[video_idx].setdefault(str(i), {k : np.zeros(4, dtype=np.float32) for k in self.config.JOINTS_NAME})
            #
            for frame_idx, a_frame in videos[video_idx].items():
                for joint_name, joint_value in a_frame.items():
                    if np.all(joint_value == 0):
                        continue
                    x_norm = joint_value[0] / self.config.IMG_SIZE[0]
                    y_norm = joint_value[1] / self.config.IMG_SIZE[1]
                    a_frame[joint_name] = np.array([x_norm, y_norm, self.vocab[joint_name], exercise_name], dtype=np.float32)
                if list(a_frame.keys()) != self.config.JOINTS_NAME:
                    print()
                    pass
        return videos

    def preprocess_joint_info(self, videos, frame_idx, joint_name):
        # Collect joint information from (BS) videos loaded by torch dataloader,
        # which results in a data with the shape of [BS, 4] (joint per frame parallel)
        joint_info = []
        joint_token = []
        for video_idx in range(len(videos)):
            joint_info.append(videos[video_idx][str(frame_idx)][joint_name])
            joint_token.append(self.vocab[joint_name])

        joint_info = np.array(joint_info)
        joint_info = torch.from_numpy(joint_info).to(self.config.DEVICE)
        joint_token = torch.tensor(joint_token).to(self.config.DEVICE)
        return joint_info, joint_token

    def forward_propagation(self, joint_info, joint_token):
        BS = joint_info.size(0)
        if self.use_embedding:
            emb_output_J_tokens = self.embedding(joint_token)  # [BS, OUT_FEAT]

        pad_mask = torch.all(joint_info == 0, dim=-1)  # [BS]
        embedding_vec = torch.zeros(BS, self.out_features, device=self.config.DEVICE)

        if (~pad_mask).any(): # not pad
            out = joint_info[~pad_mask]
            if self.use_embedding:
                out_J_token = emb_output_J_tokens[~pad_mask]

            for i, layer in enumerate(self.layers):
                y = layer(out)
                if y.shape[-1] == out.shape[-1]:
                    out = y + out
                else:
                    out = y
                if i != len(self.layers) - 1:
                    out = self.atfc(out)

            if self.use_embedding:
                out = out + out_J_token
            #
            embedding_vec[~pad_mask] = out

        if pad_mask.any():
            pad_emb = self.embedding(torch.zeros(pad_mask.sum(), dtype=torch.long, device=self.config.DEVICE))
            embedding_vec[pad_mask] = pad_emb

        return embedding_vec

    def forward(self, videos, exercise_name, view_idx):
        # consider batch
        vec_for_a_frame = []

        # The shape of videos: (BS, MAX_FRAME, 20, x, y)
        videos = self.encode_joint_info(videos, exercise_name, view_idx)

        for frame_idx in range(self.config.MAX_FRAMES):
            vec_for_a_joint = []
            for i, joint_name in enumerate(self.config.JOINTS_NAME):
                joint_info, joint_token = self.preprocess_joint_info(videos, frame_idx, joint_name)
                vec = self.forward_propagation(joint_info, joint_token)
                vec_for_a_joint.append(vec)
            a_frame = torch.stack(vec_for_a_joint, dim=1) # stacked shape: [BS, 20, 768]
            vec_for_a_frame.append(a_frame)
        videos = torch.stack(vec_for_a_frame, dim=1) # BS, MAX_FRAME, 20, 768

        return videos


if __name__ == '__main__':
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    #
    video_dataset = Video_Loader(config=config, data_path=config.DATA_PATH)
    video_loader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        collate_fn=video_dataset.collate_fn
    )

    #
    embedder = Embedder(config, video_dataset.vocab)
    for i, (videos, exercise_name, view_idx) in enumerate(tqdm(video_loader, desc='embedding', total=len(video_loader))):
        output = embedder(videos, exercise_name, view_idx)
        # BERTSUM(output)
    print()