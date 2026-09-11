
import os
import pprint

import torch
import yaml
import pickle
import random
import numpy as np
import torch.nn.functional as F
import math

from torch import nn
from tqdm import tqdm
from pprint import pprint
from types import SimpleNamespace
from Embedder.Embedder_config import config
from collections import OrderedDict
from Embedder.data_loader import Video_Loader
from others.AverageMeter import AverageMeter
# # codex로 embedder API forward, preprocess, arcface 수정하기 직전의 코드 백업
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
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode
        #
        self.vocab_path = self.get_vocab_path()
        self.vocab = self.get_vocab()
        #
        self.use_embedding = self.config.USE_EMBEDDING
        #
        self.embedding = nn.Embedding(num_embeddings=self.config.NUM_JOINTS, embedding_dim=self.config.OUT_FEAT).to(self.config.DEVICE)
        #
        self.in_features = self.config.IN_FEAT
        self.out_features = self.config.OUT_FEAT
        self.num_layer = self.config.NUM_LAYER
        #
        if self.config.EMB_MODE != 'BASIS':
            self.layers = nn.ModuleList(self.make_layer())
        #
        if not config.EMB_INIT:
            self.load_state_dict_embedding()
            self.load_state_dict_linear()
        else:
            print('[NOT LOADED] Embedder and nn.embedding module are not loaded.')
        #
        if self.config.ACTIV == 'GELU':
            self.atfc = nn.GELU()
        else:
            self.atfc = nn.ReLU()

        params = []
        self.optimizer = None
        self.scheduler = None
        if self.config.USE_ARCFACE:
            arcface_state = torch.load(self.config.PRETRAINED_ARCFACE_CLASSIFIER_PATH, map_location=self.config.DEVICE)

            if 'weight' not in arcface_state:
                raise KeyError(
                    "ArcFace checkpoint does not contain 'weight': "
                    f"{self.config.PRETRAINED_ARCFACE_CLASSIFIER_PATH}"
                )

            expected_shape = (self.config.NUM_JOINTS, self.out_features)
            loaded_shape = tuple(arcface_state['weight'].shape)

            if loaded_shape != expected_shape:
                raise ValueError(
                    f"ArcFace weight shape mismatch: "
                    f"expected={expected_shape}, loaded={loaded_shape}"
                )

            self.weight = nn.Parameter(arcface_state['weight'].to(self.config.DEVICE).detach().clone())
            params.append(self.weight)
            #
            self.criterion = nn.CrossEntropyLoss(reduction='sum')

        if self.config.EMB_MODE != 'BASIS' and not self.config.RELATIVE_FREEZE:
            params += list(self.layers.parameters())
        if self.use_embedding and not self.config.BASIS_FREEZE:
            params += list(self.embedding.parameters())

        if params:
            self.optimizer = torch.optim.Adam(params, lr=5e-5)

        if self.config.USE_ARCFACE:
            self.losses = AverageMeter()

    def get_vocab_path(self):
        if self.mode == 'train' or self.mode == 'train-valid':
            return self.config.TRAIN_JOINT_VOCAB_PATH
        else:
            return self.config.VALID_JOINT_VOCAB_PATH

    def get_vocab(self):
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        return vocab

    def load_state_dict_embedding(self):
        if self.config.USE_EMBEDDING:
            if os.path.isfile(self.config.PRETRAINED_EMB_PATH):
                pretrained_emb_weight = torch.load(self.config.PRETRAINED_EMB_PATH, map_location=self.config.DEVICE)
                self.embedding.weight.data.copy_(pretrained_emb_weight['weight'])
                if not config.BASIS_FREEZE:
                    print("Basis nn.Embedding is NOT frozen..!!!")
                    self.embedding.weight.requires_grad = True
                else:
                    print("Basis nn.Embedding is frozen..!!!")
                    self.embedding.weight.requires_grad = False

                self.embedding.to(self.config.DEVICE)
                print('Pretrained embedding weights loaded successfully.')

            else:
                raise ValueError("NOT EXIST PRETRAINED EMBEDDING WEIGHT PATH")

        else:
            pass

    def load_state_dict_linear(self):
        if os.path.isfile(self.config.PRETRAINED_PATH):
            pretrained_linear_weight = torch.load(self.config.PRETRAINED_PATH, map_location=self.config.DEVICE)
            #
            state_dict = OrderedDict()
            #
            if config.RELATIVE_FREEZE:
                print("Relative MLP Network is frozen..!")
            else:
                print("Relative MLP Network is NOT frozen..!")
            #
            for param in pretrained_linear_weight.keys():
                if param.startswith('layers.'):
                    state_dict[param[7:]] = pretrained_linear_weight[param]
                    if not config.RELATIVE_FREEZE:
                        state_dict[param].requires_grad = True
                    else:
                        state_dict[param].requires_grad = False
                else:
                    state_dict[param] = pretrained_linear_weight[param]
                    if not config.RELATIVE_FREEZE:
                        state_dict[param].requires_grad = True
                    else:
                        state_dict[param].requires_grad = False

            self.layers.load_state_dict(state_dict, strict=True)
            for param in self.layers.parameters():
                param.requires_grad = not self.config.RELATIVE_FREEZE
            self.layers.to(self.config.DEVICE)
            print('Pretrained linear weights loaded successfully.')

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
        # joint_info의 shape [BS, 3]
        BS = joint_info.size(0)
        if self.use_embedding:
            emb_output_J_tokens = self.embedding(joint_token)  # [BS, OUT_FEAT]

        # apply padding to match max_frame
        pad_mask = torch.all(joint_info == 0, dim=-1)  # [BS]
        embedding_vec = torch.zeros(BS, self.out_features, device=self.config.DEVICE)

        # handle padded and non-padded frames separately
        # non-padded frames -> apply B+R or R operation
        if (~pad_mask).any(): #
            out = joint_info[~pad_mask]
            if self.use_embedding:
                out_J_token = emb_output_J_tokens[~pad_mask]

            # only operate when emb_mode is B+R or R
            if self.config.EMB_MODE != 'BASIS':
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
                # When EMB_MODE != 'BASIS', out(R or B+R, 784dim) is used to embedding_vec.
                embedding_vec[~pad_mask] = out

            # When EMB_MODE == 'BASIS', out_J_token(output of nn.embedding(), 784dim) is used to embedding_vec.
            else:
                embedding_vec[~pad_mask] = out_J_token

        # padded frames -> embedding set to 0
        if pad_mask.any():
            pad_emb = self.embedding(torch.zeros(pad_mask.sum(), dtype=torch.long, device=self.config.DEVICE))
            embedding_vec[pad_mask] = pad_emb

        # -------------------------
        # ArcFace (valid only)
        # -------------------------
        arcface_out = None
        joint_cls_valid = None
        valid_mask = ~pad_mask

        if self.config.USE_ARCFACE and valid_mask.any():
            s, m = self.config.ARCFACE_PARAM['s'], self.config.ARCFACE_PARAM['m']

            # filter to valid samples only
            emb_valid = embedding_vec[valid_mask]  # [N_valid, D]
            joint_cls_valid = (joint_token[valid_mask]).long()  # [N_valid]

            cosine = F.linear(F.normalize(emb_valid), F.normalize(self.weight))  # [N_valid, C]

            # numerical safety: clamp cosine to avoid sqrt(negative)
            cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

            if m is not None and s is not None:
                self.cos_m = math.cos(m)
                self.sin_m = math.sin(m)
                self.th = math.cos(math.pi - m)
                self.mm = math.sin(math.pi - m) * m

            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

            one_hot = torch.zeros_like(cosine)  # [N_valid, C]
            one_hot.scatter_(1, joint_cls_valid.view(-1, 1), 1)

            arcface_out = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            arcface_out = arcface_out * s

        return arcface_out, embedding_vec, joint_cls_valid

    def forward(self, videos):
        # # # consider batch
        # vec_for_a_frame = []
        #
        # for frame_idx in range(self.config.MAX_FRAMES):
        #     vec_for_a_joint = []
        #     for i, joint_name in enumerate(self.config.JOINTS_NAME): # consider only joint(2~21), not pad, cls token (0,1)
        #         joint_info, joint_token = self.preprocess_joint_info(videos, frame_idx, joint_name)
        #         vec = self.forward_propagation(joint_info, joint_token)
        #         vec_for_a_joint.append(vec)
        #     a_frame = torch.stack(vec_for_a_joint, dim=1) # stacked shape: [BS, 20, 768]
        #     vec_for_a_frame.append(a_frame)
        # videos = torch.stack(vec_for_a_frame, dim=1) # BS, MAX_FRAME, 20, 768
        #
        # return videos

        vec_for_a_frame = []

        # ArcFace loss accumulator
        loss_sum = None
        valid_count = 0

        if self.config.USE_ARCFACE:
            loss_sum = torch.tensor(0.0, device=self.config.DEVICE)

        for frame_idx in range(self.config.MAX_FRAMES):
            vec_for_a_joint = []

            for joint_name in self.config.JOINTS_NAME:
                joint_info, joint_token = self.preprocess_joint_info(videos, frame_idx, joint_name)

                arcface_out, vec, joint_cls_valid = self.forward_propagation(joint_info, joint_token)
                vec_for_a_joint.append(vec)

                if self.config.USE_ARCFACE:
                    # only add loss when we actually computed logits for valid samples
                    if arcface_out is not None and joint_cls_valid is not None and arcface_out.numel() > 0:
                        loss_sum = loss_sum + self.criterion(arcface_out, joint_cls_valid)
                        valid_count += joint_cls_valid.numel()

            a_frame = torch.stack(vec_for_a_joint, dim=1)  # [BS, 20, 768]
            vec_for_a_frame.append(a_frame)

        videos = torch.stack(vec_for_a_frame, dim=1)  # [BS, MAX_FRAME, 20, 768]

        if self.config.USE_ARCFACE:
            return videos, loss_sum, valid_count
        else:
            return videos


if __name__ == '__main__':
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    #
    video_dataset = Video_Loader(config=config)
    video_loader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        collate_fn=video_dataset.collate_fn
    )

    #
    embedder = Embedder(config)
    for i, (videos, exercise_class) in enumerate(tqdm(video_loader, desc='embedding', total=len(video_loader))):
        output = embedder(videos)
        print()
        # BERTSUM(output)
    print()

    # with open('/home/jysuh/PycharmProjects/coord_embedding/dataset/embedder_dataset/valid.pkl', 'wb') as f:
    #     pickle.dump(lst, f)