
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
        self.embedding = nn.Embedding(num_embeddings=self.config.NUM_JOINTS, embedding_dim=self.config.OUT_FEAT).to(self.config.DEVICE)
        #
        self.in_features = self.config.IN_FEAT
        self.out_features = self.config.OUT_FEAT
        self.num_layer = self.config.NUM_LAYER
        #
        if self.config.EMB_MODE != 'BASIS':
            self.layers = nn.ModuleList(self.make_layer())

        # ArcFace classifier는 USE_ARCFACE와 관계없이 항상 생성
        self.weight = nn.Parameter(
            torch.empty(
                self.config.NUM_JOINTS,
                self.out_features,
                device=self.config.DEVICE,
            )
        )
        nn.init.xavier_uniform_(self.weight)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        #
        # 3. EMB_INIT=False일 때 세 pretrained weight 로드
        if not self.config.EMB_INIT:
            self.load_state_dict_embedding()

            if self.config.EMB_MODE != 'BASIS':
                self.load_state_dict_linear()

            self.load_state_dict_arcface_classifier()

            print(
                '[LOADED] MLP, nn.Embedding, '
                'ArcFace classifier pretrained weights loaded.'
            )
        else:
            print(
                '[INITIALIZED] MLP, nn.Embedding, '
                'ArcFace classifier randomly initialized.'
            )

        # 4. activation
        if self.config.ACTIV == 'GELU':
            self.atfc = nn.GELU()
        else:
            self.atfc = nn.ReLU()

            # ==================================================
            # Freeze 설정
            # pretrained load 또는 random init이 끝난 뒤 적용
            # ==================================================

            # Relative MLP
        if hasattr(self, 'layers'):
            self.layers.requires_grad_(
                not self.config.RELATIVE_FREEZE
            )

            # Basis nn.Embedding
        self.embedding.weight.requires_grad_(
            not self.config.BASIS_FREEZE
        )

        # ArcFace classifier
        # USE_ARCFACE=True일 때만 gradient 생성
        self.weight.requires_grad_(
            self.config.USE_ARCFACE
        )

        # ==================================================
        # PAD(0), SEP(1) embedding은 항상 고정
        # ==================================================

        # pretrained load 또는 random initialization이 끝난 현재 값을 보존
        self.register_buffer('_frozen_special_embedding', self.embedding.weight[:2].detach().clone(), persistent=False,)

        self._special_embedding_hook = None

        # nn.Embedding 전체가 unfreeze 상태인 경우에도
        # PAD와 SEP row의 gradient만 제거한다.
        if self.embedding.weight.requires_grad:
            self._special_embedding_hook = (self.embedding.weight.register_hook(self._freeze_special_embedding_gradient))

        # ==================================================
        # Embedder optimizer 파라미터 등록
        # ==================================================

        params = []

        # Relative MLP
        if (hasattr(self, 'layers') and not self.config.RELATIVE_FREEZE):
            params.extend(self.layers.parameters())

        # Basis nn.Embedding
        if not self.config.BASIS_FREEZE:
            params.extend(self.embedding.parameters())

        # ArcFace classifier
        if self.config.USE_ARCFACE:
            params.append(self.weight)

        # ==================================================
        # Optimizer / Scheduler 초기 상태
        # ==================================================

        self.optimizer = None
        self.scheduler = None

        if params:
            self.optimizer = torch.optim.Adam(
                params,
                lr=5e-5,
            )

        if self.config.USE_ARCFACE:
            self.losses = AverageMeter()

    @staticmethod
    def _freeze_special_embedding_gradient(grad):
        """
        nn.Embedding의 PAD(0), SEP(1) row gradient를 제거한다.
        관절 row 2~21의 gradient는 유지한다.
        """
        grad = grad.clone()
        grad[:2].zero_()
        return grad

    @torch.no_grad()
    def restore_frozen_special_embeddings(self):
        """
        optimizer step 이후 PAD·SEP를 학습 시작 시점 값으로 복원한다.

        gradient hook뿐 아니라 weight decay나 optimizer momentum으로
        값이 변하는 경우까지 방지한다.
        """
        self.embedding.weight[:2].copy_(self._frozen_special_embedding)

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
        if os.path.isfile(self.config.PRETRAINED_EMB_PATH):
            pretrained_emb_weight = torch.load(self.config.PRETRAINED_EMB_PATH, map_location=self.config.DEVICE)
            self.embedding.weight.data.copy_(pretrained_emb_weight['weight'])
            if not self.config.BASIS_FREEZE:
                print("Basis nn.Embedding is NOT frozen..!!!")
                self.embedding.weight.requires_grad = True
            else:
                print("Basis nn.Embedding is frozen..!!!")
                self.embedding.weight.requires_grad = False

            self.embedding.to(self.config.DEVICE)
            print('Pretrained embedding weights loaded successfully.')

        else:
            raise ValueError("NOT EXIST PRETRAINED EMBEDDING WEIGHT PATH")

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
    def load_state_dict_arcface_classifier(self):
        path = self.config.PRETRAINED_ARCFACE_CLASSIFIER_PATH

        if not os.path.isfile(path):
            raise ValueError(
                f"NOT EXIST PRETRAINED ARCFACE CLASSIFIER PATH: {path}"
            )

        arcface_state = torch.load(
            path,
            map_location=self.config.DEVICE,
        )

        if 'weight' not in arcface_state:
            raise KeyError(
                f"ArcFace checkpoint does not contain 'weight': {path}"
            )

        loaded_weight = arcface_state['weight']
        expected_shape = (
            self.config.NUM_JOINTS,
            self.out_features,
        )

        if tuple(loaded_weight.shape) != expected_shape:
            raise ValueError(
                "ArcFace weight shape mismatch: "
                f"expected={expected_shape}, "
                f"loaded={tuple(loaded_weight.shape)}"
            )

        with torch.no_grad():
            self.weight.copy_(loaded_weight)

        print('Pretrained ArcFace classifier loaded successfully.')

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


    def compute_arcface_logits(self, embeddings, labels):
        """
        embeddings: [N, 768]
        labels:     [N]
        """
        s = self.config.ARCFACE_PARAM['s']
        m = self.config.ARCFACE_PARAM['m']

        cosine = F.linear(
            F.normalize(embeddings.float(), p=2, dim=1, eps=1e-6),
            F.normalize(self.weight.float(), p=2, dim=1, eps=1e-6)
        )

        cosine = cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(0.0))

        cos_m = math.cos(m)
        sin_m = math.sin(m)
        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m

        phi = cosine * cos_m - sine * sin_m

        phi = torch.where(cosine > th, phi, cosine - mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(dim=1, index=labels.unsqueeze(1), value=1.0 )

        logits = (one_hot * phi + (1.0 - one_hot) * cosine)

        return logits * s

        ### Relative 일괄 처리 함수

    def apply_relative_layers(self, videos):
        """
        videos: [B, F, J, 3]
        return: [B, F, J, 768]
        """
        B, F, J, C = videos.shape

        relative_input = videos[..., :self.in_features]
        out = relative_input.reshape(-1, self.in_features)

        for i, layer in enumerate(self.layers):
            y = layer(out)

            if y.shape[-1] == out.shape[-1]:
                out = y + out
            else:
                out = y

            if i != len(self.layers) - 1:
                out = self.atfc(out)

        return out.reshape(B, F, J, self.out_features)

    def forward(self, videos):
        """
        videos: [B, F, J, 3]

        return:
            joint_embedding: [B, F, J, 768]
            frame_pad_mask:  [B, F]
            arcface_loss_sum: scalar or None
            arcface_count:    int
        """
        videos = videos.to(self.config.DEVICE, non_blocking=True).float()

        B, F, J, C = videos.shape

        # 원본 좌표가 전부 0이면 padding frame
        # shape: [B, F]
        frame_pad_mask = (videos.eq(0).all(dim=-1).all(dim=-1))

        joint_ids = torch.tensor([self.vocab[joint_name] for joint_name in self.config.JOINTS_NAME], dtype=torch.long, device=self.config.DEVICE)

        # Relative
        if self.config.EMB_MODE != 'BASIS':
            relative = self.apply_relative_layers(videos)
        else:
            relative = None

        # Basis: [1, 1, J, 768]
        basis = self.embedding(joint_ids).view(1, 1, J, self.out_features)

        if self.config.EMB_MODE == 'RELATIVE_BASIS':
            joint_embedding = relative + basis


        elif self.config.EMB_MODE in ('RELATIVE', 'RELATIVEwID'):
            joint_embedding = relative

        elif self.config.EMB_MODE == 'BASIS':
            joint_embedding = basis.expand(B, F, J, self.out_features)

        else:
            raise ValueError(
                f"Unsupported EMB_MODE: {self.config.EMB_MODE}"
            )

        # padding frame의 joint 위치는 PAD embedding으로 채움
        # 실제 mask 판정에는 이 값을 사용하지 않음
        pad_embedding = self.embedding.weight[0].view(1, 1, 1, self.out_features)
        joint_embedding = torch.where(frame_pad_mask[:, :, None, None], pad_embedding, joint_embedding)

        arcface_loss_sum = None
        arcface_count = 0

        if self.config.USE_ARCFACE:
            # token 0=PAD, 1=SEP, 2~21=joint
            special_ids = torch.tensor([0, 1], dtype=torch.long, device=self.config.DEVICE)
            special_embedding = self.embedding(special_ids).view(1, 1, 2, self.out_features).expand(B, F, 2, self.out_features)

            # [B, F, 22, 768]
            all_token_embedding = torch.cat([special_embedding, joint_embedding], dim=2)

            # padding frame 전체는 ArcFace 학습에서 제외
            # [N_real_frames, 22, 768]
            real_frame_embedding = all_token_embedding[~frame_pad_mask]

            if real_frame_embedding.numel() > 0:
                token_labels = torch.cat([special_ids, joint_ids], dim=0)
                token_labels = token_labels.view(1, -1).expand(real_frame_embedding.size(0), -1)

                flat_embedding = real_frame_embedding.reshape(-1, self.out_features)

                flat_labels = token_labels.reshape(-1)

                arcface_logits = self.compute_arcface_logits(flat_embedding, flat_labels)

                arcface_loss_sum = self.criterion(arcface_logits, flat_labels)

                arcface_count = flat_labels.numel()

            else:
                # 예외적인 all-padding batch
                arcface_loss_sum = (joint_embedding.sum() * 0.0)

        return (joint_embedding, frame_pad_mask, arcface_loss_sum, arcface_count)


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