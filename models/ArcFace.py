import torch
import torch.nn.functional as F
import math

from math import pi
from torch import nn
from torchvision.models.vgg import make_layers


class ArcFace(nn.Module):
    def __init__(self, num_layer, in_features, out_features, num_class, use_embedding, activation, s=30.0, m=0.40, easy_margin=False, device=torch.device("cpu")):
        super().__init__()
        self.num_layer = num_layer
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        #
        self.weight = nn.Parameter(torch.empty(num_class, out_features)).to(device)
        nn.init.xavier_normal_(self.weight)
        #
        self.easy_margin = easy_margin
        self.use_embedding = use_embedding
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0,180]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        #
        self.embedding = nn.Embedding(num_embeddings=num_class, embedding_dim=out_features).to(device)
        #
        self.layers = nn.ModuleList(self.make_layer())
        #
        if activation == 'GELU':
            self.atfc = nn.GELU()
        else:
            self.atfc = nn.ReLU()

    def make_layer(self):
        layers = []
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

    def forward(self, input, J_tokens, mode, m=None, s=None):
        output = None
        emb_output_J_tokens = self.embedding(J_tokens)  # 4, 512
        out = input
        if input.sum() != 0:
            for i, layer in enumerate(self.layers):
                y = layer(out)
                if y.shape[-1] == out.shape[-1]:
                    out = y + out
                else:
                    out = y
                if i != len(self.layers) - 1:
                    # out = nn.BatchNorm1d(out.shape[-1])(out)
                    out = self.atfc(out)

            if self.use_embedding:
                embedding_vec = out + emb_output_J_tokens
            else:
                embedding_vec = out

        else:
            embedding_vec = emb_output_J_tokens

        #
        if mode == 'training':
            label = J_tokens

            # cos(theta)
            cosine = F.linear(F.normalize(embedding_vec), F.normalize(self.weight))
            # cos(theta + m)
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            if m is not None and s is not None:
                self.cos_m = math.cos(m)
                self.sin_m = math.sin(m)
                self.th = math.cos(math.pi - m)
                self.mm = math.sin(math.pi - m) * m

            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

            # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
        return output, embedding_vec
