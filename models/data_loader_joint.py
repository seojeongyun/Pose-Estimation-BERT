from sympy.benchmarks.bench_discrete_log import data_set_1
from torch.utils.data import Dataset
import numpy as np
import os
import argparse
import torch
import glob
import random
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn

from models.encoder import PositionalEncoding


class joint_dataset(Dataset):
    def __init__(self, args, shuffle, is_test, device):
        self.args = args
        self.pts = sorted(glob.glob(os.path.join(args.bert_data_path, '*.pt')))
        self.shuffle = shuffle
        self.is_test = is_test
        self.device = device
        if self.shuffle:
            random.shuffle(self.pts)
        # =============================
        self.pad_id = args.pad_id
        self.sep_id = args.sep_id
        self.preweights = torch.load(args.weight_path)

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        data = torch.load(self.pts[idx])
        device = self.device
        src = data['src']
        tgt = torch.tensor(data['tgt']-22).to(device)
        #
        pre_emb = nn.Embedding.from_pretrained(self.preweights['weight'], freeze=True)
        pad_emb = pre_emb(torch.tensor(self.pad_id).to(device)).unsqueeze(0)
        sep_emb = pre_emb(torch.tensor(self.sep_id).to(device)).unsqueeze(0)
        #
        segment_emb = nn.Embedding(2,768).to(device)
        cls_emb = torch.zeros([1,768]).to(device)
        pos_enc = PositionalEncoding(dropout=0.2, dim=768)
        #
        tensor_list = [cls_emb]
        seg_list = [0]
        tensor_list.append(cls_emb)
        seg_list.append(0)
        for frame in range(len(src)):
            if frame % 2 == 0:
                seg_id = 0
            else:
                seg_id = 1
            tensor_list.append(sep_emb)
            seg_list.append(seg_id)
            for joint in range(len(src[frame])):
                tensor_list.append(src[frame][joint].to(device))
                seg_list.append(seg_id)
        inputs_embeds = torch.cat(tensor_list, 0)
        assert inputs_embeds.shape[0] == len(seg_list)
        seg_tensor = torch.tensor(seg_list).to(device)
        seg_emb = segment_emb(seg_tensor)
        pos_emb = pos_enc(inputs_embeds).squeeze(0)
        embeddings = inputs_embeds.to(device) + seg_emb.to(device) + pos_emb.to(device)

        mask_src = (1 - (torch.all(inputs_embeds == pad_emb,dim=-1)*1))
        mask_src = mask_src.unsqueeze(dim=1).repeat(1, inputs_embeds.shape[0]).unsqueeze(dim=0)
        return  embeddings, seg_emb, mask_src, tgt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bert_data_path", default='/storage/hjchoi/BERTSUMFORHPE/bert_data')
    parser.add_argument("-batch_size", default=10, type=int)
    parser.add_argument("-device_id", default='0', type=int)
    parser.add_argument("-num_workers", default=0, type=int)
    #
    parser.add_argument("-pad_id", default=0, type=int)
    parser.add_argument("-sep_id", default=1, type=int)
    parser.add_argument("-weight_path", default='../model_save/embedding_weights/[new]nn_embedding_model.pt')
    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.device_id))
    torch.cuda.set_device(args.device_id)
    #
    dataset_object = joint_dataset(
        args=args,
        shuffle=True,
        is_test=False,
        device = device
    )

    train_loader = DataLoader(
        dataset_object,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    for i, batch in enumerate(train_loader):
        print(batch)
