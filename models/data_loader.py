import bisect
import gc
import glob
import random

import torch
import torch.nn as nn
from others.logging import logger
from tqdm import tqdm
from .encoder import PositionalEncoding
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.utils import download as _download
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import BertModel

def get_kobert_vocab(cachedir="./tmp/"):
    # Add BOS,EOS vocab
    vocab_info = get_tokenizer()
    # vocab_file = _download(
    #     vocab_info["url"], vocab_info["fname"], vocab_info["chksum"], cachedir=cachedir
    # )

    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(
        vocab_info, padding_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
    )

    return vocab_b_obj

class Batch(object):
    def _emb(self, data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        #
        device = torch.device('cuda:{}'.format(0))
        torch.cuda.set_device(0)
        #
        pad_id = 0
        sep_id = 1
        preweights = torch.load('./model_save/embedding_weights/nn_embedding.pt')
        emb = nn.Embedding.from_pretrained(preweights['weight'], freeze=True).to(device)
        sep_emb = emb(torch.tensor(sep_id).to(device)).unsqueeze(0)
        pad_emb = emb(torch.tensor(pad_id).to(device).unsqueeze(0))
        #
        emb_for_seg = nn.Embedding(2,768).to(device)
        pos_enc = PositionalEncoding(dropout=0.2,dim=768)
        #
        cls_emb = torch.zeros([1,768]).to(device)
        #
        tensor_list  = []
        seg_list = []
        for i in range(len(data)):
            if i % 2 == 0:
                seg_id = 0
            else:
                seg_id = 1
            tensor_list.append(cls_emb)
            seg_list.append(seg_id)
            tensor_list.append(sep_emb)
            seg_list.append(seg_id)
            for j in range(len(data[i])):
                if j > 1:
                    tensor_list.append(data[i][j].to(device))
                    seg_list.append(seg_id)
            # for padding
            # tensor_list.append(pad_emb * (width - len(data[i])))
            # seg_list.append(seg_id *(width-len(data[i])))
        inputs_embeds = torch.cat(tensor_list, 0)
        assert inputs_embeds.shape[0] == len(seg_list)
        seg_tensor = torch.tensor(seg_list).to(device)
        seg_emb = emb_for_seg(seg_tensor)
        pos_emb = pos_enc(inputs_embeds).squeeze(0)
        embeddings = inputs_embeds.to(device) + seg_emb.to(device) + pos_emb.to(device)


        return inputs_embeds, seg_emb, embeddings

    def _one_hot_label(self, tgt):
        import torch.nn.functional as F
        tgt = torch.tensor(tgt)
        label = F.one_hot(tgt, num_classes=27)
        return label

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data][0]
            pre_tgt = [x[1]-22 for x in data]

            src, segs, embeddings = self._emb(pre_src, pad_id=1)
            mask = torch.ones([1, src.shape[0]], dtype=bool)
            tgt = self._one_hot_label(pre_tgt)


            setattr(self, "src", src.to(device))
            setattr(self, "segs", segs.to(device))
            setattr(self, "tgt", tgt.to(device))
            setattr(self, "embeddings", embeddings.to(device))
            setattr(self, "mask", mask.to(device))

            if is_test:
                src_str = [x[-2] for x in data]
                setattr(self, "src_str", src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, "tgt_str", tgt_str)

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        while True:
            try:
                dataset = torch.load(pt_file,map_location='cpu')
                logger.info(
                    "Loading %s dataset from %s, number of examples: %d"
                    % (corpus_type, pt_file, len(dataset))
                )
                return dataset
            except EOFError:
                print(pt_file)

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + "*.pt"))
    if pts:
        if shuffle:
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        pt = args.bert_data_path + ".pth.tar"
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if count > 6:
        return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    if len(new) == 4:
        pass
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size, device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(
            args=self.args,
            dataset=self.cur_dataset,
            batch_size=self.batch_size,
            device=self.device,
            shuffle=self.shuffle,
            is_test=self.is_test,
        )


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, device=None, is_test=False, shuffle=False):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = False
        self.sample_ver = True
        self.sort_key = lambda x: len(x[1])


        self._iterations_this_epoch = 0
        if self.args.task == "abs":
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset) # self.dataset = df (it contains N frames, frame shuffle X)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        # ex : dataset in load_dataset function
        src = ex["src"]
        tgt = ex['tgt']

        if is_test:
            return src, tgt, # segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt,#  segs, clss, src_sent_labels

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        if self.sample_ver:
            ex = self.preprocess(data, self.is_test)
        else:
            for ex in data:
                if len(ex["src"]) == 0:
                    continue
                ex = self.preprocess(ex, self.is_test)
                if ex is None:
                    continue
        minibatch.append(ex)
        size_so_far = self.batch_size_fn(ex, len(minibatch))
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            if self.args.task == "abs":
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[0][0]))

            p_batch = self.batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:
                if len(b) == 0:
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return

