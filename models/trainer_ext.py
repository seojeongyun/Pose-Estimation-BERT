import os
import csv
import torch.nn as nn
import numpy as np
import time
import torch
import math
from einops import rearrange,repeat
from tqdm import tqdm
import distributed
from models.reporter_ext import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str
from torch.utils.data.dataloader import DataLoader
from Embedder.data_loader import Video_Loader
from Embedder.Embedder_API import Embedder
from models.data_loader_joint import joint_dataset
from models.encoder import PositionalEncoding
from models.MoE import JointGroupMoE, build_exercise_condition_map

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, config, device_id, model, optim, video_dataset=None):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.device_id)
    else:
        gpu_rank = 0
        n_gpu = 0

    print("gpu_rank %d" % gpu_rank)

    #tensorboard_log_dir = args.model_path
    #
    #writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    #
    #report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(
        args, config, model, optim, grad_accum_count, n_gpu, gpu_rank,
        video_dataset=video_dataset)

    # print(tr)
    if model:
        n_params = _tally_parameters(model)
        logger.info("* number of parameters: %d" % n_params)

    return trainer

class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(
            self, args, config, model, optim, grad_accum_count=1, n_gpu=1,
            gpu_rank=1, report_manager=None, video_dataset=None,
    ):
        # Basic attributes.
        self.args = args
        self.config = config
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.save_checkpoint_epoch = args.save_checkpoint_epoch
        self.model = model
        self.train_epoch = args.train_epoch
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.device = (
            torch.device('cpu')
            if args.device_id < 0
            else torch.device('cuda:{}'.format(args.device_id))
        )
        #
        self.video_dataset = (
            video_dataset
            if video_dataset is not None
            else Video_Loader(config=self.config, mode=self.args.mode)
        )
        mapping_dataset = self.video_dataset
        self.data_loader = self.get_dataloader(self.video_dataset, mode='train')
        if self.args.mode == 'train-valid':
            self.video_dataset = Video_Loader(config=self.config, mode='validate')
            self.valid_data_loader = self.get_dataloader(self.video_dataset, mode='valid')

        self.embedder = Embedder(self.config, mode=self.args.mode).to(self.device)
        #
        if self.embedder.optimizer is not None:
            self.embedder.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.embedder.optimizer,
                T_max=self.args.train_epoch,
                eta_min=1e-6
            )

        #
        self.condition_vocab, self.exercise_vocab = self.get_vocab()
        self.cond_int2str, self.ex_int2str = self.reverse_vocab()
        self.moe_enabled = getattr(self.args, 'moe', False)
        self.expert_type = getattr(self.args, 'expert_type', 'joint')
        if self.moe_enabled and self.expert_type == 'group':
            self.moe_expert_names = list(JointGroupMoE.GROUP_NAMES)
        elif self.moe_enabled and self.expert_type == 'joint':
            self.moe_expert_names = list(self.config.JOINTS_NAME)
        else:
            self.moe_expert_names = []
        exercise_condition_map = getattr(
            self.args, 'exercise_condition_map', None)
        if (exercise_condition_map is None
                and self.moe_enabled
                and self.expert_type == 'exercise'):
            exercise_condition_map = build_exercise_condition_map(
                mapping_dataset.videos,
                num_exercises=self.config.CLASS_NUM,
                num_conditions=self.config.NUM_CONDITIONS,
            )
        self.exercise_condition_map = exercise_condition_map
        if exercise_condition_map is not None:
            exercise_condition_mask = torch.zeros(
                self.config.CLASS_NUM,
                self.config.NUM_CONDITIONS,
                dtype=torch.bool,
                device=self.device,
            )
            for exercise_index, condition_indices in enumerate(
                    exercise_condition_map):
                exercise_condition_mask[
                    exercise_index, condition_indices] = True
            self.exercise_condition_mask = exercise_condition_mask
        else:
            self.exercise_condition_mask = None
        #
        # 260226 single label classification
        # self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        #
        # 260226 multi label classification (not use weighted loss)
        # self.loss = torch.nn.MultiLabelSoftMarginLoss()
        #
        self.exercise_loss, self.conditions_loss = self.get_loss()
        self.threshold = self.args.threshold
        #
        self.pos_enc = PositionalEncoding(dropout=0.2, dim=768).to(self.device)
        assert grad_accum_count > 0
        # Set model in training mode.
        if model:
            self.model.train()

    def get_loss(self):
        exercise_loss = torch.nn.CrossEntropyLoss()
        print("[Exercise Loss Setup] CrossEntropyLoss.")
        if self.args.weighted_loss:
            weight = self.args.weighted_loss_value
            pos_weight = torch.ones(self.config.NUM_CONDITIONS, device=self.device)
            pos_weight[:] = weight
            conditions_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
            #
            print(f"[Conditions Loss Setup] Weighted BCEWithLogitsLoss activated (pos_weight={weight}).")

        else:
            conditions_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            print("[Conditions Loss Setup] Standard BCEWithLogitsLoss activated (no class weighting).")
            #
        return exercise_loss, conditions_loss
    #
    def get_vocab(self):
        import pickle

        with open(self.config.TRAIN_CONDITION_VOCAB_PATH,'rb') as f:
            condition_vocab = pickle.load(f)

        with open(self.config.TRAIN_WORKOUT_VOCAB_PATH, 'rb') as f:
            exercise_vocab = pickle.load(f)
        return condition_vocab, exercise_vocab

    def reverse_vocab(self):
        cond_int2str = {v: k for k, v in self.condition_vocab.items()}
        ex_int2str = {v: k for k, v in self.exercise_vocab.items()}

        return cond_int2str, ex_int2str

    def save_joint_expert_usage(self, counts, sample_counts, epoch):
        counts = counts.detach().cpu()
        sample_counts = sample_counts.detach().cpu()
        output_dir = os.path.join(self.args.run_dir, 'moe_expert_usage')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, 'train_epoch_{:03d}.csv'.format(epoch))
        expert_names = [
            'expert_{:02d}_{}'.format(index, joint_name)
            for index, joint_name in enumerate(self.moe_expert_names)
        ]

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['epoch', 'exercise_index', 'exercise_name', 'sample_count',
                 'total_expert_selections'] + expert_names)
            for exercise_index in range(self.config.CLASS_NUM):
                raw_exercise_id = exercise_index + 22
                expert_counts = counts[exercise_index].tolist()
                writer.writerow([
                    epoch,
                    exercise_index,
                    self.ex_int2str.get(raw_exercise_id, str(raw_exercise_id)),
                    int(sample_counts[exercise_index].item()),
                    sum(expert_counts),
                ] + expert_counts)
        return output_path

    def save_exercise_expert_usage(
            self, counts, sample_counts, epoch, selection_type):
        counts = counts.detach().cpu()
        sample_counts = sample_counts.detach().cpu()
        output_dir = os.path.join(self.args.run_dir, 'moe_expert_usage')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            'valid_{}_epoch_{:03d}.csv'.format(selection_type, epoch),
        )
        expert_names = [
            'expert_{:02d}_{}'.format(
                expert_index,
                self.ex_int2str.get(
                    expert_index + 22, str(expert_index + 22)),
            )
            for expert_index in range(self.config.CLASS_NUM)
        ]

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow([
                'epoch',
                'ground_truth_exercise_index',
                'ground_truth_exercise_name',
                'sample_count',
                'total_expert_activations',
            ] + expert_names)
            for exercise_index in range(self.config.CLASS_NUM):
                raw_exercise_id = exercise_index + 22
                expert_counts = counts[exercise_index].tolist()
                writer.writerow([
                    epoch,
                    exercise_index,
                    self.ex_int2str.get(
                        raw_exercise_id, str(raw_exercise_id)),
                    int(sample_counts[exercise_index].item()),
                    sum(expert_counts),
                ] + expert_counts)
        return output_path

    def get_dataloader(self, dataset, mode):
        if mode == 'train':
            shuffle = True
        elif mode == 'valid':
            # shuffle = False
            shuffle = True

        video_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=self.config.WORKERS,
            pin_memory=True,
            collate_fn=dataset.collate_fn)

        return video_loader

    def build_targets(self, exercise_name, conditions):
        """Build labels and the sample-level reference mask for one batch."""
        batch_size = len(exercise_name)
        num_conditions = self.config.NUM_CONDITIONS
        exercise_targets = (
            torch.tensor(
                exercise_name, dtype=torch.long, device=self.device)
            - 22
        )
        condition_labels = torch.zeros(
            batch_size, num_conditions,
            device=self.device, dtype=torch.float32)
        observed_mask = torch.zeros(
            batch_size, num_conditions,
            device=self.device, dtype=torch.bool)
        condition_offset = 22 + self.config.CLASS_NUM
        flat = [
            (batch_index, int(raw_condition) - condition_offset,
             0.0 if flag else 1.0)
            for batch_index, sample_conditions in enumerate(conditions)
            for raw_condition, flag in sample_conditions
        ]
        if flat:
            rows, columns, values = zip(*flat)
            columns_cpu = torch.tensor(columns)
            if (columns_cpu.min().item() < 0
                    or columns_cpu.max().item() >= num_conditions):
                raise ValueError(
                    "condition index out of range: [{}, {}], D={}".format(
                        columns_cpu.min().item(),
                        columns_cpu.max().item(),
                        num_conditions))
            rows = torch.tensor(rows, device=self.device)
            columns = torch.tensor(columns, device=self.device)
            values = torch.tensor(
                values, device=self.device, dtype=torch.float32)
            condition_labels[rows, columns] = values
            observed_mask[rows, columns] = True

        if self.moe_enabled and self.expert_type == 'exercise':
            if self.exercise_condition_mask is None:
                raise RuntimeError(
                    "exercise_condition_mask is required for exercise MoE")
            unexpected = (
                observed_mask
                & ~self.exercise_condition_mask[exercise_targets]
            )
            if unexpected.any().item():
                bad = unexpected.nonzero(as_tuple=False)[0].tolist()
                raise ValueError(
                    "sample condition is absent from its GT exercise map: {}"
                    .format(bad))
        return exercise_targets, condition_labels, observed_mask

    @staticmethod
    def predicted_exercise_condition_counts(
            pred_cond, tgt_cond, mask_bool, exercise_matches):
        """Count condition metrics with strict exercise-level failure.

        A sample whose predicted exercise differs from its target exercise has
        every M_ref entry scored as incorrect. Matching samples retain the
        model's thresholded condition predictions.
        """
        strict_pred_cond = torch.where(
            exercise_matches.unsqueeze(1),
            pred_cond,
            ~tgt_cond,
        ) & mask_bool
        tp = (strict_pred_cond & tgt_cond).sum().item()
        fp = (strict_pred_cond & ~tgt_cond).sum().item()
        fn = (~strict_pred_cond & tgt_cond & mask_bool).sum().item()
        element_correct = (
            (strict_pred_cond == tgt_cond) & mask_bool
        ).sum().item()
        element_total = mask_bool.sum().item()
        return tp, fp, fn, element_correct, element_total

    def forward_condition_model(self, input_embs, segs, pad_mask,
                                exercise_targets, condition_labels,
                                condition_mask):
        exercise_logits, condition_logits, moe_info = self.model(
            input_embs,
            segs,
            pad_mask,
            exercise_targets=exercise_targets,
            condition_labels=condition_labels,
            condition_mask=condition_mask,
            threshold=self.threshold,
        )
        output_mask = moe_info.get('condition_mask')
        if (self.moe_enabled and self.expert_type == 'exercise'
                and output_mask is not None):
            # Exercise experts keep their variable-width local output heads.
            # The global carrier is used only to align those local outputs by
            # condition ID. Missing M_ref entries must be negative predictions,
            # not entries removed from evaluation or PR/AP analysis.
            condition_logits = condition_logits.masked_fill(
                ~output_mask.bool(), -30.0)
        return (
            exercise_logits, condition_logits, condition_mask, moe_info)

    def emb_(self, data, frame_pad_mask):
        """
        Convert embedder output into BERT-style input embeddings.

        Args:
            data: Tensor of shape [B, F, J, D]
                  B = batch size
                  F = number of frames
                  J = number of joints per frame
                  D = embedding dimension (e.g., 768)

        Returns:
            embeddings:   Tensor [B, S, D]
            seg_emb:      Tensor [B, S, D]
            padding_mask: Tensor broadcastable to attention shape
        """

        # Unpack dimensions
        B, F, J, D = data.shape
        device = data.device

        # ===========================================================
        # 1. Build input token embeddings
        #
        # For each frame:
        #   prepend one [SEP] token
        # Then flatten all frames into a single sequence.
        # Finally prepend one [CLS] token at the beginning.
        #
        # Final sequence length:
        #   S = 1 + F * (1 + J)
        # ===========================================================

        # Get [SEP] embedding vector of shape [D]
        sep_vec = self.embedder.embedding.weight[self.model.sep_id]

        # Expand to shape [B, F, 1, D]
        sep_expanded = sep_vec.view(1, 1, 1, D).expand(B, F, 1, D)

        # Concatenate [SEP] with joint embeddings per frame
        # Result shape: [B, F, 1 + J, D]
        frames_combined = torch.cat([sep_expanded, data], dim=2)

        # Flatten frame and token dimensions
        # Shape: [B, F * (1 + J), D]
        seq_flattened = frames_combined.reshape(B, -1, D)

        # Add [CLS] token at the beginning
        if hasattr(self.model, 'cls_emb'):
            cls_vec = self.model.cls_emb.to(device)
        else:
            cls_vec = torch.zeros(D, device=device)

        cls_expanded = cls_vec.view(1, 1, D).expand(B, 1, D)

        # Final token sequence
        # Shape: [B, 1 + F * (1 + J), D]
        if self.args.attach_cls_token_to_end_of_seqlen:
            inputs_embeds = torch.cat([cls_expanded, seq_flattened , cls_expanded], dim=1)
        else:
            inputs_embeds = torch.cat([cls_expanded, seq_flattened], dim=1)
        # ===========================================================
        # 2. Build segment embeddings
        #
        # Alternate segment IDs per frame:
        # frame 0 -> 0
        # frame 1 -> 1
        # frame 2 -> 0
        # etc.
        #
        # All tokens within the same frame share the same segment ID.
        # [CLS] token uses segment ID 0.
        # ===========================================================

        # Frame pattern: [0, 1, 0, 1, ...]
        frame_pattern = torch.arange(F, device=device) % 2

        # Expand per frame to match (1 + J) tokens
        # Shape: [F * (1 + J)]
        seg_ids_flat = frame_pattern.unsqueeze(-1).expand(F, 1 + J).reshape(-1)

        # Segment ID for [CLS]
        cls_seg_id = torch.zeros(1, dtype=torch.long, device=device)

        # Full segment IDs for the batch
        # Shape: [B, S]
        if self.args.attach_cls_token_to_end_of_seqlen:
            full_seg_ids = torch.cat([cls_seg_id, seg_ids_flat, cls_seg_id]).unsqueeze(0).expand(B, -1)
        else:
            full_seg_ids = torch.cat([cls_seg_id, seg_ids_flat]).unsqueeze(0).expand(B, -1)

        # Segment embedding lookup
        # Shape: [B, S, D]
        seg_emb = self.model.segment_emb(full_seg_ids)

        # ===========================================================
        # 3. Apply positional encoding
        #
        # Standard transformer style:
        #   x_scaled = x * sqrt(D)
        #   x_pe = x_scaled + positional_encoding
        #   apply dropout
        #
        # Final embedding is:
        #   inputs + segment + positional
        # ===========================================================

        # Add positional encoding without rescaling or duplicating the input embeddings.
        seq_len = inputs_embeds.size(1)
        pos_emb = self.pos_enc.pe[:, :seq_len].to(device)
        #
        embeddings = self.model.input_layer_norm(inputs_embeds + seg_emb + pos_emb)
        embeddings = self.model.input_dropout(embeddings)

        # ===========================================================
        # 4. Build padding mask
        #
        # A token is considered padding if its embedding equals pad token embedding.
        #
        # mask_src:
        #   1 for real token
        #   0 for padding
        #
        # Output mask is reshaped for attention usage.
        # ===========================================================

        # 실제 frame이면 SEP와 joint 20개 모두 valid
        # padding frame이면 SEP와 joint 20개 모두 invalid
        frame_token_valid = (~frame_pad_mask).unsqueeze(-1).expand(B, F, 1 + J).reshape(B, F * (1 + J))

        # 시작 CLS는 항상 valid
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=device)

        if self.args.attach_cls_token_to_end_of_seqlen:
            mask_src = torch.cat([cls_valid, frame_token_valid, cls_valid], dim=1)
        else:
            mask_src = torch.cat([cls_valid, frame_token_valid], dim=1)

        return embeddings, full_seg_ids, mask_src

    def train(self, device, train_steps, valid_iter_fct=None, valid_steps=-1):
        logger.info("Start training...")        #
        #
        best_perf = 0.0
        best_model = False
        for epoch in range(self.train_epoch):
            print("\n============== [{}][Model: TRAIN_MODE] TRAIN START ==============".format(epoch+1))
            self.model.train()
            self.embedder.train()
            #
            Train_total_loss = 0.0
            Train_exercise_ce_loss = 0.0
            Train_condition_bce_loss = 0.0
            Train_moe_router_loss = 0.0
            Train_moe_aux_loss = 0.0
            Train_arcface_loss = 0.0

            exercise_classification_acc = 0
            #
            condition_tp = 0
            condition_fp = 0
            condition_fn = 0
            #
            total_samples_seen = 0
            #
            pos_cnt = 0.0
            mask_cnt = 0.0
            #
            start = time.time()

            train_expert_usage = None
            train_exercise_sample_counts = None
            if self.moe_enabled and self.expert_type in ('joint', 'group'):
                train_expert_usage = torch.zeros(
                    self.config.CLASS_NUM,
                    len(self.moe_expert_names),
                    dtype=torch.long,
                    device=self.device,
                )
                train_exercise_sample_counts = torch.zeros(
                    self.config.CLASS_NUM,
                    dtype=torch.long,
                    device=self.device,
                )

            if self.config.USE_ARCFACE:
                self.embedder.losses.reset()

            for step, (videos, exercise_name, conditions) in enumerate(self.data_loader):
                step_start = time.time()
                #
                ex_idx, cond_label, cond_mask = self.build_targets(
                    exercise_name, conditions)

                # forward
                output, frame_pad_mask, arcface_loss_sum, arcface_count = self.embedder(videos)

                #
                input_embs, segs, pad_mask = self.emb_(output, frame_pad_mask)

                #
                (ex_logits, cond_logits, evaluation_cond_mask,
                 moe_info) = self.forward_condition_model(
                    input_embs, segs, pad_mask,
                    ex_idx, cond_label, cond_mask)

                if train_expert_usage is not None:
                    selected_experts = moe_info['selected_experts'].detach()
                    exercise_indices = ex_idx.detach()
                    num_experts = train_expert_usage.size(1)
                    flat_indices = (
                        exercise_indices.unsqueeze(1) * num_experts
                        + selected_experts
                    )
                    train_expert_usage += torch.bincount(
                        flat_indices.reshape(-1),
                        minlength=self.config.CLASS_NUM * num_experts,
                    ).reshape(self.config.CLASS_NUM, num_experts)
                    train_exercise_sample_counts += torch.bincount(
                        exercise_indices,
                        minlength=self.config.CLASS_NUM,
                    )

                # loss (CE + BCEWithLogits)
                exercise_loss = self.exercise_loss(ex_logits, ex_idx)
                condition_loss = self.conditions_loss(cond_logits, cond_label)
                cond_mask_f = evaluation_cond_mask.float()
                condition_loss = (condition_loss * cond_mask_f).sum() / (cond_mask_f.sum() + 1e-8)

                # 실제 total_loss에 들어가는 가중 loss
                weighted_exercise_loss = self.args.ex_loss_weight * exercise_loss
                weighted_condition_loss = self.args.cond_loss_weight * condition_loss
                if moe_info.get('router_needs_supervision', False):
                    weighted_moe_router_loss = (
                        getattr(self.args, 'router_loss_weight', 1.0)
                        * self.exercise_loss(
                            moe_info['router_logits'], ex_idx))
                else:
                    weighted_moe_router_loss = cond_logits.new_zeros(())
                weighted_moe_aux_loss = (
                    getattr(self.args, 'aux_loss_weight', 0.0)
                    * moe_info['aux_loss'])

                loss = (weighted_exercise_loss + weighted_condition_loss
                        + weighted_moe_router_loss + weighted_moe_aux_loss)

                if self.config.USE_ARCFACE:
                    # arcface_loss /= (len(self.config.JOINTS_NAME) * self.config.MAX_FRAMES) # here
                    arcface_loss = arcface_loss_sum / (arcface_count + 1e-8)
                    total_loss = loss + arcface_loss

                else:
                    total_loss = loss

                Train_total_loss += total_loss.item()
                Train_exercise_ce_loss += weighted_exercise_loss.item()
                Train_condition_bce_loss += weighted_condition_loss.item()
                Train_moe_router_loss += weighted_moe_router_loss.item()
                Train_moe_aux_loss += weighted_moe_aux_loss.item()

                if self.config.USE_ARCFACE:
                    Train_arcface_loss += arcface_loss.item()

                # exercise acc
                pred_ex = torch.argmax(ex_logits, dim=1)
                exercise_classification_acc += (pred_ex == ex_idx).sum().item()

                # condition metrics
                mask_bool = evaluation_cond_mask.bool()
                pred_cond = (torch.sigmoid(cond_logits) > self.threshold) & mask_bool
                tgt_cond = (cond_label > 0.5) & mask_bool

                # DEBUG (before applying masking)
                # if step == 0:
                #     # ---- Exercise ----
                #     pred0 = int(pred_ex[0].item())  # 0~40
                #     tgt0 = int(ex_idx[0].item())  # 0~40
                #
                #     pred_raw = pred0 + 22
                #     tgt_raw = tgt0 + 22
                #
                #     print(f"Expected: {self.ex_int2str[tgt_raw]}\nPredicted: {self.ex_int2str[pred_raw]}")
                #
                #     # ---- Condition ----
                #     p_local = pred_cond[0].nonzero(as_tuple=True)[0]  # 0~NUM_COND-1
                #     t_local = tgt_cond[0].nonzero(as_tuple=True)[0]
                #
                #     # local -> raw (63~159)
                #     p_raw = p_local + (self.config.CLASS_NUM + 22)  # +63
                #     t_raw = t_local + (self.config.CLASS_NUM + 22)
                #
                #     p_condition_lst = [self.cond_int2str[int(idx.item())] for idx in p_raw]
                #     t_condition_lst = [self.cond_int2str[int(idx.item())] for idx in t_raw]
                #
                #     print("Expected: " + ", ".join(t_condition_lst))
                #     print("Predicted: " + ", ".join(p_condition_lst))

                # Calc tp, fp, fn
                tp = (pred_cond & tgt_cond).sum().item()
                fp = (pred_cond & ~tgt_cond).sum().item()
                fn = (~pred_cond & tgt_cond).sum().item()

                condition_tp += tp
                condition_fp += fp
                condition_fn += fn

                #
                batch_size = cond_label.size(0)
                total_samples_seen += batch_size
                #

                # cond_label: [B,D], cond_mask: [B,D] (bool)
                pos_cnt += (cond_label[cond_mask] > 0.5).sum().item()
                mask_cnt += cond_mask.sum().item()

                if step % 1000 == 0 and step != 0:
                    ex_acc = 100.0 * exercise_classification_acc / total_samples_seen
                    precision = condition_tp / (condition_tp + condition_fp + 1e-8)
                    recall = condition_tp / (condition_tp + condition_fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    if self.config.USE_ARCFACE:
                        print(f"[TRAIN][Step {step}] "
                              f"Total_Loss: {Train_total_loss / (step + 1):.4f} | "
                              f"Type_Loss: {Train_exercise_ce_loss / (step + 1):.4f} | "
                              f"Condition_Loss: {Train_condition_bce_loss / (step + 1):.4f} | "
                              f"MoE_Router_Loss: {Train_moe_router_loss / (step + 1):.4f} | "
                              f"MoE_Aux_Loss: {Train_moe_aux_loss / (step + 1):.4f} | "
                              f"Arcface_Loss: {Train_arcface_loss / (step + 1):.4f} | "
                              f"Exercise_ACC: {ex_acc:.2f}% | "
                              f"P: {precision:.4f} | "
                              f"R: {recall:.4f} | "
                              f"F1: {f1:.4f}")
                    else:
                        print(f"[TRAIN][Step {step}] "
                              f"Total_Loss: {Train_total_loss / (step + 1):.4f} | "
                              f"Type_Loss: {Train_exercise_ce_loss / (step + 1):.4f} | "
                              f"Condition_Loss: {Train_condition_bce_loss / (step + 1):.4f} | "
                              f"MoE_Router_Loss: {Train_moe_router_loss / (step + 1):.4f} | "
                              f"MoE_Aux_Loss: {Train_moe_aux_loss / (step + 1):.4f} | "
                              f"Exercise_ACC: {ex_acc:.2f}% | "
                              f"P: {precision:.4f} | "
                              f"R: {recall:.4f} | "
                              f"F1: {f1:.4f}")
                #
                # UPDATE
                if self.embedder.optimizer is not None:
                    self.embedder.optimizer.zero_grad()
                self.optim.optimizer.zero_grad()

                total_loss.backward()

                #
                if self.embedder.optimizer is not None:
                    self.embedder.optimizer.step()
                    self.embedder.restore_frozen_special_embeddings()

                # self.optim.optimizer.step()
                self.optim.optimizer.step()

                step_end = time.time()
                step_time = step_end - step_start
                # print(f"[Step {step}/{len(self.data_loader)}] Step Time: {step_time:.4f} sec")
            #
            end = time.time()
            epoch_time = end - start
            #
            num_train_batches = len(self.data_loader)
            avg_train_loss = Train_total_loss / num_train_batches
            avg_exercise_ce_loss = Train_exercise_ce_loss / num_train_batches
            avg_condition_bce_loss = Train_condition_bce_loss / num_train_batches
            avg_moe_router_loss = Train_moe_router_loss / num_train_batches
            avg_moe_aux_loss = Train_moe_aux_loss / num_train_batches
            avg_arcface_contribution = Train_arcface_loss / num_train_batches
            Train_exercise_cls_accuracy = 100.0 * exercise_classification_acc / total_samples_seen
            #
            precision = condition_tp / (condition_tp + condition_fp + 1e-8)
            recall = condition_tp / (condition_tp + condition_fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            #
            if self.config.USE_ARCFACE:
                print(
                    f"[TRAIN][Epoch {epoch + 1}] "
                    f"Avg_Loss: {avg_train_loss:.6f} | "
                    f"Avg_Type_Loss: {avg_exercise_ce_loss:.4f} | "
                    f"Avg_Condition_Loss: {avg_condition_bce_loss:.4f} | "
                    f"Avg_MoE_Router_Loss: {avg_moe_router_loss:.4f} | "
                    f"Avg_MoE_Aux_Loss: {avg_moe_aux_loss:.4f} | "
                    f"Avg_ArcFace_Loss: {avg_arcface_contribution:.4f} | "
                    f"Exercise_ACC: {Train_exercise_cls_accuracy:.2f}% | "
                    f"P: {precision:.4f} | "
                    f"R: {recall:.4f} | "
                    f"F1: {f1:.4f}"
                )
            else:
                print(f"[TRAIN][Epoch {1 + epoch}] "
                      f"Avg_Loss: {avg_train_loss:.6f} | "
                      f"Avg_Type_Loss: {avg_exercise_ce_loss:.4f} | "
                      f"Avg_Condition_Loss: {avg_condition_bce_loss:.4f} | "
                      f"Avg_MoE_Router_Loss: {avg_moe_router_loss:.4f} | "
                      f"Avg_MoE_Aux_Loss: {avg_moe_aux_loss:.4f} | "
                      f"Exercise_ACC: {Train_exercise_cls_accuracy:.2f}% | "
                      f"P: {precision:.4f} | "
                      f"R: {recall:.4f} | "
                      f"F1: {f1:.4f}"
                      )
            if train_expert_usage is not None:
                expert_usage_path = self.save_joint_expert_usage(
                    train_expert_usage,
                    train_exercise_sample_counts,
                    epoch + 1,
                )
                print(
                    '[TRAIN][Epoch {}] Joint expert usage: {}'.format(
                        epoch + 1, expert_usage_path))
            # throughput
            throughput = len(self.data_loader.dataset) / epoch_time  # samples/sec
            print(f"Throughput: {throughput:.1f} samples/s")
            #
            # elapsed time
            hours, rem = divmod(epoch_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Elapsed time per epoch: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
            #
            # Scheduler
            if self.embedder.scheduler is not None:
                self.embedder.scheduler.step()

            # CyclicLR을 사용할 땐, 매 epoch마다 스케줄러를 실행하는 것이 아니라,
            # optimizer가 업데이트 될 때 마다 실행해줘야 한다.
            if self.args.use_scheduler:
                self.optim.scheduler.step()
                # DEBUG - SCHEDULER
                lr_from_optimizer = self.optim.optimizer.param_groups[0]['lr']
                print("current_lr: {:.6f}".format(lr_from_optimizer))

            if self.args.mode == 'train-valid':
                print("\n============== [Model: EVAL_MODE] VALID START ==============")
                self.model.eval()
                self.embedder.eval()

                with torch.no_grad():
                    Valid_exercise_ce_loss = 0.0
                    Valid_condition_bce_loss = 0.0

                    exercise_classification_acc = 0
                    #
                    condition_tp = 0
                    condition_fp = 0
                    condition_fn = 0
                    predicted_condition_tp = 0
                    predicted_condition_fp = 0
                    predicted_condition_fn = 0
                    #
                    condition_element_correct = 0
                    condition_element_total = 0
                    predicted_condition_element_correct = 0
                    predicted_condition_element_total = 0
                    expert_output_condition_entries = 0
                    #
                    valid_samples_seen = 0

                    # validation 전체 배치의 condition 확률/정답/applicable mask와
                    # 정답 운동을 저장한다. label 1은 '잘못된 자세'(positive)이다.
                    pr_probabilities = []
                    pr_targets = []
                    pr_masks = []
                    pr_exercise_targets = []

                    # 새 best model에서 운동별 AP와 ROC를 계산하기 위한 값.
                    # 계산과 파일 생성은 best 갱신 시에만 수행한다.
                    exercise_probabilities = []
                    exercise_targets = []

                    valid_router_topk_expert_usage = None
                    valid_final_selected_expert_usage = None
                    valid_exercise_sample_counts = None
                    if self.moe_enabled and self.expert_type == 'exercise':
                        exercise_usage_shape = (
                            self.config.CLASS_NUM,
                            self.config.CLASS_NUM,
                        )
                        valid_router_topk_expert_usage = torch.zeros(
                            exercise_usage_shape,
                            dtype=torch.long,
                            device=self.device,
                        )
                        valid_final_selected_expert_usage = torch.zeros(
                            exercise_usage_shape,
                            dtype=torch.long,
                            device=self.device,
                        )
                        valid_exercise_sample_counts = torch.zeros(
                            self.config.CLASS_NUM,
                            dtype=torch.long,
                            device=self.device,
                        )

                    for step, (videos, exercise_name, conditions) in enumerate(self.valid_data_loader):

                        ex_idx, cond_label, cond_mask = self.build_targets(
                            exercise_name, conditions)
                        # ---- forward ----
                        output, frame_pad_mask, _, _ = self.embedder(videos)

                        input_embs, segs, pad_mask = self.emb_(output, frame_pad_mask)

                        (ex_logits, cond_logits, evaluation_cond_mask,
                         moe_info) = self.forward_condition_model(
                            input_embs, segs, pad_mask,
                            ex_idx, cond_label, cond_mask)

                        if valid_router_topk_expert_usage is not None:
                            ground_truth_exercises = ex_idx.detach()
                            router_topk_experts = moe_info[
                                'topk_indices'].detach()
                            final_selected_experts = moe_info[
                                'selected_experts'].detach()
                            num_experts = self.config.CLASS_NUM

                            router_flat_indices = (
                                ground_truth_exercises.unsqueeze(1)
                                * num_experts
                                + router_topk_experts
                            )
                            valid_router_topk_expert_usage += torch.bincount(
                                router_flat_indices.reshape(-1),
                                minlength=num_experts * num_experts,
                            ).reshape(num_experts, num_experts)

                            final_flat_indices = (
                                ground_truth_exercises * num_experts
                                + final_selected_experts
                            )
                            valid_final_selected_expert_usage += (
                                torch.bincount(
                                    final_flat_indices,
                                    minlength=num_experts * num_experts,
                                ).reshape(num_experts, num_experts)
                            )
                            valid_exercise_sample_counts += torch.bincount(
                                ground_truth_exercises,
                                minlength=num_experts,
                            )

                        # ---- Exercise ACC ----
                        pred_ex = torch.argmax(ex_logits, dim=1)
                        exercise_classification_acc += (pred_ex == ex_idx).sum().item()

                        if self.args.exercise_ap_roc_analysis:
                            exercise_probabilities.append(
                                torch.softmax(ex_logits, dim=1).cpu()
                            )
                            exercise_targets.append(ex_idx.cpu())

                        # ---- Condition metrics ----
                        mask_bool = evaluation_cond_mask.bool()
                        cond_probabilities = torch.sigmoid(cond_logits)

                        pred_cond = (cond_probabilities > self.threshold) & mask_bool
                        tgt_cond = (cond_label > 0.5) & mask_bool

                        # 운동별/condition별 PR 분석에 threshold 적용 전
                        # sigmoid 확률을 사용한다. full [B, D] 구조를 유지해야
                        # 어느 운동의 어느 condition인지 복원할 수 있다.
                        if self.args.condition_pr_analysis:
                            pr_sample_mask = mask_bool.any(dim=1)
                            if pr_sample_mask.any().item():
                                pr_probabilities.append(
                                    cond_probabilities[pr_sample_mask].cpu())
                                pr_targets.append(
                                    cond_label[pr_sample_mask].cpu())
                                pr_masks.append(
                                    mask_bool[pr_sample_mask].cpu())
                                pr_exercise_targets.append(
                                    ex_idx[pr_sample_mask].cpu())

                        if step == 0:
                            # ---- Exercise debug ----
                            pred_raw = int(pred_ex[0].item()) + 22
                            tgt_raw = int(ex_idx[0].item()) + 22

                            print(f"Expected: {self.ex_int2str[tgt_raw]}\nPredicted: {self.ex_int2str[pred_raw]}")

                            # ---- Condition debug ----
                            mask0 = mask_bool[0]  # [D]
                            p_local = (pred_cond[0] & mask0).nonzero(as_tuple=True)[0]
                            t_local = (tgt_cond[0] & mask0).nonzero(as_tuple=True)[0]

                            # local -> raw = local + CLASS_NUM + 22
                            p_raw = p_local + (self.config.CLASS_NUM + 22)
                            t_raw = t_local + (self.config.CLASS_NUM + 22)

                            p_condition_lst = [self.cond_int2str[int(idx.item())] for idx in p_raw]
                            t_condition_lst = [self.cond_int2str[int(idx.item())] for idx in t_raw]

                            print("Expected: " + ", ".join(t_condition_lst))
                            print("Predicted: " + ", ".join(p_condition_lst))

                        tp = (pred_cond & tgt_cond).sum().item()
                        fp = (pred_cond & ~tgt_cond).sum().item()
                        fn = (~pred_cond & tgt_cond).sum().item()

                        condition_tp += tp
                        condition_fp += fp
                        condition_fn += fn

                        element_correct = (
                                (pred_cond == tgt_cond)
                                & mask_bool
                        ).sum().item()

                        element_total = mask_bool.sum().item()

                        condition_element_correct += element_correct
                        condition_element_total += element_total
                        (predicted_tp, predicted_fp, predicted_fn,
                         predicted_element_correct,
                         predicted_element_total) = (
                            self.predicted_exercise_condition_counts(
                                pred_cond,
                                tgt_cond,
                                mask_bool,
                                pred_ex == ex_idx,
                            )
                        )
                        predicted_condition_tp += predicted_tp
                        predicted_condition_fp += predicted_fp
                        predicted_condition_fn += predicted_fn
                        predicted_condition_element_correct += (
                            predicted_element_correct)
                        predicted_condition_element_total += (
                            predicted_element_total)
                        output_mask = moe_info.get('condition_mask')
                        if output_mask is None:
                            output_mask = cond_mask
                        expert_output_condition_entries += (
                            output_mask.bool() & cond_mask
                        ).sum().item()
                        #
                        batch_size = cond_label.size(0)
                        valid_samples_seen += batch_size
                    #
                    Valid_exercise_cls_accuracy = 100.0 * exercise_classification_acc / valid_samples_seen
                    precision = condition_tp / (condition_tp + condition_fp + 1e-8)
                    recall = condition_tp / (condition_tp + condition_fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    element_wise_binary_accuracy = condition_element_correct / max(condition_element_total, 1)
                    predicted_precision = (
                        predicted_condition_tp
                        / (predicted_condition_tp
                           + predicted_condition_fp + 1e-8)
                    )
                    predicted_recall = (
                        predicted_condition_tp
                        / (predicted_condition_tp
                           + predicted_condition_fn + 1e-8)
                    )
                    predicted_f1 = (
                        2 * predicted_precision * predicted_recall
                        / (predicted_precision + predicted_recall + 1e-8)
                    )
                    predicted_element_wise_binary_accuracy = (
                        predicted_condition_element_correct
                        / max(predicted_condition_element_total, 1)
                    )
                    #
                    print('[VALID] EX_CLS_Acc: {:.2f}%'.format(Valid_exercise_cls_accuracy))
                    print(
                        '[VALID] P: {:.4f}, R: {:.4f}, F1: {:.4f}, '
                        'Element Acc: {:.4f}'
                        .format(precision, recall, f1,
                                element_wise_binary_accuracy))
                    print(
                        '[VALID][Predicted Exercise] P: {:.4f}, R: {:.4f}, '
                        'F1: {:.4f}, Element Acc: {:.4f}'
                        .format(
                            predicted_precision,
                            predicted_recall,
                            predicted_f1,
                            predicted_element_wise_binary_accuracy))
                    print(
                        '[VALID] Selected_Condition_Coverage: '
                        '{:.4f} ({:,}/{:,})'.format(
                            expert_output_condition_entries
                            / max(condition_element_total, 1),
                            expert_output_condition_entries,
                            condition_element_total))
                    if valid_router_topk_expert_usage is not None:
                        router_usage_path = self.save_exercise_expert_usage(
                            valid_router_topk_expert_usage,
                            valid_exercise_sample_counts,
                            epoch + 1,
                            'router_topk_expert_usage',
                        )
                        final_usage_path = self.save_exercise_expert_usage(
                            valid_final_selected_expert_usage,
                            valid_exercise_sample_counts,
                            epoch + 1,
                            'final_selected_expert_usage',
                        )
                        print(
                            '[VALID][Epoch {}] Exercise router Top-K usage: {}'
                            .format(epoch + 1, router_usage_path))
                        print(
                            '[VALID][Epoch {}] Exercise final expert usage: {}'
                            .format(epoch + 1, final_usage_path))
                    if self.args.condition_pr_analysis:
                        if pr_probabilities:
                            from others.condition_pr import save_condition_pr_analysis

                            # epoch은 코드 내부에서 0부터 시작하므로 폴더명은 epoch + 1 사용
                            pr_epoch_dir = os.path.join(
                                self.args.condition_pr_output,
                                'epoch_{:03d}'.format(epoch + 1),
                            )

                            pr_summary = save_condition_pr_analysis(
                                torch.cat(pr_probabilities, dim=0).numpy(),
                                torch.cat(pr_targets, dim=0).numpy(),
                                torch.cat(pr_masks, dim=0).numpy(),
                                torch.cat(pr_exercise_targets, dim=0).numpy(),
                                output_dir=pr_epoch_dir,
                                epoch=epoch + 1,
                                class_num=self.config.CLASS_NUM,
                                exercise_names_by_raw_id=self.ex_int2str,
                                condition_names_by_raw_id=self.cond_int2str,
                            )

                            # print(
                            #     '[VALID][Epoch {}] Condition PR analysis: {} | output: {}'
                            #     .format(epoch + 1, pr_summary, pr_epoch_dir)
                            # )
                        else:
                            print(
                                '[VALID][Epoch {}] Condition PR analysis skipped: '
                                'no valid condition targets'.format(epoch + 1)
                            )

                    ckpt = {
                        'epoch': epoch,
                        'embedder_state_dict': self.embedder.state_dict(),
                        'model_state_dict': self.model.state_dict(),
                        'embedder_optimizer_state_dict': self.embedder.optimizer.state_dict() if self.embedder.optimizer is not None else None,
                        'model_optimizer_state_dict': self.optim.optimizer.state_dict(),
                        'embedder_scheduler_state_dict': self.embedder.scheduler.state_dict() if self.embedder.scheduler is not None else None,
                        'model_scheduler_state_dict': self.optim.scheduler.state_dict() if self.args.use_scheduler else None,
                        'exercise_condition_map': self.exercise_condition_map,
                        'moe_config': {
                            'moe': self.moe_enabled,
                            'expert_type': self.expert_type,
                            'expert_mode': getattr(
                                self.args, 'expert_mode', 'baseline'),
                            'tcn_conv_type': getattr(
                                self.args, 'tcn_conv_type', 'conv1d'),
                            'tcn_channels': getattr(
                                self.args, 'tcn_channels', 128),
                            'top_k': getattr(self.args, 'top_k', 3),
                            'expert_dim': getattr(
                                self.args, 'expert_dim', 256),
                            'router_loss_weight': getattr(
                                self.args, 'router_loss_weight', 1.0),
                            'aux_loss_weight': getattr(
                                self.args, 'aux_loss_weight', 0.01),
                        },
                    }
                    #
                    if Valid_exercise_cls_accuracy > best_perf:
                        best_perf = Valid_exercise_cls_accuracy
                        best_model = True
                    else:
                        best_model = False

                    if best_model:
                        if self.args.exercise_ap_roc_analysis:
                            from others.exercise_ap_roc import save_exercise_ap_roc_analysis

                            exercise_class_names = [
                                self.ex_int2str.get(
                                    class_index + 22,
                                    'class_{}'.format(class_index),
                                )
                                for class_index in range(self.config.CLASS_NUM)
                            ]
                            exercise_summary = save_exercise_ap_roc_analysis(
                                probabilities=torch.cat(
                                    exercise_probabilities,
                                    dim=0,
                                ).numpy(),
                                targets=torch.cat(
                                    exercise_targets,
                                    dim=0,
                                ).numpy(),
                                class_names=exercise_class_names,
                                output_dir=self.args.exercise_ap_roc_output,
                                epoch=epoch + 1,
                                validation_accuracy=Valid_exercise_cls_accuracy,
                            )
                            # print(
                            #     '[VALID][Epoch {}] Best exercise AP/ROC updated: {}'
                            #     .format(epoch + 1, exercise_summary)
                            # )

                        os.makedirs(self.args.multi_cls_model_path, exist_ok=True)
                        torch.save(ckpt, os.path.join(self.args.multi_cls_model_path, 'best_model_ckpt.pt'))
                        print('Best Model save...')

                # Final Epoch Checkpoint
                os.makedirs(self.args.multi_cls_model_path, exist_ok=True)
                torch.save(ckpt, os.path.join(self.args.multi_cls_model_path, 'final_epoch_ckpt.pt'))
                print('Final Epoch Model save...')

                # Load code
                # ckpt = torch.load(save_path, map_location=self.device)
                #
                # self.embedder.load_state_dict(ckpt['embedder_state_dict'])
                # self.model.load_state_dict(ckpt['model_state_dict'])
                #
                # if self.config.USE_ARCFACE and ckpt['embedder_optimizer_state_dict'] is not None:
                #     self.embedder.optimizer.load_state_dict(ckpt['embedder_optimizer_state_dict'])
                #
                # self.optim.optimizer.load_state_dict(ckpt['model_optimizer_state_dict'])
                #
                # if self.config.USE_ARCFACE and ckpt['embedder_scheduler_state_dict'] is not None:
                #     self.embedder.scheduler.load_state_dict(ckpt['embedder_scheduler_state_dict'])
                #
                # if self.args.use_scheduler and ckpt['model_scheduler_state_dict'] is not None:
                #     self.optim.scheduler.load_state_dict(ckpt['model_scheduler_state_dict'])
                #
                # start_epoch = ckpt['epoch'] + 1

    def validate(self, video_loader, video_dataset, step=0):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        stats = Statistics()

        self.model.eval()
        self.embedder.eval()

        with torch.no_grad():
            exercise_classification_acc = 0
            #
            condition_tp = 0
            condition_fp = 0
            condition_fn = 0
            predicted_condition_tp = 0
            predicted_condition_fp = 0
            predicted_condition_fn = 0
            #
            condition_element_correct = 0
            condition_element_total = 0
            predicted_condition_element_correct = 0
            predicted_condition_element_total = 0
            #
            selected_condition_entries = 0
            ground_truth_condition_entries = 0
            #
            valid_samples_seen = 0
            pr_probabilities = []
            pr_targets = []
            pr_masks = []
            pr_exercise_targets = []
            for step, (videos, exercise_name, conditions) in enumerate(video_loader):

                ex_idx, cond_label, cond_mask = self.build_targets(
                    exercise_name, conditions)
                # ---- forward ----
                output, frame_pad_mask, _, _ = self.embedder(videos)
                input_embs, segs, pad_mask = self.emb_(
                    output, frame_pad_mask)
                (ex_logits, cond_logits, evaluation_cond_mask,
                 moe_info) = self.forward_condition_model(
                    input_embs, segs, pad_mask,
                    ex_idx, cond_label, cond_mask)

                # ---- Exercise ACC ----
                pred_ex = torch.argmax(ex_logits, dim=1)
                exercise_classification_acc += (pred_ex == ex_idx).sum().item()

                # ---- Condition metrics ----
                mask_bool = evaluation_cond_mask.bool()
                pred_cond = (torch.sigmoid(cond_logits) > self.threshold) & mask_bool
                tgt_cond = (cond_label > 0.5) & mask_bool

                if self.args.condition_pr_analysis:
                    pr_sample_mask = mask_bool.any(dim=1)
                    if pr_sample_mask.any().item():
                        pr_probabilities.append(
                            torch.sigmoid(
                                cond_logits[pr_sample_mask]).cpu())
                        pr_targets.append(
                            cond_label[pr_sample_mask].cpu())
                        pr_masks.append(mask_bool[pr_sample_mask].cpu())
                        pr_exercise_targets.append(
                            ex_idx[pr_sample_mask].cpu())

                if step % 10 == 0:
                    print('============ step {} =========='.format(step))
                    # ---- Exercise debug ----
                    pred_raw = int(pred_ex[0].item()) + 22
                    tgt_raw = int(ex_idx[0].item()) + 22

                    print(f"Expected: {self.ex_int2str[tgt_raw]}\nPredicted: {self.ex_int2str[pred_raw]}")

                    # ---- Condition debug ----
                    mask0 = mask_bool[0]  # [D]
                    p_local = (pred_cond[0] & mask0).nonzero(as_tuple=True)[0]
                    t_local = (tgt_cond[0] & mask0).nonzero(as_tuple=True)[0]

                    # local -> raw = local + CLASS_NUM + 22
                    p_raw = p_local + (self.config.CLASS_NUM + 22)
                    t_raw = t_local + (self.config.CLASS_NUM + 22)

                    p_condition_lst = [self.cond_int2str[int(idx.item())] for idx in p_raw]
                    t_condition_lst = [self.cond_int2str[int(idx.item())] for idx in t_raw]

                    print("Expected: " + ", ".join(t_condition_lst))
                    print("Predicted: " + ", ".join(p_condition_lst))

                tp = (pred_cond & tgt_cond).sum().item()
                fp = (pred_cond & ~tgt_cond).sum().item()
                fn = (~pred_cond & tgt_cond).sum().item()

                condition_tp += tp
                condition_fp += fp
                condition_fn += fn
                condition_element_correct += (
                    (pred_cond == tgt_cond) & mask_bool
                ).sum().item()
                condition_element_total += mask_bool.sum().item()
                (predicted_tp, predicted_fp, predicted_fn,
                 predicted_element_correct,
                 predicted_element_total) = (
                    self.predicted_exercise_condition_counts(
                        pred_cond,
                        tgt_cond,
                        mask_bool,
                        pred_ex == ex_idx,
                    )
                )
                predicted_condition_tp += predicted_tp
                predicted_condition_fp += predicted_fp
                predicted_condition_fn += predicted_fn
                predicted_condition_element_correct += (
                    predicted_element_correct)
                predicted_condition_element_total += predicted_element_total

                output_mask = moe_info.get('condition_mask')
                if output_mask is None:
                    output_mask = cond_mask
                selected_condition_entries += (
                    output_mask.bool() & cond_mask
                ).sum().item()
                ground_truth_condition_entries += cond_mask.sum().item()

                batch_size = cond_label.size(0)
                valid_samples_seen += batch_size
            #
            Valid_exercise_cls_accuracy = 100.0 * exercise_classification_acc / valid_samples_seen
            precision = condition_tp / (condition_tp + condition_fp + 1e-8)
            recall = condition_tp / (condition_tp + condition_fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            element_wise_binary_accuracy = (
                condition_element_correct
                / max(condition_element_total, 1)
            )
            predicted_precision = (
                predicted_condition_tp
                / (predicted_condition_tp + predicted_condition_fp + 1e-8)
            )
            predicted_recall = (
                predicted_condition_tp
                / (predicted_condition_tp + predicted_condition_fn + 1e-8)
            )
            predicted_f1 = (
                2 * predicted_precision * predicted_recall
                / (predicted_precision + predicted_recall + 1e-8)
            )
            predicted_element_wise_binary_accuracy = (
                predicted_condition_element_correct
                / max(predicted_condition_element_total, 1)
            )
            #
            print(
                '[VALID] P: {:.4f}, R: {:.4f}, F1: {:.4f}, '
                'Element Acc: {:.4f}'
                .format(precision, recall, f1,
                        element_wise_binary_accuracy))
            print(
                '[VALID][Predicted Exercise] P: {:.4f}, R: {:.4f}, '
                'F1: {:.4f}, Element Acc: {:.4f}'
                .format(
                    predicted_precision,
                    predicted_recall,
                    predicted_f1,
                    predicted_element_wise_binary_accuracy))
            print(
                '[VALID] Selected_Condition_Coverage: {:.4f} ({:,}/{:,})'
                .format(
                    selected_condition_entries
                    / max(ground_truth_condition_entries, 1),
                    selected_condition_entries,
                    ground_truth_condition_entries))

            if self.args.condition_pr_analysis:
                if pr_probabilities:
                    from others.condition_pr import save_condition_pr_analysis

                    pr_output_dir = os.path.join(
                        self.args.condition_pr_output,
                        'validation_step_{:06d}'.format(step),
                    )
                    pr_summary = save_condition_pr_analysis(
                        torch.cat(pr_probabilities, dim=0).numpy(),
                        torch.cat(pr_targets, dim=0).numpy(),
                        torch.cat(pr_masks, dim=0).numpy(),
                        torch.cat(pr_exercise_targets, dim=0).numpy(),
                        output_dir=pr_output_dir,
                        epoch=step,
                        class_num=self.config.CLASS_NUM,
                        exercise_names_by_raw_id=self.ex_int2str,
                        condition_names_by_raw_id=self.cond_int2str,
                    )
                    print('[VALID] Condition PR analysis: {}'.format(pr_summary))
                else:
                    print('[VALID] Condition PR analysis skipped: no valid condition targets')

            return stats

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i : i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        if not cal_lead and not cal_oracle:
            self.model.eval()
        stats = Statistics()

        src_path = "%s_step%d.src" % (self.args.result_path, step)
        can_path = "%s_step%d.candidate" % (self.args.result_path, step)
        gold_path = "%s_step%d.gold" % (self.args.result_path, step)
        with open(can_path, "w") as save_pred:
            with open(gold_path, "w") as save_gold:
                with open(src_path, "w") as save_src:
                    with torch.no_grad():
                        for batch in test_iter:
                            src = batch.src
                            labels = batch.src_sent_labels
                            segs = batch.segs
                            clss = batch.clss
                            mask = batch.mask_src
                            mask_cls = batch.mask_cls

                            gold = []
                            pred = []

                            if cal_lead:
                                selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                            elif cal_oracle:
                                selected_ids = [
                                    [j for j in range(batch.clss.size(1)) if labels[i][j] == 1]
                                    for i in range(batch.batch_size)
                                ]
                            else:
                                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                                loss = self.loss(sent_scores, labels.float())
                                loss = (loss * mask.float()).sum()
                                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                                stats.update(batch_stats)

                                sent_scores = sent_scores + mask.float()
                                sent_scores = sent_scores.cpu().data.numpy()
                                selected_ids = np.argsort(-sent_scores, 1)
                            # selected_ids = np.sort(selected_ids,1)
                            for i, idx in enumerate(selected_ids):
                                _pred = []
                                if len(batch.src_str[i]) == 0:
                                    continue
                                for j in selected_ids[i][: len(batch.src_str[i])]:
                                    if j >= len(batch.src_str[i]):
                                        continue
                                    candidate = " ".join(batch.src_str[i][j]).strip()
                                    if self.args.block_trigram:
                                        if not _block_tri(candidate, _pred):
                                            _pred.append(candidate)
                                    else:
                                        _pred.append(candidate)

                                    if (
                                        (not cal_oracle)
                                        and (not self.args.recall_eval)
                                        and len(_pred) == 3
                                    ):
                                        break

                                _pred = " ".join(_pred).replace(" ", "")
                                if self.args.recall_eval:
                                    _pred = " ".join(_pred.split()[: len(batch.tgt_str[i].split())])

                                pred.append(_pred)
                                gold.append(batch.tgt_str[i])
                            for i in range(len(src)):
                                save_src.write("".join("".join(j) for j in batch.src_str[i]).strip().replace("▁", " ") + "\n")
                            for i in range(len(gold)):
                                save_gold.write("".join(gold[i]).strip().replace("▁", " ") + "\n")
                            for i in range(len(pred)):
                                save_pred.write("".join(pred[i]).strip().replace("▁", " ") + "\n")
        if step != -1 and self.args.report_rouge:
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info("Rouges at step %d \n%s" % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)

        return stats

    def _gradient_accumulation(self, src_emb, segs, mask_src, tgt):

        loss_sum_weighted = 0.0
        correct_sum = 0
        sample_sum = 0

        for i, batch in enumerate(batch):
            if i == 0:
                self.model.zero_grad()


            # OUTPUT
            sent_scores = self.model(src_emb, segs, mask_src)

            loss = self.loss(sent_scores,tgt)
            # print(loss)

            # LOSS
            bs = batch.batch_size
            loss_sum_weighted += loss.item() / bs
            sample_sum += bs

            # ACC
            pred_idx = torch.argmax(sent_scores, dim=-1)
            correct_sum += (pred_idx == tgt_idx).sum().item()

            (loss / self.grad_accum_count).backward()

            # loss = (loss * mask.float()).sum()
            # (loss / loss.numel()).backward()
            # loss.div(float(normalization)).backward()
            # batch_stats = Statistics(
            #     float(loss.cpu().data.numpy()),
            #     normalization,
            #     n_correct=correct,
            #     n_total=total
            # )
            # batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)
            # total_stats.update(batch_stats)
            # epoch_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [
                        p.grad.data
                        for p in self.model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    distributed.all_reduce_and_rescale_tensors(grads, float(1))
                self.optim.step()
        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [
                    p.grad.data
                    for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                distributed.all_reduce_and_rescale_tensors(grads, float(1))
            self.optim.step()

        return loss_sum_weighted, correct_sum, sample_sum

    def _save(self, dir_path):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            "model": model_state_dict,
            # 'generator': generator_state_dict,
            "opt": self.args,
            "optim": self.optim,
        }
        dir_path = self.args.model_path + dir_path
        os.makedirs(dir_path, exist_ok=True)
        checkpoint_path = os.path.join(dir_path, "model_ENCLAYER_{}.pt".format(self.args.ext_layers))
        #
        logger.info("Saving checkpoint %s" % checkpoint_path)
        #
        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats, multigpu=self.n_gpu > 1
            )

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats, valid_stats=valid_stats
            )

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
