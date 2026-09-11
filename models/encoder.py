import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from Embedder.Embedder_config import config
from models.MoE import (
    ExerciseMoE, JointGroupMoE, JointMoE, load_balance_loss)

class Classifier(nn.Module):
    def __init__(self, hidden_size, args):
        super(Classifier, self).__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.moe_enabled = getattr(args, 'moe', False)
        self.expert_type = getattr(args, 'expert_type', 'joint')
        self.expert_mode = getattr(args, 'expert_mode', 'baseline')

        # Exercise head
        self.exercise_linear = nn.Linear(hidden_size, 256)
        self.exercise_classifier = nn.Linear(256, config.CLASS_NUM)

        self.act = nn.ReLU()

        if not self.moe_enabled:
            # Each condition attends to different joints and frames.
            self.condition_attn = nn.Linear(
                hidden_size, config.NUM_CONDITIONS)
            self.condition_norm = nn.LayerNorm(hidden_size)
            self.condition_dropout = nn.Dropout(args.ext_dropout)
            self.condition_weight = nn.Parameter(torch.empty(
                config.NUM_CONDITIONS, hidden_size))
            self.condition_bias = nn.Parameter(torch.zeros(
                config.NUM_CONDITIONS))
            nn.init.xavier_uniform_(self.condition_weight)
            return

        if self.expert_mode not in ('baseline', 'tcn'):
            raise ValueError(
                "expert_mode must be one of: baseline, tcn")
        if self.expert_type == 'exercise' and self.expert_mode == 'tcn':
            raise NotImplementedError(
                "expert_mode='tcn' is not implemented for exercise experts")

        self.router_linear = nn.Linear(hidden_size, 256)

        expert_hidden_size = getattr(args, 'expert_dim', 256)
        joint_moe_classes = {
            'joint': JointMoE,
            'group': JointGroupMoE,
        }
        if self.expert_type in joint_moe_classes:
            is_grouped = self.expert_type == 'group'
            num_experts = (
                JointGroupMoE.NUM_GROUPS
                if is_grouped
                else JointMoE.NUM_JOINTS
            )
            self.router_head = nn.Linear(256, num_experts)
            joint_moe_class = joint_moe_classes[self.expert_type]
            self.condition_moe = joint_moe_class(
                hidden_size=hidden_size,
                expert_hidden_size=expert_hidden_size,
                top_k=getattr(args, 'top_k', 3),
                dropout=args.ext_dropout,
                num_conditions=config.NUM_CONDITIONS,
                has_end_cls=args.attach_cls_token_to_end_of_seqlen,
                expert_mode=self.expert_mode,
                tcn_conv_type=getattr(args, 'tcn_conv_type', 'conv1d'),
                tcn_channels=getattr(args, 'tcn_channels', 128),
            )
        elif self.expert_type == 'exercise':
            exercise_condition_map = getattr(
                args, 'exercise_condition_map', None)
            if exercise_condition_map is None:
                raise ValueError(
                    "exercise_condition_map is required for exercise MoE")
            if len(exercise_condition_map) != config.CLASS_NUM:
                raise ValueError(
                    "exercise_condition_map must have {} entries".format(
                        config.CLASS_NUM))
            self.router_head = nn.Linear(256, config.CLASS_NUM)
            self.condition_moe = ExerciseMoE(
                hidden_size=hidden_size,
                expert_hidden_size=expert_hidden_size,
                top_k=getattr(args, 'top_k', 3),
                dropout=args.ext_dropout,
                num_conditions=config.NUM_CONDITIONS,
                exercise_condition_map=exercise_condition_map,
            )
        else:
            raise ValueError(
                "expert_type must be one of: joint, group, exercise")

    def _baseline_conditions(self, x, valid_mask):
        attn_score = self.condition_attn(x).transpose(1, 2)  # [B,97,S]
        if valid_mask is not None:
            attn_score = attn_score.masked_fill(
                ~valid_mask[:, None, :].bool(),
                torch.finfo(attn_score.dtype).min)
        attn_weight = torch.softmax(attn_score, dim=-1)
        condition_context = torch.matmul(attn_weight, x)  # [B,97,H]
        condition_context = self.condition_dropout(
            self.condition_norm(condition_context))
        return (
            torch.sum(
                condition_context * self.condition_weight.unsqueeze(0),
                dim=-1)
            + self.condition_bias
        )

    def forward(self, x, valid_mask=None, exercise_targets=None,
                condition_labels=None, condition_mask=None, threshold=0.5):
        # Before:
        # Use only the last CLS token as the condition representation.

        # After:
        # Apply attention pooling over all tokens to generate a condition representation.

        # x: [B, SEQ_LEN, H]

        # ===== Exercise =====
        first_cls = x[:, 0, :]

        exercise_out = self.exercise_linear(first_cls)
        exercise_feature = self.act(exercise_out)
        pred_exercise = self.exercise_classifier(exercise_feature)

        if not self.moe_enabled:
            pred_conditions = self._baseline_conditions(x, valid_mask)
            return pred_exercise, pred_conditions, {
                'aux_loss': x.new_zeros(()),
                'condition_mask': condition_mask,
            }

        if valid_mask is None:
            valid_mask = torch.ones(
                x.shape[:2], dtype=torch.bool, device=x.device)
        router_feature = self.act(self.router_linear(first_cls))
        router_logits = self.router_head(router_feature)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k = self.condition_moe.top_k
        _, topk_indices = torch.topk(router_probs, top_k, dim=-1)
        aux_loss = load_balance_loss(router_probs, topk_indices)

        if self.expert_type in ('joint', 'group'):
            pred_conditions = self.condition_moe(
                x, valid_mask, router_probs, topk_indices)
            selected_condition_mask = condition_mask
            selected_experts = topk_indices
            oracle_details = None
        else:
            (pred_conditions, expert_output_mask, selected_experts,
             oracle_details) = self.condition_moe(
                x=x,
                valid_mask=valid_mask,
                router_feature=router_feature,
                topk_indices=topk_indices,
                exercise_targets=exercise_targets,
                condition_labels=condition_labels,
                ground_truth_mask=condition_mask,
                threshold=threshold,
            )
            selected_condition_mask = expert_output_mask

        return pred_exercise, pred_conditions, {
            'aux_loss': aux_loss,
            'condition_mask': selected_condition_mask,
            'router_logits': router_logits,
            'router_needs_supervision': (
                self.expert_type == 'exercise'),
            'router_probs': router_probs,
            'topk_indices': topk_indices,
            'selected_experts': selected_experts,
            'oracle_details': oracle_details,
        }

# class Classifier(nn.Module):
#     def __init__(self, hidden_size, args):
#         super(Classifier, self).__init__()
#         # ver 1
#         self.args = args
#
#         self.exercise_linear = nn.Linear(hidden_size, 256)
#         self.exercise_classifier =nn.Linear(256 ,41)
#
#         self.conditions_linear = nn.Linear(hidden_size, 256)
#         self.conditions_classifier = nn.Linear(256, 97)
#         #
#
#         self.act = nn.ReLU()
#
#     def forward(self, x):
#         # [BS,SEQ_LEN,DIM]
#         first_cls = x[:, 0, :]
#
#         if self.args.attach_cls_token_to_end_of_seqlen:
#             last_cls = x[:, -1, :]
#             exercise_cls = first_cls
#             condition_cls = last_cls
#         else:
#             exercise_cls = first_cls
#             condition_cls = first_cls
#
#         exercise_out = self.exercise_linear(exercise_cls)
#         pred_exercise = self.exercise_classifier(self.act(exercise_out))
#         #
#         condition_out = self.conditions_linear(condition_cls)
#         pred_conditions = self.conditions_classifier(self.act(condition_out))
#         #
#         return pred_exercise, pred_conditions


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = (emb * math.sqrt(self.dim)).unsqueeze(0)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)].to(emb.device)
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
