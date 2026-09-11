import torch
import torch.nn as nn


def build_exercise_condition_map(samples, num_exercises, num_conditions,
                                 exercise_offset=22):
    """Build local exercise -> local condition-index lists from TRAIN samples."""
    condition_offset = exercise_offset + num_exercises
    mapping = [set() for _ in range(num_exercises)]

    for _, raw_exercise, conditions in samples:
        exercise_index = int(raw_exercise) - exercise_offset
        if not 0 <= exercise_index < num_exercises:
            raise ValueError(
                "exercise index out of range: raw={}, local={}".format(
                    raw_exercise, exercise_index))
        for raw_condition, _ in conditions:
            condition_index = int(raw_condition) - condition_offset
            if not 0 <= condition_index < num_conditions:
                raise ValueError(
                    "condition index out of range: raw={}, local={}".format(
                        raw_condition, condition_index))
            mapping[exercise_index].add(condition_index)

    result = [sorted(indices) for indices in mapping]
    missing = [index for index, indices in enumerate(result) if not indices]
    if missing:
        raise ValueError(
            "TRAIN data has exercises without conditions: {}".format(missing))
    return result


def load_balance_loss(router_probs, topk_indices):
    """Switch-style differentiable router load-balancing auxiliary loss."""
    num_experts = router_probs.size(-1)
    importance = router_probs.mean(dim=0)
    selected = torch.zeros_like(router_probs)
    selected.scatter_(1, topk_indices, 1.0)
    load = selected.mean(dim=0) / topk_indices.size(1)
    return num_experts * torch.sum(importance * load)


class JointExpert(nn.Module):
    def __init__(self, hidden_size, expert_hidden_size, dropout):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, expert_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_size, expert_hidden_size),
        )

    def forward(self, joint_tokens, frame_mask):
        frame_mask = frame_mask.bool()
        scores = self.attention(joint_tokens).squeeze(-1)
        scores = scores.masked_fill(
            ~frame_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * frame_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context = torch.sum(weights.unsqueeze(-1) * joint_tokens, dim=1)
        return self.output(context)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout, conv_type):
        super().__init__()
        if conv_type == 'conv1d':
            self.temporal_conv = nn.Conv1d(
                channels, channels, kernel_size=3,
                padding=dilation, dilation=dilation)
        elif conv_type == 'depth_point_wise':
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(
                    channels, channels, kernel_size=3,
                    padding=dilation, dilation=dilation,
                    groups=channels),
                nn.Conv1d(channels, channels, kernel_size=1),
            )
        else:
            raise ValueError(
                "tcn_conv_type must be one of: conv1d, depth_point_wise")
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, frame_mask):
        output = self.temporal_conv(x)
        output = self.norm(output.transpose(1, 2)).transpose(1, 2)
        output = self.dropout(self.activation(output))
        output = output + x
        return output * frame_mask.unsqueeze(1).to(output.dtype)


class TemporalConvEncoder(nn.Module):
    DILATIONS = (1, 2, 4)

    def __init__(self, hidden_size, tcn_channels, expert_hidden_size,
                 dropout, conv_type):
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, tcn_channels)
        self.input_norm = nn.LayerNorm(tcn_channels)
        self.activation = nn.GELU()
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(
                tcn_channels, dilation, dropout, conv_type)
            for dilation in self.DILATIONS
        ])
        self.output = nn.Linear(tcn_channels * 2, expert_hidden_size)

    def forward(self, frame_tokens, frame_mask):
        frame_mask = frame_mask.bool()
        mask_float = frame_mask.unsqueeze(-1).to(frame_tokens.dtype)
        temporal = self.activation(
            self.input_norm(self.input_projection(frame_tokens)))
        temporal = temporal * mask_float
        temporal = temporal.transpose(1, 2)
        for block in self.blocks:
            temporal = block(temporal, frame_mask)

        temporal_frames = temporal.transpose(1, 2)
        valid_count = mask_float.sum(dim=1).clamp_min(1.0)
        average_context = (temporal_frames * mask_float).sum(
            dim=1) / valid_count

        max_input = temporal_frames.masked_fill(
            ~frame_mask.unsqueeze(-1),
            torch.finfo(temporal_frames.dtype).min,
        )
        max_context = max_input.max(dim=1).values
        has_valid_frame = frame_mask.any(dim=1, keepdim=True)
        max_context = torch.where(
            has_valid_frame, max_context, torch.zeros_like(max_context))
        return self.output(torch.cat(
            [average_context, max_context], dim=-1))


class JointTCNExpert(nn.Module):
    def __init__(self, hidden_size, expert_hidden_size, dropout,
                 tcn_channels, conv_type):
        super().__init__()
        self.temporal_encoder = TemporalConvEncoder(
            hidden_size=hidden_size,
            tcn_channels=tcn_channels,
            expert_hidden_size=expert_hidden_size,
            dropout=dropout,
            conv_type=conv_type,
        )

    def forward(self, joint_tokens, frame_mask):
        return self.temporal_encoder(joint_tokens, frame_mask)


JOINT_GROUP_NAMES = (
    'Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle')

# Indices follow Embedder_config.JOINTS_NAME after [SEP] is removed.
JOINT_GROUP_INDICES = (
    (0, 13),                 # Head, Neck
    (13, 1, 2, 3, 4),       # Neck, Shoulders, Elbows
    (1, 2, 3, 4, 5, 6),     # Shoulders, Elbows, Wrists
    (3, 4, 5, 6, 14, 15),   # Elbows, Wrists, Palms
    (16, 17, 7, 8, 9, 10),  # Back, Waist, Hips, Knees
    (7, 8, 9, 10, 11, 12),  # Hips, Knees, Ankles
    (9, 10, 11, 12, 18, 19),  # Knees, Ankles, Feet
)


def spatial_attention_pool(group_tokens, group_mask, spatial_query):
    group_mask = group_mask.bool()
    spatial_scores = torch.sum(group_tokens * spatial_query, dim=-1)
    spatial_scores = spatial_scores.masked_fill(
        ~group_mask, torch.finfo(spatial_scores.dtype).min)
    spatial_weights = torch.softmax(spatial_scores, dim=-1)
    spatial_weights = spatial_weights * group_mask.to(spatial_weights.dtype)
    spatial_weights = spatial_weights / spatial_weights.sum(
        dim=-1, keepdim=True).clamp_min(1e-8)
    frame_context = torch.sum(
        spatial_weights.unsqueeze(-1) * group_tokens, dim=2)
    frame_mask = group_mask.any(dim=-1)
    return frame_context, frame_mask


class JointGroupExpert(nn.Module):
    def __init__(self, hidden_size, expert_hidden_size, dropout):
        super().__init__()
        self.spatial_query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(
            self.spatial_query, mean=0.0, std=hidden_size ** -0.5)
        self.temporal_attention = nn.Linear(hidden_size, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, expert_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_size, expert_hidden_size),
        )

    def forward(self, group_tokens, group_mask):
        frame_context, frame_mask = spatial_attention_pool(
            group_tokens, group_mask, self.spatial_query)
        temporal_scores = self.temporal_attention(
            frame_context).squeeze(-1)
        temporal_scores = temporal_scores.masked_fill(
            ~frame_mask, torch.finfo(temporal_scores.dtype).min)
        temporal_weights = torch.softmax(temporal_scores, dim=-1)
        temporal_weights = temporal_weights * frame_mask.to(
            temporal_weights.dtype)
        temporal_weights = temporal_weights / temporal_weights.sum(
            dim=-1, keepdim=True).clamp_min(1e-8)
        context = torch.sum(
            temporal_weights.unsqueeze(-1) * frame_context, dim=1)
        return self.output(context)


class JointGroupTCNExpert(nn.Module):
    def __init__(self, hidden_size, expert_hidden_size, dropout,
                 tcn_channels, conv_type):
        super().__init__()
        self.spatial_query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(
            self.spatial_query, mean=0.0, std=hidden_size ** -0.5)
        self.temporal_encoder = TemporalConvEncoder(
            hidden_size=hidden_size,
            tcn_channels=tcn_channels,
            expert_hidden_size=expert_hidden_size,
            dropout=dropout,
            conv_type=conv_type,
        )

    def forward(self, group_tokens, group_mask):
        frame_context, frame_mask = spatial_attention_pool(
            group_tokens, group_mask, self.spatial_query)
        return self.temporal_encoder(frame_context, frame_mask)


class JointMoE(nn.Module):
    NUM_JOINTS = 20
    TOKENS_PER_FRAME = 21  # [SEP] + 20 joints

    def __init__(self, hidden_size, expert_hidden_size, top_k, dropout,
                 num_conditions, has_end_cls=False,
                 expert_joint_indices=None, expert_mode='baseline',
                 tcn_conv_type='conv1d', tcn_channels=128):
        super().__init__()
        self.is_grouped = expert_joint_indices is not None
        if expert_joint_indices is None:
            expert_joint_indices = tuple(
                (index,) for index in range(self.NUM_JOINTS))
        self.expert_joint_indices = tuple(
            tuple(indices) for indices in expert_joint_indices)
        num_experts = len(self.expert_joint_indices)
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                "joint top_k must be in [1, {}]".format(num_experts))
        self.top_k = top_k
        self.has_end_cls = has_end_cls
        self.expert_hidden_size = expert_hidden_size
        if expert_mode == 'baseline':
            expert_class = (
                JointGroupExpert if self.is_grouped else JointExpert)
            self.experts = nn.ModuleList([
                expert_class(hidden_size, expert_hidden_size, dropout)
                for _ in range(num_experts)
            ])
        elif expert_mode == 'tcn':
            expert_class = (
                JointGroupTCNExpert if self.is_grouped else JointTCNExpert)
            self.experts = nn.ModuleList([
                expert_class(
                    hidden_size, expert_hidden_size, dropout,
                    tcn_channels, tcn_conv_type)
                for _ in range(num_experts)
            ])
        else:
            raise ValueError("expert_mode must be one of: baseline, tcn")
        self.condition_head = nn.Linear(expert_hidden_size, num_conditions)

    def _joint_tokens(self, x, valid_mask):
        sequence_end = x.size(1) - int(self.has_end_cls)
        frame_token_count = sequence_end - 1
        if frame_token_count <= 0 or frame_token_count % self.TOKENS_PER_FRAME:
            raise ValueError(
                "invalid packed sequence length {} for 21 tokens/frame".format(
                    x.size(1)))
        num_frames = frame_token_count // self.TOKENS_PER_FRAME
        frame_blocks = x[:, 1:sequence_end, :].reshape(
            x.size(0), num_frames, self.TOKENS_PER_FRAME, x.size(2))
        mask_blocks = valid_mask[:, 1:sequence_end].reshape(
            x.size(0), num_frames, self.TOKENS_PER_FRAME)
        # index 0 is [SEP]; preprocessing orders Head first at joint index 0.
        return frame_blocks[:, :, 1:, :], mask_blocks[:, :, 1:].bool()

    def _selected_expert_outputs(self, x, valid_mask, topk_indices):
        joint_tokens, joint_mask = self._joint_tokens(x, valid_mask)
        expert_outputs = x.new_zeros(
            x.size(0), topk_indices.size(1), self.expert_hidden_size)

        for expert_index, expert in enumerate(self.experts):
            selected = (topk_indices == expert_index)
            sample_indices, slots = selected.nonzero(as_tuple=True)
            if sample_indices.numel() == 0:
                continue
            selected_joint_tokens = joint_tokens.index_select(
                0, sample_indices)
            selected_joint_mask = joint_mask.index_select(
                0, sample_indices)
            if self.is_grouped:
                group_indices = torch.tensor(
                    self.expert_joint_indices[expert_index],
                    dtype=torch.long,
                    device=x.device,
                )
                expert_output = expert(
                    selected_joint_tokens.index_select(2, group_indices),
                    selected_joint_mask.index_select(2, group_indices),
                )
            else:
                joint_index = self.expert_joint_indices[expert_index][0]
                expert_output = expert(
                    selected_joint_tokens[:, :, joint_index, :],
                    selected_joint_mask[:, :, joint_index],
                )
            expert_outputs[sample_indices, slots] = expert_output
        return expert_outputs

    def forward(self, x, valid_mask, router_probs, topk_indices):
        expert_outputs = self._selected_expert_outputs(
            x, valid_mask, topk_indices)
        topk_weights = router_probs.gather(1, topk_indices)
        topk_weights = topk_weights / topk_weights.sum(
            dim=1, keepdim=True).clamp_min(1e-8)
        representation = torch.sum(
            expert_outputs * topk_weights.unsqueeze(-1), dim=1)

        return self.condition_head(representation)


class JointGroupMoE(JointMoE):
    NUM_GROUPS = len(JOINT_GROUP_NAMES)
    GROUP_NAMES = JOINT_GROUP_NAMES
    GROUP_JOINT_INDICES = JOINT_GROUP_INDICES

    def __init__(self, hidden_size, expert_hidden_size, top_k, dropout,
                 num_conditions, has_end_cls=False, expert_mode='baseline',
                 tcn_conv_type='conv1d', tcn_channels=128):
        super().__init__(
            hidden_size=hidden_size,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            dropout=dropout,
            num_conditions=num_conditions,
            has_end_cls=has_end_cls,
            expert_joint_indices=self.GROUP_JOINT_INDICES,
            expert_mode=expert_mode,
            tcn_conv_type=tcn_conv_type,
            tcn_channels=tcn_channels,
        )


class ExerciseExpert(nn.Module):
    def __init__(self, hidden_size, router_size, expert_hidden_size,
                 output_size, dropout):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.context_projection = nn.Linear(hidden_size, expert_hidden_size)
        self.router_projection = nn.Linear(router_size, expert_hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(expert_hidden_size, output_size)

    def forward(self, x, valid_mask, router_feature):
        valid_mask = valid_mask.bool()
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(
            ~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        hidden = (
            self.context_projection(context)
            + self.router_projection(router_feature)
        )
        return self.output(self.dropout(self.activation(hidden)))


class ExerciseMoE(nn.Module):
    """Exercise experts with variable-width, direct condition outputs."""

    def __init__(self, hidden_size, expert_hidden_size, top_k, dropout,
                 num_conditions, exercise_condition_map):
        super().__init__()
        self.num_experts = len(exercise_condition_map)
        if not 1 <= top_k <= self.num_experts:
            raise ValueError(
                "exercise top_k must be in [1, {}]".format(self.num_experts))
        self.top_k = top_k
        self.num_conditions = num_conditions
        self.experts = nn.ModuleList()

        for expert_index, indices in enumerate(exercise_condition_map):
            if not indices:
                raise ValueError(
                    "exercise expert {} has no condition outputs".format(
                        expert_index))
            index_tensor = torch.tensor(indices, dtype=torch.long)
            self.register_buffer(
                "condition_indices_{}".format(expert_index), index_tensor)
            self.experts.append(ExerciseExpert(
                hidden_size, 256, expert_hidden_size, len(indices), dropout))

    def _indices(self, expert_index):
        return getattr(self, "condition_indices_{}".format(expert_index))

    def _run_assignments(self, x, valid_mask, router_feature, assignments):
        batch_size, num_slots = assignments.shape
        logits = x.new_zeros(
            batch_size, num_slots, self.num_conditions)
        output_mask = torch.zeros(
            batch_size, num_slots, self.num_conditions,
            dtype=torch.bool, device=x.device)

        for expert_index, expert in enumerate(self.experts):
            selected = assignments == expert_index
            sample_indices, slots = selected.nonzero(as_tuple=True)
            if sample_indices.numel() == 0:
                continue
            local_logits = expert(
                x[sample_indices],
                valid_mask[sample_indices],
                router_feature[sample_indices],
            )
            condition_indices = self._indices(expert_index)
            logits[sample_indices[:, None], slots[:, None],
                   condition_indices[None, :]] = local_logits
            output_mask[sample_indices[:, None], slots[:, None],
                        condition_indices[None, :]] = True
        return logits, output_mask

    def forward(self, x, valid_mask, router_feature, topk_indices,
                exercise_targets=None, condition_labels=None,
                ground_truth_mask=None, threshold=0.5):
        if self.training:
            if exercise_targets is None:
                raise ValueError(
                    "exercise_targets are required to train exercise experts")
            assignments = exercise_targets.view(-1, 1)
            candidate_logits, candidate_masks = self._run_assignments(
                x, valid_mask, router_feature, assignments)
            selected_experts = exercise_targets
            return (
                candidate_logits[:, 0, :],
                candidate_masks[:, 0, :],
                selected_experts,
                None,
            )

        candidate_logits, candidate_masks = self._run_assignments(
            x, valid_mask, router_feature, topk_indices)
        if self.top_k == 1:
            return (
                candidate_logits[:, 0, :],
                candidate_masks[:, 0, :],
                topk_indices[:, 0],
                None,
            )

        if condition_labels is None or ground_truth_mask is None:
            raise ValueError(
                "condition_labels and ground_truth_mask are required for "
                "GT-assisted Top-K exercise-expert selection")

        # Compare every candidate on exactly the same GT condition entries.
        # Conditions not produced by a candidate expert count as negative
        # predictions instead of disappearing from that expert's denominator.
        evaluation_mask = ground_truth_mask[:, None, :].expand_as(
            candidate_masks)
        predictions = (
            (torch.sigmoid(candidate_logits) > threshold)
            & candidate_masks
        )
        targets = condition_labels[:, None, :] > 0.5
        correct = ((predictions == targets) & evaluation_mask).sum(dim=-1)
        counts = evaluation_mask.sum(dim=-1)
        accuracies = correct.float() / counts.clamp_min(1).float()
        accuracies = accuracies.masked_fill(counts == 0, -1.0)
        best_slots = accuracies.argmax(dim=1)
        batch_indices = torch.arange(x.size(0), device=x.device)
        return (
            candidate_logits[batch_indices, best_slots],
            candidate_masks[batch_indices, best_slots],
            topk_indices[batch_indices, best_slots],
            {
                "candidate_accuracies": accuracies,
                "candidate_entry_counts": counts,
                "best_slots": best_slots,
            },
        )
