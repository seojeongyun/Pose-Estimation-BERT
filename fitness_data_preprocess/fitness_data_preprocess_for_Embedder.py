"""
AI Hub fitness 데이터셋에서 BERTFORHPE Embedder 입력 데이터를 생성한다.

기존 dataset 구조는 유지한다.

    dataset = [
        [
            video_dict,                           # 16 frames
            exercise_raw_id,
            [[condition_raw_id, boolean_flag], ...],
        ],
        ...
    ]

모든 feature mode는 image origin 기준 관절 좌표를 torso length로 나눈
P=[norm_x, norm_y]를 기본 입력으로 사용한다.

지원 mode:
    BASELINE          : [P, joint_id/21]                                      -> 3D
    VECTOR            : [P, center-to-joint vector, joint_id/21]             -> 5D
    GRAPH_1HOP        : [P, 1-hop neighbor offset, joint_id/21]              -> 5D
    GRAPH_2HOP        : [P, 1-hop offset, 2-hop offset, joint_id/21]         -> 7D
    VECTOR_GRAPH_1HOP : [P, vector, 1-hop offset, joint_id/21]              -> 7D
    VECTOR_GRAPH_2HOP : [P, vector, 1-hop offset, 2-hop offset, joint_id/21] -> 9D
    SKELETON_EDGE_1HOP: [P, parent-to-joint edge, joint_id/21]               -> 5D
    VECTOR_SKELETON_EDGE_1HOP:
        [P, vector, parent-to-joint edge, joint_id/21]                        -> 7D

데이터 검색, 운동/condition 처리, view 유효성 검사, 16-frame padding,
workout/condition label 및 pickle outer 구조는 기존 로직을 유지한다.
"""

import argparse
import json
import os
import pickle

from glob import glob

import numpy as np
from tqdm import tqdm

DATA_TYPE = 'TRAIN' # 'TRAIN' 'VALID'
FEATURE_MODE = 'VECTOR_SKELETON_EDGE_1HOP'

IS_CONTAIN_HARD_EXERCISE = True # True: Hard Exercise 포함한 데이터를 만들겠다.
IS_INTEGRATE_ROW = True  # True: 바벨로우-덤벨로우 통합
IS_JOINT_ID_NORM = True
IS_INTEGRATE_PUSHUP = True

MAX_FRAME = 16
W, H = 1920, 1080
TORSO_EPS = 1e-6
JOINT_ID_DIVISOR = 21.0 if IS_JOINT_ID_NORM else 1.0

# Feature configuration
FEATURE_MODES = (
    'BASELINE',
    'VECTOR',
    'GRAPH_1HOP',
    'GRAPH_2HOP',
    'VECTOR_GRAPH_1HOP',
    'VECTOR_GRAPH_2HOP',
    'SKELETON_EDGE_1HOP',
    'VECTOR_SKELETON_EDGE_1HOP',
)

row_mode_dir = (
    'row_integrated' if IS_INTEGRATE_ROW else 'row_not_integrated'
)
hard_exercise_mode_dir = (
    'hard_exercise_included'
    if IS_CONTAIN_HARD_EXERCISE
    else 'hard_exercise_excluded'
)
joint_id_mode_dir = (
    'joint_id_normalized' if IS_JOINT_ID_NORM else 'joint_id_raw'
)

pushup_mode_dir = (
      'pushup_integrated'
      if IS_INTEGRATE_PUSHUP
      else 'pushup_not_integrated'
  )

VOCAB_TREE_ROOT = os.path.join(
      '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/'
      'fitness_data_preprocess/vocab',
      row_mode_dir,
      hard_exercise_mode_dir,
      pushup_mode_dir,
  )


BASE_PATH = {
    'TRAIN': [
        '/storage/jysuh/fitness/fitness/train/label/furniture_Labeling',
        '/storage/jysuh/fitness/fitness/train/label/body_Labeling_new_220128',
        '/storage/jysuh/fitness/fitness/train/label/barbell_dumbbell_Labeling_new_220128',
    ],
    'VALID': [
        '/storage/jysuh/fitness/fitness/validation/label/barbell_dumbbell_Labeling',
        '/storage/jysuh/fitness/fitness/validation/label/body_Labeling',
        '/storage/jysuh/fitness/fitness/validation/label/furniture_Labeling',
    ],
}

VOCAB_PATH = {
    'TRAIN': {
        'JOINT': os.path.join(VOCAB_TREE_ROOT, 'TRAIN', 'joint_vocab.pkl'),
        'WORKOUT': os.path.join(VOCAB_TREE_ROOT, 'TRAIN', 'workout_vocab.pkl'),
        'CONDITION': os.path.join(VOCAB_TREE_ROOT, 'TRAIN', 'condition_vocab.pkl'),
    },
    'VALID': {
        'JOINT': os.path.join(VOCAB_TREE_ROOT, 'VALID', 'joint_vocab.pkl'),
        'WORKOUT': os.path.join(VOCAB_TREE_ROOT, 'VALID', 'workout_vocab.pkl'),
        'CONDITION': os.path.join(VOCAB_TREE_ROOT, 'VALID', 'condition_vocab.pkl'),
    },
}

JOINTS_ORDER = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee',
    'Right Knee', 'Left Ankle', 'Right Ankle', 'Neck', 'Left Palm',
    'Right Palm', 'Back', 'Waist', 'Left Foot', 'Right Foot',
]

# 기존 padding frame의 key 순서를 그대로 유지한다.
JOINTS_ORDER_HEAD_ver = [
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee',
    'Right Knee', 'Left Ankle', 'Right Ankle', 'Neck', 'Left Palm',
    'Right Palm', 'Back', 'Waist', 'Left Foot', 'Right Foot', 'Head',
]

# 기존 실제 frame은 Head key를 가장 먼저 저장한다.
OUTPUT_JOINTS_ORDER = ['Head'] + [
    joint_name for joint_name in JOINTS_ORDER
    if joint_name not in ('Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear')
]

HEAD_KEYS = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear']
VIEWS = ['view1', 'view2', 'view3', 'view4', 'view5']

HARD_EXERCISE = [
    '바이시클 크런치', '크런치', '시저크로스', '라잉 레그 레이즈',
    '플랭크', '힙쓰러스트', '푸시업', '니푸쉬업',
]


SKELETON_EDGES = (
    ('Head', 'Neck'),
    ('Neck', 'Left Shoulder'),
    ('Neck', 'Right Shoulder'),
    ('Neck', 'Back'),
    ('Left Shoulder', 'Left Elbow'),
    ('Left Elbow', 'Left Wrist'),
    ('Left Wrist', 'Left Palm'),
    ('Right Shoulder', 'Right Elbow'),
    ('Right Elbow', 'Right Wrist'),
    ('Right Wrist', 'Right Palm'),
    ('Back', 'Waist'),
    ('Waist', 'Left Hip'),
    ('Waist', 'Right Hip'),
    ('Left Hip', 'Left Knee'),
    ('Left Knee', 'Left Ankle'),
    ('Left Ankle', 'Left Foot'),
    ('Right Hip', 'Right Knee'),
    ('Right Knee', 'Right Ankle'),
    ('Right Ankle', 'Right Foot'),
)


# Back을 root로 사용하는 방향성 skeleton tree.
# 각 key 관절의 feature에는 parent -> key 방향의 torso-normalized edge를 저장한다.
SKELETON_ROOT = 'Back'
SKELETON_PARENT = {
    'Head': 'Neck',
    'Neck': 'Back',
    'Left Shoulder': 'Neck',
    'Right Shoulder': 'Neck',
    'Left Elbow': 'Left Shoulder',
    'Right Elbow': 'Right Shoulder',
    'Left Wrist': 'Left Elbow',
    'Right Wrist': 'Right Elbow',
    'Left Palm': 'Left Wrist',
    'Right Palm': 'Right Wrist',
    'Waist': 'Back',
    'Left Hip': 'Waist',
    'Right Hip': 'Waist',
    'Left Knee': 'Left Hip',
    'Right Knee': 'Right Hip',
    'Left Ankle': 'Left Knee',
    'Right Ankle': 'Right Knee',
    'Left Foot': 'Left Ankle',
    'Right Foot': 'Right Ankle',
}


def build_normalized_adjacency():
    """20개 관절 adjacency를 만들고 각 행을 degree로 정규화한다."""
    joint_to_index = {name: idx for idx, name in enumerate(OUTPUT_JOINTS_ORDER)}
    num_joints = len(OUTPUT_JOINTS_ORDER)
    adjacency = np.zeros((num_joints, num_joints), dtype=np.float32)

    for left_joint, right_joint in SKELETON_EDGES:
        left_idx = joint_to_index[left_joint]
        right_idx = joint_to_index[right_joint]
        adjacency[left_idx, right_idx] = 1.0
        adjacency[right_idx, left_idx] = 1.0

    degree = adjacency.sum(axis=1, keepdims=True)
    adjacency_norm = adjacency / np.maximum(degree, 1.0)
    return adjacency, adjacency_norm


ADJACENCY, ADJACENCY_NORM = build_normalized_adjacency()


def get_feature_dim(feature_mode):
    """선택한 mode가 관절 하나당 생성하는 feature 차원을 반환한다."""
    feature_dims = {
        'BASELINE': 3,
        'VECTOR': 5,
        'GRAPH_1HOP': 5,
        'GRAPH_2HOP': 7,
        'VECTOR_GRAPH_1HOP': 7,
        'VECTOR_GRAPH_2HOP': 9,
        'SKELETON_EDGE_1HOP': 5,
        'VECTOR_SKELETON_EDGE_1HOP': 7,
    }
    try:
        return feature_dims[feature_mode]
    except KeyError as exc:
        raise ValueError(
            f'Unsupported FEATURE_MODE={feature_mode}. '
            f'Expected one of {FEATURE_MODES}.'
        ) from exc


def compute_body_normalized_coordinates(joint_xy):
    """
    관절 절대좌표와 center-to-joint vector를 각각 torso length로 정규화한다.

    Args:
        joint_xy: OUTPUT_JOINTS_ORDER 순서의 좌표, shape [20, 2].

    Returns:
        torso_normalized_xy:
            image origin 기준 joint_xy / torso_length, shape [20, 2].
        center_to_joint:
            (joint_xy - hip_center) / torso_length, shape [20, 2].
        torso_length: scalar.
    """
    joint_to_index = {name: idx for idx, name in enumerate(OUTPUT_JOINTS_ORDER)}

    left_shoulder = joint_xy[joint_to_index['Left Shoulder']]
    right_shoulder = joint_xy[joint_to_index['Right Shoulder']]
    left_hip = joint_xy[joint_to_index['Left Hip']]
    right_hip = joint_xy[joint_to_index['Right Hip']]

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0

    torso_length = float(np.linalg.norm(shoulder_center - hip_center))
    safe_torso_length = max(torso_length, TORSO_EPS)

    torso_normalized_xy = joint_xy / safe_torso_length
    center_to_joint = (
        joint_xy - hip_center[None, :]
    ) / safe_torso_length

    return (
        torso_normalized_xy.astype(np.float32),
        center_to_joint.astype(np.float32),
        torso_length,
    )


def compute_parent_edge_offsets(torso_normalized_xy):
    """Back-rooted tree의 parent-to-joint 1-hop edge를 계산한다.

    Args:
        torso_normalized_xy: image-origin 관절 좌표 / torso length, [20, 2].

    Returns:
        parent_edge_offsets: (joint - parent), [20, 2].
            부모가 없는 Back의 edge는 [0, 0]이다.
    """
    expected_shape = (len(OUTPUT_JOINTS_ORDER), 2)
    if torso_normalized_xy.shape != expected_shape:
        raise ValueError(
            'torso_normalized_xy shape mismatch: '
            f'expected={expected_shape}, actual={torso_normalized_xy.shape}'
        )

    joint_to_index = {
        joint_name: joint_idx
        for joint_idx, joint_name in enumerate(OUTPUT_JOINTS_ORDER)
    }
    parent_edge_offsets = np.zeros(expected_shape, dtype=np.float32)

    for joint_name, parent_name in SKELETON_PARENT.items():
        joint_idx = joint_to_index[joint_name]
        parent_idx = joint_to_index[parent_name]
        parent_edge_offsets[joint_idx] = (
            torso_normalized_xy[joint_idx]
            - torso_normalized_xy[parent_idx]
        )

    return parent_edge_offsets


def build_joint_features(
    joint_xy,
    feature_mode,
    joint_vocab,
):
    """
    한 frame의 [20,2] 좌표를 mode에 맞는 [20,D] feature로 변환한다.
    유효 관절의 마지막 원소에는 joint_vocab[관절명]/21을 추가한다.
    """
    expected_shape = (len(OUTPUT_JOINTS_ORDER), 2)
    if joint_xy.shape != expected_shape:
        raise ValueError(
            f'joint_xy shape mismatch: expected={expected_shape}, '
            f'actual={joint_xy.shape}'
        )

    get_feature_dim(feature_mode)
    torso_normalized_xy, center_to_joint, torso_length = (
        compute_body_normalized_coordinates(joint_xy)
    )

    one_hop_mean = ADJACENCY_NORM @ torso_normalized_xy
    one_hop_offset = one_hop_mean - torso_normalized_xy
    parent_edge_offset = compute_parent_edge_offsets(torso_normalized_xy)
    feature_parts = [torso_normalized_xy]

    if feature_mode.startswith('VECTOR'):
        feature_parts.append(center_to_joint)

    if feature_mode in (
        'SKELETON_EDGE_1HOP',
        'VECTOR_SKELETON_EDGE_1HOP',
    ):
        feature_parts.append(parent_edge_offset)

    if 'GRAPH' in feature_mode:
        feature_parts.append(one_hop_offset)

        if feature_mode.endswith('2HOP'):
            two_hop_mean = ADJACENCY_NORM @ one_hop_mean
            two_hop_offset = two_hop_mean - torso_normalized_xy
            feature_parts.append(two_hop_offset)

    normalized_joint_ids = np.asarray(
        [
            [joint_vocab[joint_name] / JOINT_ID_DIVISOR]
            for joint_name in OUTPUT_JOINTS_ORDER
        ],
        dtype=np.float32,
    )
    feature_parts.append(normalized_joint_ids)

    features = np.concatenate(feature_parts, axis=-1).astype(np.float32)
    expected_feature_shape = (
        len(OUTPUT_JOINTS_ORDER),
        get_feature_dim(feature_mode),
    )
    if features.shape != expected_feature_shape:
        raise RuntimeError(
            f'Feature shape mismatch for {feature_mode}: '
            f'expected={expected_feature_shape}, actual={features.shape}'
        )

    return features, torso_length


def collect_frame_joint_coordinates(frame, view_idx):
    """기존 좌표 범위 검사와 다섯 head point 평균을 수행한다."""
    raw_joint_xy = {}
    head_xy = np.zeros(2, dtype=np.float32)

    for joint_name in JOINTS_ORDER:
        x = frame[view_idx]['pts'][joint_name]['x']
        y = frame[view_idx]['pts'][joint_name]['y']

        if x < 0 or y < 0 or x >= W or y >= H:
            return None, False

        xy = np.asarray([x, y], dtype=np.float32)
        if joint_name in HEAD_KEYS:
            head_xy += xy
        else:
            raw_joint_xy[joint_name] = xy

    raw_joint_xy['Head'] = head_xy / float(len(HEAD_KEYS))
    joint_xy = np.stack(
        [raw_joint_xy[joint_name] for joint_name in OUTPUT_JOINTS_ORDER],
        axis=0,
    )
    return joint_xy, True


def make_padding_frame(feature_dim):
    """기존 padding key 순서와 선택 mode의 feature 차원으로 zero frame을 만든다."""
    return {
        joint_name: np.zeros(feature_dim, dtype=np.float32)
        for joint_name in JOINTS_ORDER_HEAD_ver
    }


def load_vocab(vocab_path):
    with open(vocab_path['JOINT'], 'rb') as file_pointer:
        joint_vocab = pickle.load(file_pointer)

    with open(vocab_path['WORKOUT'], 'rb') as file_pointer:
        workout_vocab = pickle.load(file_pointer)

    with open(vocab_path['CONDITION'], 'rb') as file_pointer:
        condition_vocab = pickle.load(file_pointer)

    return joint_vocab, workout_vocab, condition_vocab


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build torso-normalized Embedder preprocessing features.'
    )
    parser.add_argument(
        '--data-type',
        choices=tuple(BASE_PATH.keys()),
        default=DATA_TYPE,
    )
    parser.add_argument(
        '--feature-mode',
        choices=FEATURE_MODES,
        default=FEATURE_MODE,
    )
    parser.add_argument(
        '--output-path',
        default=None,
        help='Optional output pickle path. Default includes data type and mode.',
    )
    parser.add_argument(
        '--self-test',
        action='store_true',
        help='Run shape/numerical/padding tests without reading the dataset.',
    )
    return parser.parse_args()


def run_self_test():
    """Adjacency, feature mode와 16-frame padding에 필요한 구조를 검증한다."""
    test_joint_vocab = {
        joint_name: joint_id
        for joint_id, joint_name in enumerate(
            OUTPUT_JOINTS_ORDER,
            start=2,
        )
    }
    if OUTPUT_JOINTS_ORDER[0] != 'Head':
        raise AssertionError('Real frame output must keep Head as the first key.')
    if JOINTS_ORDER_HEAD_ver[-1] != 'Head':
        raise AssertionError('Padding frame must keep the original Head-last order.')
    if set(OUTPUT_JOINTS_ORDER) != set(JOINTS_ORDER_HEAD_ver):
        raise AssertionError('Real and padding frame joint sets must match.')
    if not np.allclose(ADJACENCY, ADJACENCY.T):
        raise AssertionError('Adjacency must be symmetric.')
    if np.any(ADJACENCY.sum(axis=1) == 0):
        raise AssertionError('Every joint must have at least one neighbor.')
    if not np.allclose(ADJACENCY_NORM.sum(axis=1), 1.0):
        raise AssertionError('Every normalized adjacency row must sum to 1.')

    expected_children = set(OUTPUT_JOINTS_ORDER) - {SKELETON_ROOT}
    if set(SKELETON_PARENT) != expected_children:
        raise AssertionError(
            'SKELETON_PARENT must define exactly one parent for every non-root joint.'
        )
    for joint_name in expected_children:
        visited = set()
        current_joint = joint_name
        while current_joint != SKELETON_ROOT:
            if current_joint in visited:
                raise AssertionError('SKELETON_PARENT contains a cycle.')
            visited.add(current_joint)
            current_joint = SKELETON_PARENT[current_joint]

    joint_xy = np.stack(
        [
            np.asarray(
                [100.0 + float(idx), 200.0 + float(idx % 4) * 5.0],
                dtype=np.float32,
            )
            for idx in range(len(OUTPUT_JOINTS_ORDER))
        ],
        axis=0,
    )
    joint_to_index = {name: idx for idx, name in enumerate(OUTPUT_JOINTS_ORDER)}
    joint_xy[joint_to_index['Left Shoulder']] = [90.0, 180.0]
    joint_xy[joint_to_index['Right Shoulder']] = [110.0, 180.0]
    joint_xy[joint_to_index['Left Hip']] = [90.0, 200.0]
    joint_xy[joint_to_index['Right Hip']] = [110.0, 200.0]

    torso_normalized_xy, center_to_joint, torso_length = (
        compute_body_normalized_coordinates(joint_xy)
    )
    if not np.allclose(
        torso_normalized_xy[joint_to_index['Left Shoulder']],
        [4.5, 9.0],
    ):
        raise AssertionError('Joint/torso normalization value check failed.')
    if not np.allclose(
        center_to_joint[joint_to_index['Left Shoulder']],
        [-0.5, -1.0],
    ):
        raise AssertionError('Center-to-joint vector value check failed.')
    if not np.isclose(torso_length, 20.0):
        raise AssertionError(
            f'Expected torso_length=20, got {torso_length}.'
        )

    if np.allclose(center_to_joint, torso_normalized_xy):
        raise AssertionError(
            'P and center-to-joint vector must not be treated as identical.'
        )

    left_elbow_idx = joint_to_index['Left Elbow']
    left_shoulder_idx = joint_to_index['Left Shoulder']
    left_wrist_idx = joint_to_index['Left Wrist']
    expected_elbow_neighbor = (
        torso_normalized_xy[left_shoulder_idx]
        + torso_normalized_xy[left_wrist_idx]
    ) / 2.0
    actual_elbow_neighbor = (
        ADJACENCY_NORM[left_elbow_idx] @ torso_normalized_xy
    )
    if not np.allclose(actual_elbow_neighbor, expected_elbow_neighbor):
        raise AssertionError('Left Elbow 1-hop neighbor mean check failed.')

    parent_edge_offset = compute_parent_edge_offsets(torso_normalized_xy)
    back_idx = joint_to_index['Back']
    waist_idx = joint_to_index['Waist']
    if not np.allclose(parent_edge_offset[back_idx], [0.0, 0.0]):
        raise AssertionError('Back root edge must be [0, 0].')
    if not np.allclose(
        parent_edge_offset[waist_idx],
        torso_normalized_xy[waist_idx] - torso_normalized_xy[back_idx],
    ):
        raise AssertionError('Back-to-Waist parent edge check failed.')
    if not np.allclose(
        parent_edge_offset[left_wrist_idx],
        torso_normalized_xy[left_wrist_idx]
        - torso_normalized_xy[left_elbow_idx],
    ):
        raise AssertionError('Left-Elbow-to-Left-Wrist parent edge check failed.')

    for feature_mode in FEATURE_MODES:
        feature_dim = get_feature_dim(feature_mode)
        features, mode_torso_length = build_joint_features(
            joint_xy,
            feature_mode,
            joint_vocab=test_joint_vocab,
        )
        expected_shape = (len(OUTPUT_JOINTS_ORDER), feature_dim)
        if features.shape != expected_shape:
            raise AssertionError(
                f'{feature_mode}: expected={expected_shape}, '
                f'actual={features.shape}'
            )
        if not np.isfinite(features).all():
            raise AssertionError(f'{feature_mode}: non-finite feature detected.')
        expected_joint_ids = np.asarray(
            [
                test_joint_vocab[joint_name] / 21.0
                for joint_name in OUTPUT_JOINTS_ORDER
            ],
            dtype=np.float32,
        )
        if not np.allclose(features[:, -1], expected_joint_ids):
            raise AssertionError(
                f'{feature_mode}: normalized joint ID must be the last feature.'
            )
        if (
            feature_mode == 'BASELINE'
            and not np.allclose(features[:, :2], torso_normalized_xy)
        ):
            raise AssertionError(
                'BASELINE must contain torso-normalized coordinates and joint ID.'
            )
        if (
            feature_mode.startswith('VECTOR')
            and not np.allclose(features[:, 2:4], center_to_joint)
        ):
            raise AssertionError(
                f'{feature_mode}: vector must be the torso-normalized '
                'center-to-joint offset.'
            )
        if (
            feature_mode == 'SKELETON_EDGE_1HOP'
            and not np.allclose(features[:, 2:4], parent_edge_offset)
        ):
            raise AssertionError(
                'SKELETON_EDGE_1HOP must contain parent-to-joint edges.'
            )
        if (
            feature_mode == 'VECTOR_SKELETON_EDGE_1HOP'
            and not np.allclose(features[:, 4:6], parent_edge_offset)
        ):
            raise AssertionError(
                'VECTOR_SKELETON_EDGE_1HOP must contain parent-to-joint '
                'edges after the center vector.'
            )
        if not np.isclose(mode_torso_length, 20.0):
            raise AssertionError(
                f'{feature_mode}: torso length check failed.'
            )

        padding_frame = make_padding_frame(feature_dim)
        if list(padding_frame.keys()) != JOINTS_ORDER_HEAD_ver:
            raise AssertionError(
                f'{feature_mode}: padding key order changed.'
            )
        if any(
            value.shape != (feature_dim,)
            or value.dtype != np.float32
            or np.any(value != 0)
            for value in padding_frame.values()
        ):
            raise AssertionError(
                f'{feature_mode}: padding value check failed.'
            )

        scaled_features, _ = build_joint_features(
            joint_xy * 3.5,
            feature_mode,
            joint_vocab=test_joint_vocab,
        )
        if not np.allclose(features, scaled_features, atol=1e-5):
            raise AssertionError(
                f'{feature_mode}: torso scale invariance check failed.'
            )

    translated_xy = joint_xy + np.asarray(
        [120.0, 250.0],
        dtype=np.float32,
    )
    translated_vector, _ = build_joint_features(
        translated_xy,
        'VECTOR',
        joint_vocab=test_joint_vocab,
    )
    original_vector, _ = build_joint_features(
        joint_xy,
        'VECTOR',
        joint_vocab=test_joint_vocab,
    )
    if np.allclose(original_vector[:, :2], translated_vector[:, :2]):
        raise AssertionError(
            'Image-origin P should change when the person is translated.'
        )
    if not np.allclose(
        original_vector[:, 2:],
        translated_vector[:, 2:],
        atol=1e-5,
    ):
        raise AssertionError(
            'Center-to-joint vector should be translation invariant.'
        )

    print(
        '[SELF-TEST PASS] Embedder dataset structure, torso/vector/graph '
        'and skeleton-edge features with mode-aware padding'
    )


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    data_type = args.data_type
    feature_mode = args.feature_mode
    feature_dim = get_feature_dim(feature_mode)

    joint_vocab, workout_vocab, condition_vocab = load_vocab(
        VOCAB_PATH[data_type]
    )

    json_lst = [
        path
        for base_path in BASE_PATH[data_type]
        for path in glob(base_path + '/*/*/*.json')
        if not path.endswith('3d.json')
    ]

    videos = []
    degenerate_torso_count = 0

    for idx in tqdm(
        range(len(json_lst)),
        leave=True,
        desc='extracting pts and building Embedder features',
    ):
        json_file = json_lst[idx]
        with open(json_file, 'r') as file_pointer:
            data = json.load(file_pointer)

        exercise_name = data['type_info']['exercise']
        if not IS_CONTAIN_HARD_EXERCISE:
            if exercise_name in HARD_EXERCISE:
                continue

        if not 0 < len(data['frames']) <= MAX_FRAME:
            continue

        if IS_INTEGRATE_PUSHUP:
            if exercise_name in ['푸시업', '니푸쉬업']:
                exercise_name = '푸시업'

        if IS_INTEGRATE_ROW:
            if exercise_name in ['덤벨 벤트오버 로우', '바벨 로우']:
                exercise_name = '바벨-덤벨 로우'

        if exercise_name == '바벨 컬 ':
            exercise_name = '바벨 컬'

        # 기존 condition 정제 로직 유지.
        condition_lst = []
        for condition_info in data['type_info']['conditions']:
            condition_str = condition_info['condition']
            if (exercise_name == '바벨-덤벨 로우' and condition_str in ['바벨 궤적과 몸 밀착', '덤벨 궤적과 몸 밀착']):
                condition_str = '바벨-덤벨 궤적과 몸 밀착'

            condition_idx = condition_vocab[condition_str]
            value = int(condition_info['value'])
            condition_lst.append([condition_idx, value])

        for view_idx in VIEWS:
            a_video_buffer = {}

            for frame_idx in range(len(data['frames'])):
                frame_key = str(frame_idx)
                joint_xy, in_range = collect_frame_joint_coordinates(
                    data['frames'][frame_idx],
                    view_idx,
                )
                if not in_range:
                    break

                features, torso_length = build_joint_features(
                    joint_xy,
                    feature_mode,
                    joint_vocab=joint_vocab,
                )
                if torso_length < TORSO_EPS:
                    degenerate_torso_count += 1

                feature_by_joint = {
                    joint_name: features[joint_idx]
                    for joint_idx, joint_name
                    in enumerate(OUTPUT_JOINTS_ORDER)
                }

                # 기존 실제 frame의 Head-first dictionary 순서를 유지한다.
                a_video_buffer[frame_key] = {
                    joint_name: feature_by_joint[joint_name]
                    for joint_name in OUTPUT_JOINTS_ORDER
                }

            else:
                keys = list(a_video_buffer.keys())
                is_sorted = keys == sorted(keys, key=lambda key: int(key))
                if not is_sorted:
                    raise ValueError(
                        f"[Error] Frame keys in video '{len(videos)}' "
                        f"of view '{view_idx}' for exercise "
                        f"'{exercise_name}' are not in ascending order."
                    )

                if len(a_video_buffer) < MAX_FRAME:
                    for padding_idx in range(
                        len(a_video_buffer.keys()),
                        MAX_FRAME,
                    ):
                        a_video_buffer.setdefault(
                            str(padding_idx),
                            make_padding_frame(feature_dim),
                        )

                videos.append(
                    [
                        a_video_buffer,
                        workout_vocab[exercise_name],
                        condition_lst,
                    ]
                )

    if args.output_path is None:
        output_path = (
            '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/'
            'fitness_data_preprocess/Embedder Dataset/'
            f'{feature_mode}/'
            f'{row_mode_dir}/'
            f'{hard_exercise_mode_dir}/'
            f'{pushup_mode_dir}/'
            f'{joint_id_mode_dir}/'
            f'{data_type}.pkl'
        )
    else:
        output_path = args.output_path

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as file_pointer:
        pickle.dump(videos, file_pointer, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'[DONE] DATA_TYPE={data_type}')
    print(f'[DONE] FEATURE_MODE={feature_mode}')
    print(f'[DONE] FEATURE_DIM={feature_dim}')
    print(f'[DONE] NUM_VIDEOS={len(videos)}')
    print(f'[DONE] DEGENERATE_TORSO_COUNT={degenerate_torso_count}')
    print(f'[DONE] OUTPUT={output_path}')


if __name__ == '__main__':
    main()
