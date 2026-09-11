import os
import pickle
import re
import torch
from types import SimpleNamespace



def parse_arcface_parameter(path, parameter_name):
    pattern = rf'(?:^|[\s/]){re.escape(parameter_name)}:(\d+(?:\.\d+)?)(?=$|[\s/])'
    match = re.search(pattern, path)

    if match is None:
        raise ValueError(
            f"ArcFace parameter '{parameter_name}'을 경로에서 찾을 수 없습니다: {path}"
        )

    return float(match.group(1))


config = SimpleNamespace()

#
# MODE
config.TASK_MODE = 'TRAIN' # ['TRAIN', 'VAL']
config.DATA_MODE = 'compare_aihub' # ['compare_aihub', 'custom']
config.EMB_MODE = 'RELATIVE_BASIS' # ['RELATIVE_BASIS', 'RELATIVE', 'RELATIVEwID', 'RELATIVEwIDNorm', 'BASIS']
config.FEATURE_MODES = 'VECTOR_GRAPH_1HOP'
# 'BASELINE', 'VECTOR', 'GRAPH_1HOP', 'GRAPH_2HOP',
# 'VECTOR_GRAPH_1HOP', 'VECTOR_GRAPH_2HOP',
# 'SKELETON_EDGE_1HOP', 'VECTOR_SKELETON_EDGE_1HOP', 'VECTOR_GRAPH_SKELETON_EDGE_1HOP'

# Data Version
IS_CONTAIN_HARD_EXERCISE = True # False
hard_exercise_mode_dir = 'hard_exercise_included' if IS_CONTAIN_HARD_EXERCISE else 'hard_exercise_excluded'

IS_INTEGRATE_ROW = True  # False 바벨로우 덤벨로우 통합할래 말래
row_mode_dir = 'row_integrated' if IS_INTEGRATE_ROW else 'row_not_integrated'

if config.EMB_MODE == 'RELATIVEwID':
  IS_JOINT_ID_NORM = False
elif config.EMB_MODE == 'RELATIVEwIDNorm':
  IS_JOINT_ID_NORM = True
else:
  IS_JOINT_ID_NORM = False
joint_id_mode_dir = 'joint_id_normalized' if IS_JOINT_ID_NORM else 'joint_id_raw'

IS_INTEGRATE_PUSHUP = True
pushup_mode_dir = 'pushup_integrated' if IS_INTEGRATE_PUSHUP else 'pushup_not_integrated'
pushup_merge_reduction = 1 if IS_INTEGRATE_PUSHUP and IS_CONTAIN_HARD_EXERCISE else 0
#
# Freeze
config.BASIS_FREEZE = False
config.RELATIVE_FREEZE = False

config.USE_ARCFACE = False
config.EMB_INIT = False # True: initialization, False: pretrained model load
#

# GPU / WORKERS / BATCH
config.GPUS = '1'
config.WORKERS = 0
config.DEVICE = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")
config.BATCH_SIZE = 32

# SEED
config.SEED = 42
# DATA
config.FEATURE_MODES_IN_DIM = {
  'BASELINE': 3,
  'VECTOR': 5,
  'GRAPH_1HOP': 5,
  'GRAPH_2HOP': 7,
  'VECTOR_GRAPH_1HOP': 7,
  'VECTOR_GRAPH_2HOP': 9,
  'SKELETON_EDGE_1HOP': 5,
  'VECTOR_SKELETON_EDGE_1HOP': 7,
  'VECTOR_GRAPH_SKELETON_EDGE_1HOP': 9,
}
ID_AWARE_EMB_MODES = (
  'RELATIVEwID',
  'RELATIVEwIDNorm',
)

full_dim = config.FEATURE_MODES_IN_DIM[config.FEATURE_MODES]

config.IN_FEAT = (
  full_dim
  if config.EMB_MODE in ID_AWARE_EMB_MODES
  else full_dim - 1
)
#
VALID_DATA_MODES = ('compare_aihub', 'custom')
if config.DATA_MODE not in VALID_DATA_MODES:
    raise ValueError(
        f'Unsupported DATA_MODE: {config.DATA_MODE}. '
        f'Choose one of {VALID_DATA_MODES}'
    )

CUSTOM_EMBEDDER_DATASET_ROOT = (
    '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/'
    'fitness_data_preprocess/Embedder Dataset'
)
CUSTOM_VOCAB_ROOT = (
    '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/'
    'fitness_data_preprocess/vocab'
)
COMPARE_AIHUB_ROOT = (
    '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/'
    'fitness_data_preprocess/compare_aihub'
)

if config.DATA_MODE == 'compare_aihub':
    compare_aihub_feature_modes = (
        'BASELINE',
        'GRAPH_1HOP',
        'VECTOR_GRAPH_1HOP',
        'VECTOR_GRAPH_SKELETON_EDGE_1HOP',
    )
    if config.FEATURE_MODES not in compare_aihub_feature_modes:
        raise ValueError(
            'Unsupported compare_aihub feature mode: '
            f'{config.FEATURE_MODES}. Choose one of '
            f'{compare_aihub_feature_modes}'
        )
    if config.EMB_MODE == 'RELATIVEwIDNorm':
        raise ValueError(
            'compare_aihub에는 joint_id_normalized 데이터가 없습니다.'
        )

    config.DATASET_NAME = 'compare_aihub_joint_id_raw'
    embedder_dataset_dir = os.path.join(
        COMPARE_AIHUB_ROOT,
        'Embedder Dataset_to_compare_aihub',
        config.FEATURE_MODES,
        'joint_id_raw',
    )
    vocab_tree_root = os.path.join(
        COMPARE_AIHUB_ROOT,
        'vocab_to_compare_aihub',
    )
else:
    config.DATASET_NAME = (
        f'{hard_exercise_mode_dir}_'
        f'{row_mode_dir}_'
        f'{pushup_mode_dir}_'
        f'{joint_id_mode_dir}'
    )
    embedder_dataset_dir = os.path.join(
        CUSTOM_EMBEDDER_DATASET_ROOT,
        config.FEATURE_MODES,
        row_mode_dir,
        hard_exercise_mode_dir,
        pushup_mode_dir,
        joint_id_mode_dir,
    )
    vocab_tree_root = os.path.join(
        CUSTOM_VOCAB_ROOT,
        row_mode_dir,
        hard_exercise_mode_dir,
        pushup_mode_dir,
    )

config.TRAIN_DATA_PATH = os.path.join(embedder_dataset_dir, 'TRAIN.pkl')
config.VALID_DATA_PATH = os.path.join(embedder_dataset_dir, 'VALID.pkl')

train_vocab_dir = os.path.join(vocab_tree_root, 'TRAIN')
valid_vocab_dir = os.path.join(vocab_tree_root, 'VALID')

config.TRAIN_JOINT_VOCAB_PATH = os.path.join(train_vocab_dir, 'joint_vocab.pkl')
config.TRAIN_WORKOUT_VOCAB_PATH = os.path.join(train_vocab_dir, 'workout_vocab.pkl')
config.TRAIN_CONDITION_VOCAB_PATH = os.path.join(train_vocab_dir, 'condition_vocab.pkl')

config.VALID_JOINT_VOCAB_PATH = os.path.join(valid_vocab_dir, 'joint_vocab.pkl')
config.VALID_WORKOUT_VOCAB_PATH = os.path.join(valid_vocab_dir, 'workout_vocab.pkl')
config.VALID_CONDITION_VOCAB_PATH = os.path.join(valid_vocab_dir, 'condition_vocab.pkl')

config.IMG_SIZE = [1920, 1080]

with open(config.TRAIN_WORKOUT_VOCAB_PATH, 'rb') as file_pointer:
    train_workout_vocab = pickle.load(file_pointer)
with open(config.TRAIN_CONDITION_VOCAB_PATH, 'rb') as file_pointer:
    train_condition_vocab = pickle.load(file_pointer)
with open(config.VALID_WORKOUT_VOCAB_PATH, 'rb') as file_pointer:
    valid_workout_vocab = pickle.load(file_pointer)
with open(config.VALID_CONDITION_VOCAB_PATH, 'rb') as file_pointer:
    valid_condition_vocab = pickle.load(file_pointer)

config.CLASS_NUM = len(train_workout_vocab)
config.NUM_CONDITIONS = len(train_condition_vocab)

expected_workout_ids = set(range(22, 22 + config.CLASS_NUM))
expected_condition_ids = set(range(
    22 + config.CLASS_NUM,
    22 + config.CLASS_NUM + config.NUM_CONDITIONS,
))
if set(train_workout_vocab.values()) != expected_workout_ids:
    raise ValueError('TRAIN workout vocab IDs are not contiguous.')
if set(train_condition_vocab.values()) != expected_condition_ids:
    raise ValueError('TRAIN condition vocab IDs are not contiguous.')

for name, raw_id in valid_workout_vocab.items():
    if train_workout_vocab.get(name) != raw_id:
        raise ValueError(f'VALID workout ID mismatch: {name}')
for name, raw_id in valid_condition_vocab.items():
    if train_condition_vocab.get(name) != raw_id:
        raise ValueError(f'VALID condition ID mismatch: {name}')

config.NUM_JOINTS = 22  # 0, 1 = PAD, SEP, others joints
config.MAX_FRAMES = 16
# config.CLASS_NUM = 41 if config.TASK_MODE == 'TRAIN' else 27
config.JOINTS_NAME = [
    'Head', 'Left Shoulder', 'Right Shoulder',
    'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist',
    'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee',
    'Left Ankle', 'Right Ankle',
    'Neck', 'Left Palm',
    'Right Palm', 'Back',
    'Waist', 'Left Foot',
    'Right Foot'
    ]

# PRETRAINED MODEL PATH
config.ARCFACE_PARAM = {}
if config.EMB_MODE == 'RELATIVE_BASIS':
    #
    # BASELINE
    # config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/BASELINE/B+R/[B+R] LAYERS_NUM:4 IN_DIM:2 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_metric_learning_model.pth.tar'
    # config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/BASELINE/B+R/[B+R] LAYERS_NUM:4 IN_DIM:2 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_nn_embedding.pt'
    # config.PRETRAINED_ARCFACE_CLASSIFIER_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/BASELINE/B+R/[B+R] LAYERS_NUM:4 IN_DIM:2 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_arcface_classifier.pt'

    # GRAPH_1HOP
    # config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/GRAPH_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:4 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_metric_learning_model.pth.tar'
    # config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/GRAPH_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:4 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_nn_embedding.pt'
    # config.PRETRAINED_ARCFACE_CLASSIFIER_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/GRAPH_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:4 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_arcface_classifier.pt'

    # VECTOR_GRAPH_1HOP
    config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/VECTOR_GRAPH_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:6 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_metric_learning_model.pth.tar'
    config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/VECTOR_GRAPH_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:6 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_nn_embedding.pt'
    config.PRETRAINED_ARCFACE_CLASSIFIER_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/VECTOR_GRAPH_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:6 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_arcface_classifier.pt'

    # VECTOR_GRAPH_SKELETON_EDGE_1HOP
    # config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/VECTOR_GRAPH_SKELETON_EDGE_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:8 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_metric_learning_model.pth.tar'
    # config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/VECTOR_GRAPH_SKELETON_EDGE_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:8 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_nn_embedding.pt'
    # config.PRETRAINED_ARCFACE_CLASSIFIER_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/VECTOR_GRAPH_SKELETON_EDGE_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:8 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_arcface_classifier.pt'

    # VECTOR_SKELETON_EDGE_1HOP
    # config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/VECTOR_SKELETON_EDGE_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:6 DIM:768 S:10 M:0.1 hard_exercise_included_row_integrated_pushup_integrated_joint_id_normalized/weights/best_metric_learning_model.pth.tar'
    # config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/VECTOR_SKELETON_EDGE_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:6 DIM:768 S:10 M:0.1 hard_exercise_included_row_integrated_pushup_integrated_joint_id_normalized/weights/best_nn_embedding.pt'
    # config.PRETRAINED_ARCFACE_CLASSIFIER_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/VECTOR_SKELETON_EDGE_1HOP/B+R/[B+R] LAYERS_NUM:4 IN_DIM:6 DIM:768 S:10 M:0.1 hard_exercise_included_row_integrated_pushup_integrated_joint_id_normalized/weights/best_arcface_classifier.pt'
    #
    config.ARCFACE_PARAM['s'] = parse_arcface_parameter(config.PRETRAINED_PATH, 'S')
    config.ARCFACE_PARAM['m'] = parse_arcface_parameter(config.PRETRAINED_PATH, 'M')


elif config.EMB_MODE == 'RELATIVE':
    #
    config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/BASELINE/R/[R] LAYERS_NUM:4 IN_DIM:2 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_metric_learning_model.pth.tar'
    config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/BASELINE/R/[R] LAYERS_NUM:4 IN_DIM:2 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_nn_embedding.pt'
    config.PRETRAINED_ARCFACE_CLASSIFIER_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/BASELINE/R/[R] LAYERS_NUM:4 IN_DIM:2 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_arcface_classifier.pt'
    #
    config.ARCFACE_PARAM['s'] = parse_arcface_parameter(config.PRETRAINED_PATH, 'S')
    config.ARCFACE_PARAM['m'] = parse_arcface_parameter(config.PRETRAINED_PATH, 'M')

#
elif config.EMB_MODE == 'RELATIVEwID':
    #
    config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/BASELINE/RwID/[RwID] LAYERS_NUM:4 IN_DIM:3 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_metric_learning_model.pth.tar'
    config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/BASELINE/RwID/[RwID] LAYERS_NUM:4 IN_DIM:3 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_nn_embedding.pt'
    config.PRETRAINED_ARCFACE_CLASSIFIER_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/BASELINE/RwID/[RwID] LAYERS_NUM:4 IN_DIM:3 DIM:768 S:10 M:0.1 compare_aihub_joint_id_raw/weights/best_arcface_classifier.pt'
    #
    config.ARCFACE_PARAM['s'] = parse_arcface_parameter(config.PRETRAINED_PATH, 'S')
    config.ARCFACE_PARAM['m'] = parse_arcface_parameter(config.PRETRAINED_PATH, 'M')

elif config.EMB_MODE == 'RELATIVEwIDNorm':
    #
    config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/R/[R] LAYERS_NUM:4 IN_DIM:2 DIM:768 S:10 M:0.1/weights/metric_learning_model.pth.tar'
    config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/R/[R] LAYERS_NUM:4 IN_DIM:2 DIM:768 S:10 M:0.1/weights/nn_embedding.pt'
    config.PRETRAINED_ARCFACE_CLASSIFIER_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/R/[R] LAYERS_NUM:4 IN_DIM:2 DIM:768 S:10 M:0.1/weights/arcface_classifier.pt'
    #
    config.ARCFACE_PARAM['s'] = parse_arcface_parameter(config.PRETRAINED_PATH, 'S')
    config.ARCFACE_PARAM['m'] = parse_arcface_parameter(config.PRETRAINED_PATH, 'M')

# ========================
OUT_FEAT = int(config.PRETRAINED_PATH.split()[3].split(':')[-1])
NUM_LAYER = int(re.search(r'LAYERS_NUM:(\d+)', config.PRETRAINED_PATH).group(1)) # for train
ACTIV = 'GELU'
#
config.OUT_FEAT = OUT_FEAT
config.NUM_LAYER = NUM_LAYER # EMB_LAYER
config.ACTIV = ACTIV
#

config.EMB_INIT = False # True: initialization, False: pretrained model load
if config.BASIS_FREEZE and config.RELATIVE_FREEZE:
    is_freeze = 'B,R Freeze'
elif not config.BASIS_FREEZE and not config.RELATIVE_FREEZE:
    is_freeze = 'B,R Unfreeze'
else:
    raise ValueError('Both B and R must be either frozen or unfrozen.')

if config.USE_ARCFACE:
    is_arcface = 'Use_ArcFace'
elif not config.USE_ARCFACE:
    is_arcface = 'Not_Use_ArcFace'

if config.EMB_INIT:
    is_init = 'Initialized Embedder'
elif not config.EMB_INIT:
    is_init = 'Not Initialized Embedder'

config.FILE_NAME = f'[{config.DATA_MODE}][{config.EMB_MODE}][{config.FEATURE_MODES}][{is_freeze}][{is_arcface}][{is_init}][NLayer:{NUM_LAYER}][NEnc'

print('====== MODE Configuration ======')
print('Task Mode: {}'.format(config.TASK_MODE))
print('Data Mode: {}'.format(config.DATA_MODE))
print('Dataset Name: {}'.format(config.DATASET_NAME))
print('Feature Mode: {}'.format(config.FEATURE_MODES))
print('EMB_MODE: {}\n'.format(config.EMB_MODE))

print('====== Path Configuration ======')
print('BASIS WEIGHT PATH: {}'.format(config.PRETRAINED_EMB_PATH))
print('RELATIVE WEIGHT PATH: {}'.format(config.PRETRAINED_PATH))
print('ARCFACE WEIGHT PATH: {}'.format(config.PRETRAINED_ARCFACE_CLASSIFIER_PATH))
print(f'TRAIN_DATA_PATH: {config.TRAIN_DATA_PATH}')
print(f'VALID_DATA_PATH: {config.VALID_DATA_PATH}\n')

print('====== Freeze Configuration ======')
print('BASIS_FREEZE: {}'.format(config.BASIS_FREEZE))
print('RELATIVE_FREEZE: {}'.format(config.RELATIVE_FREEZE))
print('USE_ARCFACE: {}\n'.format(config.USE_ARCFACE))

print('====== Other Configuration ======')
print('CLASS_NUM: {}'.format(config.CLASS_NUM))
print('NUM_CONDITIONS: {}'.format(config.NUM_CONDITIONS))
print('IN_FEAT: {}'.format(config.IN_FEAT))
print('OUT_FEAT: {}'.format(config.OUT_FEAT))
print('EMB_LAYER: {}'.format(config.NUM_LAYER))
print('FILE_NAME: {}'.format(config.FILE_NAME))
