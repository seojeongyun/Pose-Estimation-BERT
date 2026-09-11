import torch
from types import SimpleNamespace


config = SimpleNamespace()

# MODE
config.TASK_MODE = 'TRAIN' # ['TRAIN', 'VAL']
config.EMB_MODE = 'RELATIVE_BASIS' # ['RELATIVE_BASIS', 'RELATIVE']
config.BASIS_FREEZE = False
config.RELATIVE_FREEZE = False

# SEED
config.SEED = 478

# DATA
config.JOINTS_NAME = [
    'Left Shoulder', 'Right Shoulder',
    'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist',
    'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee',
    'Left Ankle', 'Right Ankle',
    'Neck', 'Left Palm',
    'Right Palm', 'Back',
    'Waist', 'Left Foot',
    'Right Foot', 'Head'
    ]

config.IMG_SIZE = [1920, 1080]
config.DATA_PATH = '/storage/hjchoi/BERTSUMFORHPE/embedder_train.json'
config.NUM_JOINTS = 22  # 0, 1 = PAD, SEP, others joints
config.MAX_FRAMES = 21

if config.TASK_MODE == 'TRAIN':
    config.CLASS_NUM = 41
else:
    config.CLASS_NUM = 27

# GPU / WORKERS
config.GPUS = '0'
config.WORKERS = 0
config.DEVICE = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")

# if you use relative model, set the config.USE_EMBEDDING = False
if config.EMB_MODE == 'RELATIVE_BASIS':
    config.USE_EMBEDDING = True
    config.PRETRAINED_PATH = '/storage/hjchoi/BERTSUMFORHPE/checkpoint/find_optimal_model/[Basis+Relative] LAYERS_NUM:4 DIM:768 ACT:GELU s:10 m:0.1 norm.pth.tar'
elif config.EMB_MODE == 'RELATIVE':
    config.USE_EMBEDDING = False
    config.PRETRAINED_PATH = '/storage/hjchoi/BERTSUMFORHPE/checkpoint/find_optimal_model/[Relative] LAYERS_NUM:4 DIM:768 ACT:GELU s:10 m:0.1 norm.pth.tar'

config.PRETRAINED_EMB_PATH = '/storage/hjchoi/BERTSUMFORHPE/checkpoint/find_sm/[s,m Basis+Relative] LAYERS_NUM:6 DIM:768 ACT:GELU s:10 m:0.45/nn_embedding_model.pt'

# Model
# if config.USE_EMBEDDING:
#     OUT_FEAT = int(config.PRETRAINED_PATH.split('/')[6].split()[3].split(':')[-1])
#     NUM_LAYER = int(config.PRETRAINED_PATH.split('/')[6].split()[2].split(':')[-1])
#     ACTIV = config.PRETRAINED_PATH.split('/')[6].split()[4].split(':')[-1]
# else:
OUT_FEAT = int(config.PRETRAINED_PATH.split('/')[6].split()[2].split(':')[-1])
NUM_LAYER = int(config.PRETRAINED_PATH.split('/')[6].split()[1].split(':')[-1])
ACTIV = config.PRETRAINED_PATH.split('/')[6].split()[3].split(':')[-1]
#
config.IN_FEAT = 4
config.OUT_FEAT = OUT_FEAT
config.NUM_LAYER = NUM_LAYER ##EMB_LAYER
config.ACTIV = ACTIV
#

# SAVE DIR NAME
if config.EMB_MODE == 'RELATIVE_BASIS':
    config.DIR_PATH = config.EMB_MODE + "/basis_{}".format(config.BASIS_FREEZE) \
                                      + "/relative_{}".format(config.RELATIVE_FREEZE) \
                                      + "NUM_EMB_LAYER:{}".format(NUM_LAYER)\


elif config.EMB_MODE == 'RELATIVE':
    config.DIR_PATH = config.EMB_MODE + "/relative_{}".format(config.RELATIVE_FREEZE)\
                                      + "NUM_LAYER:{}".format(NUM_LAYER) \



# OPERATION
config.BATCH_SIZE = 32
print()
print('####### CONFIG #######')
print('BASIS_FREEZE: {}'.format(config.BASIS_FREEZE))
print('RELATIVE_FREEZE: {}'.format(config.RELATIVE_FREEZE))
print('EMB_MODE: {}'.format(config.EMB_MODE))
print('EMB_LAYER: {}'.format(config.NUM_LAYER))
