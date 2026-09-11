'''
    AI Hub에서 제공하는 fitness 데이터셋을 전처리 하기 위한 스크립트 파일

'''

import os
import pickle

from glob import glob
from tqdm import tqdm
from collections import defaultdict

counter = defaultdict(int)

import json, numpy as np

DATA_TYPE = 'VALID' # 'TRAIN' 'VALID'

IS_CONTAIN_HARD_EXERCISE = True # True: Hard Exercise 포함한 데이터를 만들겠다.
IS_INTEGRATE_ROW = True  # True: 바벨로우-덤벨로우 통합
IS_INTEGRATE_PUSHUP = True

MAX_FRAME = 16
W, H = 1920, 1080
TORSO_EPS = 1e-6
JOINT_ID_DIVISOR = 21.0

row_mode_dir = 'row_integrated' if IS_INTEGRATE_ROW else 'row_not_integrated'
hard_exercise_mode_dir = 'hard_exercise_included' if IS_CONTAIN_HARD_EXERCISE else 'hard_exercise_excluded'
pushup_mode_dir = 'pushup_integrated' if IS_INTEGRATE_PUSHUP else 'pushup_not_integrated'

# Configuration
BASE_PATH = {
    'TRAIN': ['/storage/jysuh/fitness/fitness/train/label/furniture_Labeling',
              '/storage/jysuh/fitness/fitness/train/label/body_Labeling_new_220128',
              '/storage/jysuh/fitness/fitness/train/label/barbell_dumbbell_Labeling_new_220128'],
    'VALID': ['/storage/jysuh/fitness/fitness/validation/label/barbell_dumbbell_Labeling',
              '/storage/jysuh/fitness/fitness/validation/label/body_Labeling',
              '/storage/jysuh/fitness/fitness/validation/label/furniture_Labeling']
             }

JOINTS_ORDER = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
          'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
          'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee',
          'Right Knee', 'Left Ankle', 'Right Ankle', 'Neck', 'Left Palm',
          'Right Palm', 'Back', 'Waist', 'Left Foot', 'Right Foot']

# 헤드가 포함된 관절 순서 리스트
JOINTS_ORDER_HEAD_ver = ['Head', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
                'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee',
                'Left Ankle', 'Right Ankle', 'Neck', 'Left Palm', 'Right Palm',
                'Back', 'Waist', 'Left Foot', 'Right Foot']

ALL_WORKOUT_LIST = ['풀업', '랫풀 다운', '딥스', '케이블 푸시 다운', '로잉머신',
                '페이스 풀', '케이블 크런치', '행잉 레그 레이즈', '푸시업',
                '라잉 레그 레이즈', '시저크로스', '니푸쉬업', '바이시클 크런치',
                '크런치', 'Y - Exercise', '플랭크', '힙쓰러스트', '스텝 포워드 다이나믹 런지',
                '버피 테스트', '스탠딩 니업', '스탠딩 사이드 크런치', '스텝 백워드 다이나믹 런지',
                '사이드 런지', '크로스 런지', '바벨 로우', '업라이트로우', '바벨 스티프 데드리프트',
                '프런트 레이즈', '굿모닝', '라잉 트라이셉스 익스텐션', '덤벨 인클라인 체스트 플라이',
                '덤벨 풀 오버', '덤벨 체스트 플라이', '바벨 스쿼트', '덤벨 벤트오버 로우',
                    '바벨 데드리프트', '바벨 런지', '바벨 컬 ', '덤벨 컬', '오버 헤드 프레스', '사이드 레터럴 레이즈']

VIEWS = ['view1', 'view2', 'view3', 'view4', 'view5']

HARD_EXERCISE = ['바이시클 크런치', '크런치', '시저크로스', '라잉 레그 레이즈', '플랭크', '힙쓰러스트', '푸시업', '니푸쉬업']

# Train: Train 전체 운동 개수 - Hard Ex 개수 - 로우 통합 = 41 - 8 - 1 = 32
# Valid: Valid 전체 운동 개수 - Hard Ex 개수 - 로우 통합 = 27 - 8 - 1 = 18
pushup_merge_reduction = (
      1
      if IS_INTEGRATE_PUSHUP and IS_CONTAIN_HARD_EXERCISE
      else 0
  )

NUMBER_of_WORKOUT = {
  'TRAIN': (
      41
      - (0 if IS_CONTAIN_HARD_EXERCISE else len(HARD_EXERCISE))
      - (1 if IS_INTEGRATE_ROW else 0)
      - pushup_merge_reduction
  ),
  'VALID': (
      27
      - (0 if IS_CONTAIN_HARD_EXERCISE else len(HARD_EXERCISE))
      - (1 if IS_INTEGRATE_ROW else 0)
      - pushup_merge_reduction
  ),
}
vocab = {'PAD': 0, 'SEP' : 1}
#
joint_vocab = {}
workout_vocab = {}
condition_vocab = {}
train_workout_vocab = {}
train_condition_vocab = {}

#
VOCAB_DIR = "/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/fitness_data_preprocess/vocab"
VOCAB_TREE_ROOT = os.path.join(
      VOCAB_DIR,
      row_mode_dir,
      hard_exercise_mode_dir,
      pushup_mode_dir,
  )
OUTPUT_VOCAB_DIR = os.path.join(VOCAB_TREE_ROOT, DATA_TYPE)

# VALID에는 실제로 등장한 항목만 저장하고, integer ID는 TRAIN vocab에서 가져온다.
if DATA_TYPE == 'VALID':
    try:
        with open(os.path.join(VOCAB_TREE_ROOT, 'TRAIN', 'workout_vocab.pkl'), "rb") as f:
            train_workout_vocab = pickle.load(f)

        with open(os.path.join(VOCAB_TREE_ROOT, 'TRAIN', 'condition_vocab.pkl'), "rb") as f:
            train_condition_vocab = pickle.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "VALID vocab을 생성하기 전에 DATA_TYPE='TRAIN'으로 TRAIN vocab을 먼저 생성해야 합니다."
        ) from exc
#
#
#

json_lst = [path for base_path in BASE_PATH[DATA_TYPE] for path in glob(base_path + '/*/*/*.json') if not path.endswith('3d.json')]

for idx in tqdm(range(len(json_lst)), leave=True, desc="Generating Vocab.."):
    json_file = json_lst[idx]
    with open(json_file, 'r') as f:
        data = json.load(f)

        # 관절 보캡에 아무것도 들어있지 않으면
        if len(joint_vocab.keys()) == 0:
            for i, joint_name in enumerate(JOINTS_ORDER_HEAD_ver):
                joint_vocab[joint_name] = i + len(vocab)

        # HARD Exercise 제외를 위한 운동 이름 추출
        exercise_name = data['type_info']['exercise']
        if not IS_CONTAIN_HARD_EXERCISE:
            if exercise_name in HARD_EXERCISE:
                continue

        # # 런지 통합
        # if exercise_name in ['스텝 포워드 다이나믹 런지', '스텝 백워드 다이나믹 런지']:
        #     exercise_name = '다이나믹 런지'

        if IS_INTEGRATE_PUSHUP:
            if exercise_name in ['푸시업', '니푸쉬업']:
                exercise_name = '푸시업'

        # 바벨-덤벨 로우 통합
        if IS_INTEGRATE_ROW:
            if exercise_name in ['덤벨 벤트오버 로우', '바벨 로우']:
                # condition 처리
                for i in range(len(data['type_info']['conditions'])):
                    condition = data['type_info']['conditions'][i]['condition']
                    if condition in ['바벨 궤적과 몸 밀착', '덤벨 궤적과 몸 밀착']:
                        data['type_info']['conditions'][i]['condition'] = '바벨-덤벨 궤적과 몸 밀착'

                # 이름 변경
                exercise_name = '바벨-덤벨 로우'

        # '바벨 컬 ' << 의 불필요한 공백 제거
        if exercise_name == '바벨 컬 ':
            exercise_name = '바벨 컬'

        # 현재 운동 이름이 vocab에 포함되어있지 않은 경우 vocab에 등록
        if exercise_name not in workout_vocab.keys():
            if DATA_TYPE == 'VALID':
                if exercise_name not in train_workout_vocab:
                    raise KeyError(
                        f"VALID 운동 '{exercise_name}'이 TRAIN workout vocab에 없습니다."
                    )
                workout_vocab[exercise_name] = train_workout_vocab[exercise_name]
            else:
                workout_vocab[exercise_name] = len(vocab) + len(joint_vocab) + len(workout_vocab)

        # VALID condition도 부분집합만 저장하되 TRAIN의 동일한 integer ID를 사용한다.
        for condition_info in data['type_info']['conditions']:
            condition = condition_info['condition']

            if condition not in condition_vocab.keys():
                if DATA_TYPE == 'VALID':
                    if condition not in train_condition_vocab:
                        raise KeyError(
                            f"VALID condition '{condition}'이 TRAIN condition vocab에 없습니다."
                        )
                    condition_vocab[condition] = train_condition_vocab[condition]
                else:
                    condition_vocab[condition] = (
                        len(vocab)
                        + len(joint_vocab)
                        + len(condition_vocab)
                        + NUMBER_of_WORKOUT[DATA_TYPE]
                    )

# ID 매핑은 변경하지 않고, (key, value) 쌍을 value 기준 오름차순으로 정렬한다.
vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
joint_vocab = dict(sorted(joint_vocab.items(), key=lambda item: item[1]))
workout_vocab = dict(sorted(workout_vocab.items(), key=lambda item: item[1]))
condition_vocab = dict(sorted(condition_vocab.items(), key=lambda item: item[1]))

assert len(workout_vocab) == NUMBER_of_WORKOUT[DATA_TYPE]

os.makedirs(OUTPUT_VOCAB_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_VOCAB_DIR, 'vocab.pkl'), "wb") as f:
    pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(OUTPUT_VOCAB_DIR, 'joint_vocab.pkl'), "wb") as f:
    pickle.dump(joint_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(OUTPUT_VOCAB_DIR, 'workout_vocab.pkl'), "wb") as f:
    pickle.dump(workout_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(OUTPUT_VOCAB_DIR, 'condition_vocab.pkl'), "wb") as f:
    pickle.dump(condition_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
