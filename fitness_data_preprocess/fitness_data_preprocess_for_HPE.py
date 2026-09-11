'''
    AI Hub에서 제공하는 fitness 데이터셋을 전처리 하기 위한 스크립트 파일
    HPE의 입력을 위한 구조로 전처리하는 코드
    HPE 입력 구조는 아래와 같다.

    OUTPUT = {
        "/.../000001.jpg":
            array([[988,369],
                   [981,358],
                   ...
                   [962,865]], dtype=uint16),

        "/.../000002.jpg":
            array(...)
    }
}
'''

import os
import pickle

from glob import glob
from tqdm import tqdm
from collections import defaultdict

counter = defaultdict(int)

import json, numpy as np

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

VIEWS = ['view1', 'view2', 'view3', 'view4', 'view5']

HARD_EXERCISE = ['바이시클 크런치', '크런치', '시저크로스', '라잉 레그 레이즈', '플랭크', '힙쓰러스트', '푸시업', '니푸쉬업']

DATA_TYPE = 'TRAIN' # 'TRAIN' or 'VALID'

OUTPUT = {}
#
#
#

json_lst = [path for base_path in BASE_PATH[DATA_TYPE] for path in glob(base_path + '/*/*/*.json') if not path.endswith('3d.json')]

for idx in tqdm(range(len(json_lst)), leave=True, desc="extracting img path and pts from json file"):
    json_file = json_lst[idx]
    with open(json_file, 'r') as f:
        data = json.load(f)

        # HARD Exercise 제외
        exercise_name = data['type_info']['exercise']
        if exercise_name in HARD_EXERCISE:
            continue

        # img_key를 활용해 이미지 패스에 접근하기 위한 전처리
        label2image = json_file.replace('label', 'image').split('/')
        img_base_path = os.path.join('/'.join(label2image[0:7]), '/'.join(label2image[8:-2]))

        for view_idx in VIEWS:
            #
            view_buffer = {}
            for frame_idx in range(len(data['frames'])):
                # img_path와 img_key를 결합
                img_path = os.path.join(img_base_path, data['frames'][frame_idx][view_idx]['img_key'])

                # 해당 이미지가 없다면 continue하여 다음 ima_path 검사
                if not os.path.exists(img_path):
                    continue

                # pts 추출
                pts = data['frames'][frame_idx][view_idx]['pts']

                # JOINTS_ORDER와 pts를 대조해 누락된 관절이 있는지 검사
                missing = [joint for joint in JOINTS_ORDER if joint not in pts]
                if missing:
                    raise ValueError(f"Missing joints: {missing} | "
                                     f"json_file={json_file} | frame_idx={frame_idx} | view_idx={view_idx}")

                # JOINTS_ORDER와 pts를 대조해 또 다른 관절이 있는지 검사
                extra = [joint for joint in pts if joint not in JOINTS_ORDER]
                if extra:
                    raise ValueError(f"Unexpected joints: {extra}")

                # 관절 좌표를 JOINTS_ORDER 순서대로 OUTPUT[img_path]에 저장
                joints = []
                IN_RANGE = True
                for joint_name in JOINTS_ORDER:
                    x, y = data['frames'][frame_idx][view_idx]['pts'][joint_name]['x'], data['frames'][frame_idx][view_idx]['pts'][joint_name]['y']

                    # 만약 1920 x 1080의 해상도를 벗어나는 관절 좌표가 있다면
                    if x < 0 or y < 0 or x >= 1920 or y >= 1080:   # 음수도 함께 검사
                        # 범위 안을 의미하는 IN_RANGE 변수를 False로 변경하고 JOINT_ORDER를 순회하는 for문을 break
                        IN_RANGE = False
                        break

                    joint_pts = [x, y]
                    joints.append(joint_pts)

                # IN_RANGE가 False라는 것은 1920 x 1080의 해상도를 벗어나는 데이터가 있음을 의미하므로
                # break 함으로써 현재의 view video는 데이터에서 제외
                if not IN_RANGE:
                    break

                # joints의 shape이 24,2가 아닌 경우 에러 발생
                joints = np.asarray(joints, dtype=np.uint16)
                if joints.shape != (24, 2):
                    raise ValueError(
                        f"Invalid joint shape: {joints.shape} | "
                        f"json_file={json_file} | "
                        f"frame_idx={frame_idx} | "
                        f"view_idx={view_idx}"
                    )
                # 문제 없는 경우 joints를 buff에 삽입
                view_buffer[img_path] = joints

            else:
                # break 없이 이 view의 모든 프레임을 통과한 경우에만 한꺼번에 업데이트
                OUTPUT.update(view_buffer)

with open(f"/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/fitness_data_preprocess/HPE Dataset/{DATA_TYPE}_HPE.pkl", "wb") as f:
    pickle.dump(OUTPUT, f, protocol=pickle.HIGHEST_PROTOCOL)






