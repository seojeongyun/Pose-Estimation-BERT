import os
import json

from glob import glob
import pickle
from tqdm import tqdm
from collections import defaultdict
from collections import Counter


if __name__ == '__main__':
    # *-*-*-*-*-*- Collect validation json files *-*-*-*-*-*-
    base_path = '/storage/jysuh/fitness/fitness/validation/label'
    assert os.path.exists(base_path)

    json_list_valid = []  # 3139
    json_list_train = []  # 34468
    # total number of json files is 37670

    # [Validation] max_frame = 17
    # [Train] max_frame = 21

    for equipment_type_idx, _ in enumerate(os.listdir(base_path)):
        path = base_path + '/' + os.listdir(base_path)[equipment_type_idx]
        for idx in range(len(os.listdir(path))):
            path = path + '/' + os.listdir(path)[idx]
            for idx in range(len(os.listdir(path))):
                path = path + '/' + os.listdir(path)[idx]
        for _, json_files in enumerate(os.listdir(path)):
            if '3d' not in json_files:
                json_list_valid.append(path + '/' + json_files)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    #

    # *-*-*-*-*-*- Collect train json files *-*-*-*-*-*-
    base_path = '/storage/jysuh/fitness/fitness/train/label'

    for equipment_type_idx, _ in enumerate(os.listdir(base_path)):
        if 'json' not in os.listdir(base_path)[equipment_type_idx] and 'new' not in os.listdir(base_path)[
            equipment_type_idx]:
            # equipment_type_list.append(os.listdir(base_path)[equipment_type_idx]) # DB
            equipment_type_path = '/'.join([base_path, os.listdir(base_path)[equipment_type_idx]])
            for equipment_idx in range(len(os.listdir(equipment_type_path))):
                equipment_path = '/'.join([equipment_type_path, os.listdir(equipment_type_path)[equipment_idx]])
                for idx in range(len(os.listdir(equipment_path))):
                    day_path = '/'.join([equipment_path, os.listdir(equipment_path)[idx]])

                for _, json_files in enumerate(os.listdir(day_path)):
                    if '3d' not in json_files:
                        # if os.listdir(base_path)[equipment_type_idx] == 'barbell_dumbbell_Labeling':    # DB
                        #     bd_list.append(day_path + '/' + json_files) # DB
                        # elif os.listdir(base_path)[equipment_type_idx] == 'furniture_Labeling': # DB
                        #     furniture_list.append(day_path + '/' + json_files) # DB
                        # elif os.listdir(base_path)[equipment_type_idx] == 'body_Labeling': # DB
                        #     counter[day_path.split('/')[8]] += 1
                        #     body_list.append(day_path + '/' + json_files)   # DB
                        json_list_train.append(day_path + '/' + json_files)
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    # *-*-*-*-*-*- extracting information from json files *-*-*-*-*-*-
    json_files_list = json_list_train + json_list_valid

    # parameter init
    video_counter = defaultdict(int)
    coord_dict = {}
    thr_num_files = 0
    cnt = 0
    save_condition = False
    if save_condition:
        with open('/home/jysuh/PycharmProjects/coord_embedding/dataset/embedder_dataset/condition_vocab.pkl',
                  'rb') as f:
            condition_vocab = pickle.load(f)

    # ===
    head_key = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear']
    for json_file in tqdm(json_list_valid, desc='...', leave=True):
        with open(json_file , 'r') as f:
            data = json.load(f)
            exercise_name = data['type_info']['exercise']
            # To make validation set, about a video per a wrkout
            # if exercise_name in coord_dict.keys():
            #     if len(coord_dict[exercise_name].keys()) == 1:
            #         continue
            # else:
            #     coord_dict.setdefault(exercise_name, {})

            coord_dict.setdefault(exercise_name, {})

            #
            video_idx = video_counter[exercise_name]
            video_counter[exercise_name] += 1

        for frame_idx in range(len(data['frames'])):
            if len(data['frames'][frame_idx].keys()) != 5:
                print()
            for view_idx in data['frames'][frame_idx].keys():
                coord_dict[exercise_name].setdefault(view_idx, {})
                coord_dict[exercise_name][view_idx].setdefault(video_idx, {})
                # 260226 add exercise condition
                if save_condition:
                    coord_dict[exercise_name][view_idx][video_idx]['conditions'] = []
                    for i in range(len(data['type_info']['conditions'])):
                        condition = condition_vocab[data['type_info']['conditions'][i]['condition']]
                        value = int(data['type_info']['conditions'][i]['value'])
                        coord_dict[exercise_name][view_idx][video_idx]['conditions'].append([condition, value])
                # ----------------------------------------------------------
                coord_dict[exercise_name][view_idx][video_idx].setdefault(frame_idx, {})
                coord_dict_for_a_frame = {}
                head_x, head_y = 0, 0
                for joint_idx, joint_name in enumerate(data['frames'][frame_idx][view_idx]['pts'].keys()):
                    if joint_name in head_key:
                        head_x += data['frames'][frame_idx][view_idx]['pts'][joint_name]['x']
                        head_y += data['frames'][frame_idx][view_idx]['pts'][joint_name]['y']
                    else:
                        coord_dict_for_a_frame[joint_name]= [data['frames'][frame_idx][view_idx]['pts'][joint_name]['x'], data['frames'][frame_idx][view_idx]['pts'][joint_name]['y']]
                head_x, head_y = head_x / 5, head_y / 5
                coord_dict_for_a_frame['Head'] = [int(head_x), int(head_y)]
                coord_dict[exercise_name][view_idx][video_idx][frame_idx] = coord_dict_for_a_frame

    with open('/home/jysuh/PycharmProjects/coord_embedding/dataset/embedder_dataset/valid_contained_condition.json', 'w') as f:
        json.dump(coord_dict, f)


cnt = 0
for exer in coord_dict.keys():
    for vid in coord_dict[exer].keys():
        for frame in coord_dict[exer][vid].keys():
            for view in coord_dict[exer][vid][frame].keys():
                cnt += 1

cnt = 0
for exercise_name in coord_dict:
    for view_idx in coord_dict[exercise_name]:
        for video_idx in coord_dict[exercise_name][view_idx]:
            cnt += 1

# orig = 188035
# now = 187895
# diff = 140

# only train json file -> 34468
# only valid json file ->