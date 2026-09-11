from pprint import pprint
import json
import tqdm

from torch.utils.data import Dataset
from Embedder.Embedder_config import config
from tqdm import tqdm

class Video_Loader(Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.data = self.get_data(data_path=data_path)
        self.vocab = self.get_vocab()
        self.videos = self.preprocess()
        pprint('NUMBER OF VIDEOS:' + str(len(self.videos)))

    def get_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        #
        pprint('VIDEO FILE SUCCESSFULLY LOADED')
        return data

    def get_vocab(self):
        vocab = {'PAD': 0, 'SEP' : 1}
        #
        exercise_name = list(self.data.keys())[0]
        video_idx = list(self.data[exercise_name])[0]
        frame_idx = list(self.data[exercise_name][video_idx])[0]
        view_idx = list(self.data[exercise_name][video_idx][frame_idx])[0]

        for joint_name in self.data[exercise_name][video_idx][frame_idx][view_idx]:
            if joint_name not in vocab:
                vocab[joint_name] = len(vocab)

        for exercise_name in self.data.keys():
            if exercise_name not in vocab:
                vocab[exercise_name] = len(vocab)

        pprint('VOCAB SUCCESSFULLY BUILT')
        return vocab

    def preprocess(self):
        videos = []
        self.cnt = 0
        for exercise_name in tqdm(self.data.keys(), total=len(list(self.data.keys())), desc='preprocess'):
            for view_idx  in self.data[exercise_name].keys():
                for video_idx in self.data[exercise_name][view_idx].keys():
                    self.cnt += 1
                    video = self.data[exercise_name][view_idx][video_idx]
                    #
                    GO_TO_NEXT_VIDEO = False
                    for frame_idx in video.keys():
                        if list(video[frame_idx].keys()) != self.config.JOINTS_NAME:
                            GO_TO_NEXT_VIDEO = True
                            break

                    if GO_TO_NEXT_VIDEO:
                        continue
                    #
                    keys = list(video.keys())  # frames key
                    is_sorted = keys == sorted(keys, key=lambda x: int(x))
                    if not is_sorted:
                        raise ValueError(
                            f"[Error] Frame keys in video '{video_idx}' of view '{view_idx}' "
                            f"for exercise '{exercise_name}' are not in ascending order."
                        )
                    # workout_name, view_idx
                    videos.append([video, exercise_name, view_idx])

        pprint("VIDEOS SUCCESSFULLY SEPARATED - FRAME KEYS ARE SORTED IN ASCENDING ORDER.")
        pprint('ORIGIANL NUMBER OF VIDEOS:' + str(self.cnt))
        return videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx][0]
        exercise_name = self.vocab[self.videos[idx][1]]
        view_idx = self.videos[idx][2]
        return video, exercise_name, view_idx

    def collate_fn(self, batch):
        videos, exercise_name, view_idx = zip(*batch)
        return videos, exercise_name, view_idx

if __name__ == '__main__':
    loader = Video_Loader(config, config.DATA_PATH)
    print('done')