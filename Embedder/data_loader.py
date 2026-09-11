from pprint import pprint
import pickle
import tqdm
import torch
import numpy as np

from torch.utils.data import Dataset
from Embedder.Embedder_config import config
from tqdm import tqdm

class Video_Loader(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.data_path = self.get_data_path()
        self.videos = self.get_data()

        # self.vocab = self.get_vocab()
        pprint('NUMBER OF VIDEOS:' + str(len(self.videos)))

    def get_data_path(self):
        if self.mode == 'train' or self.mode == 'train-valid':
            return config.TRAIN_DATA_PATH
        else:
            return config.VALID_DATA_PATH

    def get_data(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        #
        pprint('VIDEO FILE SUCCESSFULLY LOADED USING PICKLE')
        return data

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video, workout_class, conditions = self.videos[idx]
        return video, workout_class, conditions

    # def collate_fn(self, batch):
    #     videos, exercise_name, conditions = zip(*batch)
    #     return videos, exercise_name, conditions

    def collate_fn(self, batch):
        videos, exercise_name, conditions = zip(*batch)

        video_array = np.stack([
            np.stack([
                np.stack([
                    video[str(frame_idx)][joint_name]
                    for joint_name in self.config.JOINTS_NAME
                ], axis=0)
                for frame_idx in range(self.config.MAX_FRAMES)
            ], axis=0)
            for video in videos
        ], axis=0).astype(np.float32)

        videos = torch.from_numpy(video_array)

        return videos, exercise_name, conditions
if __name__ == '__main__':
    loader = Video_Loader(config)
    print('done')