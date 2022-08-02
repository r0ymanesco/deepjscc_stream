import os
import os.path
import cv2
import glob
import ipdb
import random
import numpy as np

import torch.utils.data as data

from utils import np_to_torch


class UCF101(data.Dataset):
    def __init__(self, path, mode, frames_per_clip):
        self.mode = mode
        self.frames_per_clip = frames_per_clip

        self._get_video_fns(path)
        self._get_video_list(mode)

    def _get_video_fns(self, path):
        bad_list = [
            path + '/v_PommelHorse_g05_c01.avi',
            path + '/v_PommelHorse_g05_c02.avi',
            path + '/v_PommelHorse_g05_c03.avi',
            path + '/v_PommelHorse_g05_c04.avi',
            path + '/v_Knitting_g04_c04_out.avi',
        ]

        self.video_fns = []
        for fn in glob.iglob(path + '/*avi'):
            if fn not in bad_list:
                self.video_fns.append(fn)

        self.num_videos = len(self.video_fns)

    def _get_video_list(self, mode):
        random.Random(4).shuffle(self.video_fns)
        train_size = int(self.num_videos // 1.25)
        eval_size = int(self.num_videos // 10)

        match mode:
            case 'train':
                self.video_fns = self.video_fns[:train_size]
            case 'val':
                self.video_fns = self.video_fns[train_size:train_size+eval_size]
            case 'eval':
                self.video_fns = self.video_fns[train_size+eval_size:train_size+2*eval_size]

        print('Number of {} videos loaded: {}'.format(mode, len(self.video_fns)))

    def __getitem__(self, index):
        vid_fn = self.video_fns[index]
        vid = cv2.VideoCapture(vid_fn)

        frames = []
        for _ in range(self.frames_per_clip):
            flag, frame = vid.read()
            assert flag is True
            frames.append(frame)

        frames = np.concatenate(frames, axis=2)
        frames = np_to_torch(frames)
        frames = frames / 255.0
        return frames, vid_fn

    def __len__(self):
        return len(self.video_fns)

    @staticmethod
    def get_config():
        frame_sizes = (3, 240, 320)
        reduced_arch = False
        return frame_sizes, reduced_arch

    def __str__(self):
        return f'UCF101({self.frames_per_clip})'
