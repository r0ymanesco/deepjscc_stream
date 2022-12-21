import cv2
import glob
import numpy as np

import torch.utils.data as data

from utils import np_to_torch


class TinyImageNet(data.Dataset):
    def __init__(self, path, mode, n_classes=200):
        self.path = path
        self.mode = mode

        self.get_image_list(mode, n_classes)

    def get_image_list(self, mode, n_classes):
        self.wnids = []
        label_file = self.path + '/wnids.txt'
        wnids = open(label_file, 'r').readlines()
        for line in wnids[:n_classes]:
            self.wnids.append(line.strip())

        self.fns = []
        self.labels = []
        for fn in glob.iglob(self.path + '/**/*.JPEG', recursive=True):
            if mode is not None:
                file_wnid = fn.split('/')[-1]
                file_wnid = file_wnid[:-5].split('_')[0]
                if file_wnid in self.wnids:
                    self.fns.append(fn)
                    self.labels.append(self.wnids.index(file_wnid))
            else:
                self.fns.append(fn)
                self.labels.append(0)

        n_images = len(self.fns)

        match mode:
            case 'train':
                self.fns = self.fns[:int(np.floor(n_images*0.8))]
                self.labels = self.labels[:int(np.floor(n_images*0.8))]
            case 'val':
                self.fns = self.fns[int(np.floor(n_images*0.8)):int(np.floor(n_images*0.9))]
                self.labels = self.labels[int(np.floor(n_images*0.8)):int(np.floor(n_images*0.9))]
            case 'eval':
                self.fns = self.fns[int(np.floor(n_images*0.9)):]
                self.labels = self.labels[int(np.floor(n_images*0.9)):]
        print('Number of images loaded: {}'.format(len(self.fns)))

    def __getitem__(self, index):
        image_fn = self.fns[index]
        label = self.labels[index]
        image = cv2.imread(image_fn)

        image = np_to_torch(image)
        image = image / 255.0
        return image, int(label)

    def __len__(self):
        return len(self.fns)

    @staticmethod
    def get_config():
        img_sizes = (3, 64, 64)
        reduced_arch = False
        return img_sizes, reduced_arch

    def __str__(self):
        return 'TinyImageNet'
