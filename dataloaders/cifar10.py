import os
import numpy as np

import torch.utils.data as data

from utils import np_to_torch, unpickle


class CIFAR10(data.Dataset):
    def __init__(self, path, mode):
        train_data = np.empty((50000, 32, 32, 3), dtype=np.uint8)
        train_labels = np.empty(50000, dtype=np.uint8)
        for i in range(0, 5):
            data_train = unpickle(os.path.join(path, 'data_batch_{}'.format(i+1)))
            train_data[i*10000:(i+1)*10000] = data_train[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
            train_labels[i * 10000:(i + 1) * 10000] = data_train[b'labels']
        self.train = train_data, train_labels
        data_test = unpickle(os.path.join(path, 'test_batch'))
        test_set = data_test[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1)), data_test[b'labels']
        self.test = (test_set[0][:5000], test_set[1][:5000])
        self.valid = (test_set[0][5000:], test_set[1][5000:])

        match mode:
            case 'train':
                self.dataset = self.train
            case 'val':
                self.dataset = self.valid
            case 'eval':
                self.dataset = self.test
            case _:
                raise ValueError

    def __getitem__(self, index):
        img, label = self.dataset[0][index], self.dataset[1][index]
        img = np_to_torch(img) / 255.
        return img, int(label)

    def __len__(self):
        return len(self.dataset[0])

    @staticmethod
    def get_config():
        img_sizes = (3, 32, 32)
        reduced_arch = True
        return img_sizes, reduced_arch

    def __str__(self):
        return 'CIFAR10'
