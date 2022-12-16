import cv2
import csv
import random

import torch.utils.data as data

from utils import np_to_torch, crop_cv2


class ImageNet(data.Dataset):
    def __init__(self, path, csv_file, mode, crop=256):
        assert crop <= 256  # NOTE the compiled csv file only contains images bigger than 256
        assert crop != 0  # NOTE make sure crop value is set (default 0)

        self.path = path
        self.mode = mode
        self.crop = crop
        self._get_image_list(csv_file)

    def _get_fn_from_csv(self, file):
        fns = []
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img_file = row[0].split('/')[1]
                img_file = self.path + '/' + img_file
                fns.append(img_file)
        return fns

    def _get_image_list(self, csv_file):
        fns = self._get_fn_from_csv(csv_file)
        random.Random(4).shuffle(fns)
        num_images = len(fns)
        train_size = int(num_images // 1.25)
        eval_size = int(num_images // 10)

        match self.mode:
            case 'train':
                self.fns = fns[:train_size]
            case 'val':
                self.fns = fns[train_size:train_size+eval_size]
            case 'eval':
                self.fns = fns[train_size+eval_size:train_size+2*eval_size]
            case _:
                raise ValueError

        print('Number of {} images loaded: {}'.format(self.mode, len(self.fns)))

    def __getitem__(self, index):
        image_fn = self.fns[index]
        image = cv2.imread(image_fn)

        image = crop_cv2(image, self.crop)
        image = np_to_torch(image)
        image = image / 255.0
        return image, image_fn

    def __len__(self):
        return len(self.fns)

    @staticmethod
    def get_config(crop):
        img_sizes = (3, crop, crop)
        reduced_arch = False
        return img_sizes, reduced_arch

    def __str__(self):
        return f'ImageNet({self.crop})'
