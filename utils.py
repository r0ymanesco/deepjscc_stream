import ipdb
import pickle
import random
import numpy as np
from collections import Counter
from itertools import combinations

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as LS

from pytorch_msssim import ms_ssim, ssim

from modules.lookahead import Lookahead


def get_dataloader(dataset, params):
    dataset_aux = {}

    match dataset:
        case 'ucf101':
            from dataloaders.ucf101 import UCF101
            frame_sizes, reduced_arch = UCF101.get_config()
            dataset_aux['frame_sizes'] = frame_sizes
            dataset_aux['reduced_arch'] = reduced_arch
            train_loader = UCF101(params.path, 'train', params.frames_per_clip)
            val_loader = UCF101(params.path, 'val', params.frames_per_clip)
            eval_loader = UCF101(params.path, 'eval', params.frames_per_clip)
        case 'imagenet':
            from dataloaders.imagenet import ImageNet
            img_sizes, reduced_arch = ImageNet.get_config(params.crop)
            dataset_aux['img_sizes'] = img_sizes
            dataset_aux['reduced_arch'] = reduced_arch
            train_loader = ImageNet(params.path, params.path + '/256_images.csv', 'train', params.crop)
            val_loader = ImageNet(params.path, params.path + '/256_images.csv', 'val', params.crop)
            eval_loader = ImageNet(params.path, params.path + '/256_images.csv', 'eval', params.crop)
        case 'cifar10':
            from dataloaders.cifar10 import CIFAR10
            img_sizes, reduced_arch = CIFAR10.get_config()
            dataset_aux['img_sizes'] = img_sizes
            dataset_aux['reduced_arch'] = reduced_arch
            train_loader = CIFAR10(params.path, 'train')
            val_loader = CIFAR10(params.path, 'val')
            eval_loader = CIFAR10(params.path, 'eval')
        case _:
            raise NotImplementedError
    return (train_loader, val_loader, eval_loader), dataset_aux


def get_optimizer(params, modules):
    optimizer_aux = {}
    module_params = [{'params': mod.parameters()} for mod in modules]

    match params.solver:
        case 'adam':
            solver = optim.Adam(module_params, lr=params.lr)
            optimizer_aux['str'] = f'Adam({params.lr})'
        case 'radam':
            solver = optim.RAdam(module_params, lr=params.lr)
            optimizer_aux['str'] = f'RAdam({params.lr})'
        case _:
            raise NotImplementedError

    if params.lookahead:
        solver = Lookahead(solver, alpha=params.lookahead_alpha, k=params.lookahead_k)
        optimizer_aux['str'] += f'_Lookahead({params.lookahead_alpha},{params.lookahead_k})'

    return solver, optimizer_aux


def get_scheduler(solver, params):
    scheduler_aux = {}

    match params.scheduler:
        case 'mult_lr':
            lr_scheduler = LS.MultiplicativeLR(solver, lr_lambda=lambda x: params.lr_schedule_factor)
            scheduler_aux['str'] = f'MultLR({params.lr_schedule_factor})'
        case _:
            raise NotImplementedError

    return lr_scheduler, scheduler_aux


def np_to_torch(img):
    img = np.swapaxes(img, 0, 1)  # w, h, c
    img = np.swapaxes(img, 0, 2)  # c, h, w
    return torch.from_numpy(img).float()


def to_chan_last(img):
    img = img.transpose(1, 2)
    img = img.transpose(2, 3)
    return img


def as_img_array(image):
    image = image.clamp(0, 1) * 255.0
    return torch.round(image)


def calc_psnr(predictions, targets):
    metric = []
    for (pred, targ) in zip(predictions, targets):
        original = as_img_array(targ)
        prediction = as_img_array(pred)
        mse = torch.mean((original - prediction) ** 2., dtype=torch.float32)
        psnr = 20 * torch.log10(255. / torch.sqrt(mse))
        metric.append(psnr.item())
    return metric


def calc_ssim(predictions, targets):
    metric = []
    for (pred, targ) in zip(predictions, targets):
        original = as_img_array(targ)
        prediction = as_img_array(pred)
        ssim_val = ssim(original, prediction,
                        data_range=255, size_average=True)
        metric.append(ssim_val.item())
    return metric


def calc_msssim(predictions, targets):
    metric = []
    for (pred, targ) in zip(predictions, targets):
        original = as_img_array(targ)
        prediction = as_img_array(pred)
        msssim = ms_ssim(original, prediction,
                         data_range=255, size_average=True)
        metric.append(msssim.item())
    return metric


def calc_loss(prediction, target, loss, reduction='mean'):
    loss_aux = {}
    shape = prediction.size()
    n_dims = len(shape)
    assert len(shape) <= 5

    match loss:
        case 'l2':
            loss = F.mse_loss(prediction, target, reduction='none')
            match reduction:
                case 'mean':
                    loss = loss.mean()
                case 'batch':
                    loss = torch.mean(loss, dim=[x for x in range(n_dims-1, n_dims-4, -1)])
                case _:
                    raise NotImplementedError
        case 'msssim':
            if len(shape) == 5:
                prediction = prediction.view(-1, shape[2], shape[3], shape[4])
                target = target.view(-1, shape[2], shape[3], shape[4])

            avg = False
            if reduction == 'mean': avg = True
            loss = 1 - ms_ssim(prediction, target,
                               data_range=1, size_average=avg)
            if not avg:
                loss = loss.view(-1, shape[1])
        case _:
            raise NotImplementedError
    return loss, loss_aux


def perms_without_reps(s):
    partitions = list(Counter(s).items())

    def _helper(idxset, i):
        if len(idxset) == 0:
            yield ()
            return
        for pos in combinations(idxset, partitions[i][1]):
            for res in _helper(idxset - set(pos), i+1):
                yield (pos,) + res

    n = len(s)
    for poses in _helper(set(range(n)), 0):
        out = [None] * n
        for i, pos in enumerate(poses):
            for idx in pos:
                out[idx] = partitions[i][0]
        yield out


def split_list_by_val(x, s):
    size = len(x)
    idx_list = [idx + 1 for idx, val in enumerate(x) if val == s]
    res = [x[i:j] for i, j in zip([0] + idx_list, idx_list + [size])]
    return res


def crop_cv2(img, patch):
    assert patch > 0
    height, width, _ = img.shape
    start_x = random.randint(0, height - patch)
    start_y = random.randint(0, width - patch)
    return img[start_x:start_x + patch, start_y:start_y + patch]


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
