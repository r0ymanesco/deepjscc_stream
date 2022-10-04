import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as LS

from pytorch_msssim import ms_ssim

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

    match loss:
        case 'l2':
            loss = F.mse_loss(prediction, target, reduction='none')
            match reduction:
                case 'mean':
                    loss = loss.mean()
                case 'batch':
                    loss = torch.mean(loss, dim=(4, 3, 2))
                case _:
                    raise NotImplementedError
        case 'msssim':
            shape = prediction.size()
            assert len(shape) <= 5
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
