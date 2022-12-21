import ipdb
import os
import numpy as np

from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.utils.data as data

from trainer.base_trainer import BaseTrainer

from modules.modem import Modem
from modules.channel import Channel
from modules.scheduler import EarlyStopping
from modules.feature_encoder_AF import FeatureEncoderAF, FeatureDecoderAF

from utils import calc_loss, calc_msssim, calc_ssim, calc_psnr
from utils import get_dataloader, get_optimizer, get_scheduler


class DeepJSCC_Q(BaseTrainer):
    def __init__(self, dataset, loss, params, resume=False):
        super().__init__('DeepJSCC_Q', dataset, loss, resume, params.device)

        self.epoch = 0
        self.params = params
        self.save_dir = params.save_dir

        self._get_config(params)

    def _get_config(self, params):
        self.job_name = f'{self.trainer}({self.loss})'

        (self.train_loader,
         self.val_loader,
         self.eval_loader), dataset_aux = self._get_data(params.dataset)

        self.frame_dims = dataset_aux['img_sizes']
        self.reduced_arch = dataset_aux['reduced_arch']
        if self.reduced_arch or self.loss == 'ssim':
            self.sim_metric = 'ssim'
        else:
            self.sim_metric = 'msssim'

        self.encoder, self.decoder = self._get_encoder(params.encoder, self.frame_dims, self.reduced_arch)

        self.modem = self._get_modem(params.modem)

        self.channel = self._get_channel(params.channel)

        modules = [self.encoder, self.decoder, self.modem, self.channel]
        self.optimizer, optimizer_aux = get_optimizer(params.optimizer, modules)
        self.job_name += '_' + optimizer_aux['str']

        self.lr_scheduler, scheduler_aux = get_scheduler(self.optimizer, params.scheduler)
        self.job_name += '_' + scheduler_aux['str']

        self.es = EarlyStopping(mode=params.early_stop.mode,
                                min_delta=params.early_stop.delta,
                                patience=params.early_stop.patience,
                                percentage=False)
        self.job_name += '_' + str(self.es)

        self.scheduler_fn = lambda epochs: epochs % (params.early_stop.patience//2) == 0

        if len(params.comments) != 0: self.job_name += f'_Ref({params.comments})'

        if self.resume: self.load_weights()

    def _get_data(self, params):
        (train_loader, val_loader, eval_loader), dataset_aux = get_dataloader(params.dataset, params)
        self.job_name += '_' + str(train_loader)
        train_loader = data.DataLoader(
            dataset=train_loader,
            batch_size=params.train_batch_size,
            shuffle=True,
            num_workers=2,
        )
        val_loader = data.DataLoader(
            dataset=val_loader,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=2,
        )
        eval_loader = data.DataLoader(
            dataset=eval_loader,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=2,
        )
        return (train_loader, val_loader, eval_loader), dataset_aux

    def _get_encoder(self, params, img_sizes, reduced):
        if reduced:
            self.feat_dims = (params.c_out, *[dim // 4 for dim in img_sizes[1:]])
        else:
            self.feat_dims = (params.c_out, *[dim // 16 for dim in img_sizes[1:]])

        encoder = FeatureEncoderAF(
            c_in=params.c_in,
            c_feat=params.c_feat,
            c_out=params.c_out,
            reduced=reduced,
        ).to(self.device)
        decoder = FeatureDecoderAF(
            c_in=params.c_out,
            c_feat=params.c_feat,
            c_out=params.c_in,
            feat_dims=self.feat_dims,
            reduced=reduced
        ).to(self.device)
        self.job_name += '_' + str(encoder)
        return encoder ,decoder

    def _get_modem(self, params):
        modem = Modem(params.modem,
                      mod_order=params.mod_order,
                      return_likelihoods=False,
                      commitment=params.commitment,
                      anneal='linear',
                      sigma_start=params.sigma_start,
                      sigma_max=params.sigma_max,
                      sigma_period=params.sigma_period,
                      sigma_scale=params.sigma_scale).to(self.device)
        self.job_name += '_' + str(modem)
        return modem

    def _get_channel(self, params):
        channel = Channel(params.model, params).to(self.device)
        self.job_name += '_' + str(channel)
        return channel

    def __call__(self, snr, *args, **kwargs):
        self.check_mode_set()

        terminate = False
        epoch_trackers = {
            'loss_hist': [],
            'psnr_hist': [],
            f'{self.sim_metric}_hist': [],
        }

        with tqdm(self.loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
            for batch_idx, (images, img_fns) in enumerate(tepoch):
                pbar_desc = f'epoch: {self.epoch}, {self.mode}'
                tepoch.set_description(pbar_desc)

                epoch_postfix = OrderedDict()
                batch_trackers = {
                    'batch_loss': [],
                    'batch_psnr': [],
                    f'batch_{self.sim_metric}': [],
                }

                batch_snr = (snr[1] - snr[0]) * torch.rand((1, 1), device=self.device) + snr[0]
                epoch_postfix['snr'] = '{:.2f}'.format(batch_snr.item())

                images = images.to(self.device)
                code = self.encoder(images, batch_snr)
                symbols, modem_aux = self.modem.modulate(code)
                epoch_postfix['sigma'] = '{:.2f}'.format(modem_aux['sigma'])

                rx_symbols, _ = self.channel(symbols, [batch_snr.item()])
                demod_symbols = self.modem.demodulate(rx_symbols)
                prediction = torch.sigmoid(self.decoder(demod_symbols, batch_snr))


                loss, batch_trackers = self._get_loss(prediction, images, batch_trackers)
                if self._training: self._update_params(loss)

                epoch_trackers, epoch_postfix = self._update_epoch_postfix(batch_trackers,
                                                                           epoch_trackers,
                                                                           epoch_postfix)
                tepoch.set_postfix(**epoch_postfix)

        loss_mean, return_aux = self._get_return_aux(epoch_trackers)
        if self._validate: terminate = self._update_es(loss_mean)

        self.reset()
        return loss_mean, terminate, return_aux

    def _get_return_aux(self, epoch_trackers):
        return_aux = {}
        loss_mean = np.nanmean(epoch_trackers['loss_hist'])

        if not self._training:
            psnr_mean = np.nanmean(epoch_trackers['psnr_hist'])
            sim_mean = np.nanmean(epoch_trackers[f'{self.sim_metric}_hist'])

            if self._validate:
                return_aux['psnr_mean'] = psnr_mean
                return_aux[f'{self.sim_metric}_mean'] = sim_mean

            elif self._evaluate:
                psnr_std = np.sqrt(np.var(epoch_trackers['psnr_hist']))
                sim_std = np.sqrt(np.var(epoch_trackers[f'{self.sim_metric}_hist']))

                return_aux['psnr_mean'] = psnr_mean
                return_aux['psnr_std'] = psnr_std
                return_aux[f'{self.sim_metric}_mean'] = sim_mean
                return_aux[f'{self.sim_metric}_std'] = sim_std
        return loss_mean, return_aux

    def _update_epoch_postfix(self, batch_trackers, epoch_trackers, epoch_postfix):
        epoch_trackers['loss_hist'].append(np.nanmean(batch_trackers['batch_loss']))
        epoch_postfix[f'{self.loss} loss'] = '{:.5f}'.format(epoch_trackers['loss_hist'][-1])

        if not self._training:
            epoch_trackers['psnr_hist'].extend(batch_trackers['batch_psnr'])
            batch_psnr_mean = np.nanmean(batch_trackers['batch_psnr'])
            epoch_postfix['psnr'] = '{:.5f}'.format(batch_psnr_mean)

            epoch_trackers[f'{self.sim_metric}_hist'].extend(batch_trackers[f'batch_{self.sim_metric}'])
            batch_sim_mean = np.nanmean(batch_trackers[f'batch_{self.sim_metric}'])
            epoch_postfix[f'{self.sim_metric}'] = '{:.5f}'.format(batch_sim_mean)
        return epoch_trackers, epoch_postfix

    def _get_loss(self, predicted, target, batch_trackers):
        loss, _ = calc_loss(predicted, target, self.loss)
        batch_trackers['batch_loss'].append(loss.item())

        if not self._training:
            img_psnr = calc_psnr([predicted], [target])
            batch_trackers['batch_psnr'].extend(img_psnr)

            match self.sim_metric:
                case 'ssim':
                    img_sim = calc_ssim([predicted], [target])
                case 'msssim':
                    img_sim = calc_msssim([predicted], [target])
                case _:
                    raise ValueError
            batch_trackers[f'batch_{self.sim_metric}'].extend(img_sim)
        return loss, batch_trackers

    def _update_es(self, loss):
        flag, best_loss, best_epoch, bad_epochs = self.es.step(torch.Tensor([loss]), self.epoch)
        if flag:
            print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
        else:
            if bad_epochs == 0:
                self.save_weights()
                print('Saving best weights')
            elif self.scheduler_fn(bad_epochs):
                self.lr_scheduler.step()
                print('lr updated: {:.7f}'.format(self.lr_scheduler.get_last_lr()[0]))
            print('ES status: best: {:.6f}; bad epochs: {}/{}; best epoch: {}'
                  .format(best_loss.item(), bad_epochs, self.es.patience, best_epoch))
        return flag

    def _set_mode(self):
        match self.mode:
            case 'train':
                self.epoch += 1
                torch.set_grad_enabled(True)
                self.encoder.train()
                self.encoder.requires_grad_(True)

                self.decoder.train()
                self.decoder.requires_grad_(True)

                self.modem.train()
                self.modem.requires_grad_(True)

                self.channel.train()
                self.channel.requires_grad_(True)

                self.loader = self.train_loader
            case 'val':
                torch.set_grad_enabled(False)
                self.encoder.eval()
                self.encoder.requires_grad_(False)

                self.decoder.eval()
                self.decoder.requires_grad_(False)

                self.modem.eval()
                self.modem.requires_grad_(False)

                self.channel.eval()
                self.channel.requires_grad_(False)

                self.loader = self.val_loader
            case 'eval':
                torch.set_grad_enabled(False)
                self.encoder.eval()
                self.encoder.requires_grad_(False)

                self.decoder.eval()
                self.decoder.requires_grad_(False)

                self.modem.eval()
                self.modem.requires_grad_(False)

                self.channel.eval()
                self.channel.requires_grad_(False)

                self.loader = self.eval_loader

    def save_weights(self):
        if not os.path.exists(self.save_dir):
            print('Creating model directory: {}'.format(self.save_dir))
            os.makedirs(self.save_dir)

        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'modem': self.modem.state_dict(),
            'channel': self.channel.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'es': self.es.state_dict(),
            'epoch': self.epoch
        }, '{}/{}.pth'.format(self.save_dir, self.job_name))

    def load_weights(self):
        cp = torch.load('{}/{}.pth'.format(self.save_dir, self.job_name), map_location='cpu')

        self.encoder.load_state_dict(cp['encoder'])
        self.decoder.load_state_dict(cp['decoder'])
        self.modem.load_state_dict(cp['modem'])
        self.channel.load_state_dict(cp['channel'])

        self.optimizer.load_state_dict(cp['optimizer'])
        self.lr_scheduler.load_state_dict(cp['scheduler'])

        self.es.load_state_dict(cp['es'])
        self.epoch = cp['epoch']
        print('Loaded weights from epoch {}'.format(self.epoch))

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--save_dir', type=str, help='directory to save checkpoints')

        parser.add_argument('--dataset.dataset', type=str, help='dataset: dataset to use')
        parser.add_argument('--dataset.path', type=str, help='dataset: path to dataset')
        parser.add_argument('--dataset.crop', type=int, default=0, help='dataset: crop image size (not for cifar10)')
        parser.add_argument('--dataset.train_batch_size', type=int, help='dataset: training batch size')
        parser.add_argument('--dataset.eval_batch_size', type=int, help='dataset: evaluate batch size')

        parser.add_argument('--optimizer.solver', type=str, help='optimizer: optimizer to use')
        parser.add_argument('--optimizer.lr', type=float, help='optimizer: optimizer learning rate')

        parser.add_argument('--optimizer.lookahead', action='store_true', help='optimizer: to use lookahead')
        parser.add_argument('--optimizer.lookahead_alpha', type=float, default=0.0, help='optimizer: lookahead alpha')
        parser.add_argument('--optimizer.lookahead_k', type=int, default=1, help='optimizer: lookahead steps (k)')

        parser.add_argument('--scheduler.scheduler', type=str, help='scheduler: scheduler to use')
        parser.add_argument('--scheduler.lr_schedule_factor', type=float, help='scheduler: multi_lr: reduction factor')

        parser.add_argument('--encoder.c_in', type=int, help='encoder: number of input channels')
        parser.add_argument('--encoder.c_feat', type=int, help='encoder: number of feature channels')
        parser.add_argument('--encoder.c_out', type=int, help='encoder: number of output channels')

        parser.add_argument('--modem.modem', type=str, help='modem: modem to use')
        parser.add_argument('--modem.mod_order', type=int, help='modem: constellation order')
        parser.add_argument('--modem.commitment', type=float, help='modem: quantisation loss commitment')
        parser.add_argument('--modem.sigma_start', type=float, help='modem: quantisation hardness init')
        parser.add_argument('--modem.sigma_max', type=float, help='modem: quantisation hardness end')
        parser.add_argument('--modem.sigma_period', type=int, help='modem: quantisation hardness increment speed')
        parser.add_argument('--modem.sigma_scale', type=float, help='modem: quantisation hardness step size')

        parser.add_argument('--channel.model', type=str, help='channel: model to use')
        parser.add_argument('--channel.train_snr', type=list, help='channel: training snr(s)')
        parser.add_argument('--channel.eval_snr', type=list, help='channel: evaluate snr')
        parser.add_argument('--channel.test_snr', type=list, help='channel: test snr(s)')

        parser.add_argument('--early_stop.mode', type=str, help='early_stop: min/max mode')
        parser.add_argument('--early_stop.delta', type=float, help='early_stop: improvement quantity')
        parser.add_argument('--early_stop.patience', type=int, help='early_stop: number of epochs to wait')
        return parser

    def __str__(self):
        return self.job_name
