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
from modules.feature_encoder import FeatureEncoder
from modules.recursive_encoder import TFRecursiveEncoder, TFRecursiveDecoder

from utils import calc_loss, calc_msssim, calc_psnr
from utils import get_dataloader, get_optimizer, get_scheduler


class RecursiveCoding(BaseTrainer):
    def __init__(self, dataset, loss, staged_training, params, resume=False):
        super().__init__('RecursiveCoding', dataset, loss, resume, params.device)

        self.epoch = 0
        self.params = params
        self.save_dir = params.save_dir
        self.staged_training = staged_training

        self.feature_stages = params.feature_stages if staged_training else -1

        self._get_config(params)

    def _get_config(self, params):
        self.job_name = f'{self.trainer}({self.loss},{self.staged_training},{self.feature_stages})'

        (self.train_loader,
         self.val_loader,
         self.eval_loader), dataset_aux = self._get_data(params.dataset)

        self.encoder, self.decoder = self._get_encoder(
            params.encoder, dataset_aux['frame_sizes'], dataset_aux['reduced_arch'])

        self.modem = self._get_modem(params.modem)

        self.channel = self._get_channel(params.channel)

        all_modules = (self.encoder, self.decoder, self.modem, self.channel)
        self.optimizer, optimizer_aux = get_optimizer(params.optimizer, all_modules)
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

    def _get_encoder(self, params, frame_sizes, reduced_arch):
        down_factor = FeatureEncoder.get_config(reduced_arch)
        feat_dims = (params.c_out, *[dim // down_factor for dim in frame_sizes[1:]])
        encoder = TFRecursiveEncoder(
            c_in=params.c_in,
            c_feat=params.c_feat,
            feat_dims=feat_dims,
            reduced=reduced_arch,
            tf_layers=params.tf_layers,
            tf_heads=params.tf_heads,
            tf_ff=params.tf_ff,
            max_seq_len=params.max_seq_len
        ).to(self.device)
        decoder = TFRecursiveDecoder(
            c_in=params.c_in,
            c_feat=params.c_feat,
            feat_dims=feat_dims,
            reduced=reduced_arch,
            tf_layers=params.tf_layers,
            tf_heads=params.tf_heads,
            tf_ff=params.tf_ff,
            max_seq_len=params.max_seq_len
        ).to(self.device)
        self.job_name += '_' + str(encoder) + '_' + str(decoder)
        return encoder, decoder

    def _get_modem(self, params):
        modem = Modem(params.modem).to(self.device)
        self.job_name += '_' + str(modem)
        return modem

    def _get_channel(self, params):
        channel = Channel(params.model, params).to(self.device)
        self.job_name += '_' + str(channel)
        return channel

    def _get_gop_struct(self, n_frames):
        if self._training:
            # gop_len = np.random.randint(3, 10)  # NOTE this upperbound is due to memory
            gop_len = 5
        else:
            gop_len = 5
        return np.arange(0, n_frames+1, gop_len), gop_len

    def __call__(self, snr, *args, **kwargs):
        self.check_mode_set()

        loss_hist = []
        psnr_hist = []
        msssim_hist = []

        with tqdm(self.loader, unit='batch') as tepoch:
            for batch_idx, (frames, vid_fns) in enumerate(tepoch):
                pbar_desc = f'epoch: {self.epoch}, {self.mode} [{self.stage}]'
                tepoch.set_description(pbar_desc)

                epoch_postfix = OrderedDict()
                batch_loss = []
                batch_psnr = []
                batch_msssim = []
                predicted_frames = []

                n_frames = frames.size(1) // 3
                frames = torch.chunk(frames.to(self.device), chunks=n_frames, dim=1)

                gop_struct, gop_len = self._get_gop_struct(n_frames)
                epoch_postfix['gop'] = gop_len

                for (f_start, f_end) in zip(gop_struct[:-1], gop_struct[1:]):
                    gop = frames[f_start:f_end]
                    codeword, _ = self.encoder(gop, self.stage)

                    symbols = self.modem.modulate(codeword)

                    rx_symbols, channel_aux = self.channel(symbols, snr)
                    epoch_postfix['snr'] = '{:.2f}'.format(channel_aux['channel_snr'])

                    demod_symbols = self.modem.demodulate(rx_symbols)

                    predicted_gop, _ = self.decoder(demod_symbols, gop_len, self.stage)
                    pred_gop = torch.stack(predicted_gop, dim=1)
                    targ_gop = torch.stack(gop, dim=1)

                    gop_loss, _ = calc_loss(pred_gop, targ_gop, self.loss)
                    batch_loss.append(gop_loss.item())

                    if self._training:
                        self.optimizer.zero_grad()
                        gop_loss.backward()
                        self.optimizer.step()
                    else:
                        predicted_frames.extend(predicted_gop)

                        gop_psnr = calc_psnr(predicted_gop, gop)
                        batch_psnr.extend(gop_psnr)

                        gop_msssim = calc_msssim(predicted_gop, gop)
                        batch_msssim.extend(gop_msssim)

                loss_hist.append(np.nanmean(batch_loss))
                epoch_postfix[f'{self.loss} loss'] = '{:.5f}'.format(np.nanmean(batch_loss))
                if not self._training:
                    psnr_hist.extend(batch_psnr)
                    batch_psnr_mean = np.nanmean(batch_psnr)
                    epoch_postfix['psnr'] = '{:.5f}'.format(batch_psnr_mean)

                    msssim_hist.extend(batch_msssim)
                    batch_msssim_mean = np.nanmean(msssim_hist)
                    epoch_postfix['msssim'] = '{:.5f}'.format(batch_msssim_mean)

                tepoch.set_postfix(**epoch_postfix)

            loss_mean = np.nanmean(loss_hist)

            terminate = False
            return_aux = {}
            if not self._training:
                psnr_mean = np.nanmean(psnr_hist)
                psnr_std = np.sqrt(np.var(psnr_hist))

                msssim_mean = np.nanmean(msssim_hist)
                msssim_std = np.sqrt(np.var(msssim_hist))

                if self._validate:
                    return_aux = {
                        'psnr_mean': psnr_mean,
                        'msssim_mean': msssim_mean
                    }

                    terminate, save, update_scheduler = self._update_es(loss_mean)
                    if terminate:
                        self.load_weights()
                    elif save:
                        self.save_weights()
                        print('Saving best weights')
                    elif update_scheduler:
                        self.lr_scheduler.step()
                        print('lr updated: {:.7f}'.format(self.lr_scheduler.get_last_lr()[0]))

                elif self._evaluate:
                    terminate = False
                    return_aux = {
                        'psnr_mean': psnr_mean,
                        'psnr_std': psnr_std,
                        'msssim_mean': msssim_mean,
                        'msssim_std': msssim_std
                    }

            self.reset()
        return loss_mean, terminate, return_aux

    def _update_es(self, loss):
        save_nets = False
        update_scheduler = False

        flag, best_loss, best_epoch, bad_epochs = self.es.step(torch.Tensor([loss]), self.epoch)
        if flag:
            print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
        else:
            if bad_epochs == 0:
                save_nets = True
            elif self.scheduler_fn(bad_epochs):
                update_scheduler = True
            print('ES status: best: {:.6f}; bad epochs: {}/{}; best epoch: {}'
                    .format(best_loss.item(), bad_epochs, self.es.patience, best_epoch))
        return flag, save_nets, update_scheduler

    def _set_mode(self):
        match self.mode:
            case 'train':
                self.epoch += 1
                torch.set_grad_enabled(True)
                self.encoder.train()
                self.decoder.train()
                self.modem.train()
                self.channel.train()
                self.loader = self.train_loader
            case 'val':
                torch.set_grad_enabled(False)
                self.encoder.eval()
                self.decoder.eval()
                self.modem.eval()
                self.channel.eval()
                self.loader = self.val_loader
            case 'eval':
                torch.set_grad_enabled(False)
                self.encoder.eval()
                self.decoder.eval()
                self.modem.eval()
                self.channel.eval()
                self.loader = self.eval_loader

        self._set_stage()

    def _set_stage(self):
        if self.epoch <= self.feature_stages:
            self.stage = 'feature'
        else:
            self.stage = 'joint'

        if self.epoch == self.feature_stages+1: self.es.reset()

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
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'es': self.es.state_dict(),
            'epoch': self.epoch
        }, f'{self.save_dir}/{self.job_name}.pth')

    def load_weights(self):
        cp = torch.load(f'{self.save_dir}/{self.job_name}.pth', map_location='cpu')
        self.encoder.load_state_dict(cp['encoder'])
        self.decoder.load_state_dict(cp['decoder'])
        self.modem.load_state_dict(cp['modem'])
        self.channel.load_state_dict(cp['channel'])
        self.optimizer.load_state_dict(cp['optimizer'])
        self.lr_scheduler.load_state_dict(cp['lr_scheduler'])
        self.es.load_state_dict(cp['es'])
        self.epoch = cp['epoch']
        print('Loaded weights from epoch {}'.format(self.epoch))

    @staticmethod
    def get_parser(parser):
        # TODO organize nested namespaces better
        # can consider a separate file for defining params or putting them in their respective classes
        # depending on the modules used, the parser can be passed through each module and obtain the
        # necessary arguments to be parsed
        parser.add_argument('--save_dir', type=str, help='directory to save checkpoints')
        parser.add_argument('--staged_training', action='store_true', help='eparate feature and TF training')
        parser.add_argument('--feature_stages', type=int, help='feature stage training epochs')

        parser.add_argument('--dataset.dataset', type=str, help='dataset: dataset to use')
        parser.add_argument('--dataset.path', type=str, help='dataset: path to dataset')
        parser.add_argument('--dataset.frames_per_clip', type=int, help='dataset: number of frames to extract from each video')
        parser.add_argument('--dataset.train_batch_size', type=int, help='dataset: training batch size')
        parser.add_argument('--dataset.eval_batch_size', type=int, help='dataset: evaluate batch size')

        parser.add_argument('--optimizer.solver', type=str, help='optimizer: optimizer to use')
        parser.add_argument('--optimizer.lr', type=float, help='optimizer: optimizer learning rate')

        parser.add_argument('--optimizer.lookahead', action='store_true', help='optimizer: to use lookahead')
        parser.add_argument('--optimizer.lookahead_alpha', type=float, help='optimizer: lookahead alpha')
        parser.add_argument('--optimizer.lookahead_k', type=int, help='optimizer: lookahead steps (k)')

        parser.add_argument('--scheduler.scheduler', type=str, help='scheduler: scheduler to use')
        parser.add_argument('--scheduler.lr_schedule_factor', type=float, help='scheduler: multi_lr: reduction factor')

        parser.add_argument('--encoder.c_in', type=int, help='encoder: number of input channels')
        parser.add_argument('--encoder.c_feat', type=int, help='encoder: number of feature channels')
        parser.add_argument('--encoder.c_out', type=int, help='encoder: number of output channels')
        parser.add_argument('--encoder.tf_layers', type=int, help='encoder: number of attention layers')
        parser.add_argument('--encoder.tf_heads', type=int, help='encoder: number of attention heads')
        parser.add_argument('--encoder.tf_ff', type=int, help='encoder: number of attention dense layers')
        parser.add_argument('--encoder.max_seq_len', type=int, help='encoder: max number of frames in single codeword')

        parser.add_argument('--modem.modem', type=str, help='modem: modem to use')

        parser.add_argument('--channel.model', type=str, help='channel: model to use')
        parser.add_argument('--channel.train_snr', type=list, help='channel: training snr(s)')
        parser.add_argument('--channel.eval_snr', type=list, help='channel: evaluate snr')

        parser.add_argument('--early_stop.mode', type=str, help='early_stop: min/max mode')
        parser.add_argument('--early_stop.delta', type=float, help='early_stop: improvement quantity')
        parser.add_argument('--early_stop.patience', type=int, help='early_stop: number of epochs to wait')
        return parser

    def __str__(self):
        return self.job_name
