import ipdb
import os
import numpy as np

from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.utils.data as data
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer

from modules.modem import Modem
from modules.channel import Channel
from modules.scheduler import EarlyStopping
from modules.feature_encoder import FeatureEncoder
from modules.vct_encoding_bandwidth import VCTEncoderBandwidth, VCTDecoderBandwidth, VCTPredictor

from utils import calc_loss, calc_msssim, calc_psnr
from utils import get_dataloader, get_optimizer, get_scheduler


class VCTBandwidthAllocation(BaseTrainer):
    def __init__(self, dataset, loss, params, resume=False):
        super().__init__('VCTBandwidthAllocation', dataset, loss, resume, params.device)

        self.epoch = 0
        self.params = params
        self.save_dir = params.save_dir

        self.coding_stages = params.coding_stages

        self._get_config(params)

    def _get_config(self, params):
        self.job_name = f'{self.trainer}({self.loss},{self.coding_stages})'

        (self.train_loader,
         self.val_loader,
         self.eval_loader), dataset_aux = self._get_data(params.dataset)

        self.encoder, self.decoder, self.predictor = self._get_encoder(
            params.encoder, dataset_aux['frame_sizes'], dataset_aux['reduced_arch'])

        self.modem = self._get_modem(params.modem)

        self.channel = self._get_channel(params.channel)

        coding_modules = (self.encoder, self.decoder, self.modem, self.channel)
        predictor_modules = (self.predictor,)
        self.coding_optimizer, optimizer_aux = get_optimizer(params.optimizer, coding_modules)
        self.predictor_optimizer, _ = get_optimizer(params.optimizer, predictor_modules)
        self.job_name += '_' + optimizer_aux['str']

        self.coding_scheduler, scheduler_aux = get_scheduler(self.coding_optimizer, params.scheduler)
        self.predictor_scheduler, _ = get_scheduler(self.predictor_optimizer, params.scheduler)
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
        encoder = VCTEncoderBandwidth(
            c_in=params.c_in,
            c_feat=params.c_feat,
            feat_dims=feat_dims,
            reduced=reduced_arch,
            c_win=params.c_win,
            p_win=params.p_win,
            tf_layers=params.tf_layers,
            tf_heads=params.tf_heads,
            tf_ff=params.tf_ff,
            tf_dropout=params.tf_dropout
        ).to(self.device)
        decoder = VCTDecoderBandwidth(
            c_in=params.c_in,
            c_feat=params.c_feat,
            feat_dims=feat_dims,
            reduced=reduced_arch,
            c_win=params.c_win,
            p_win=params.p_win,
            tf_layers=params.tf_layers,
            tf_heads=params.tf_heads,
            tf_ff=params.tf_ff,
            tf_dropout=params.tf_dropout
        ).to(self.device)
        predictor = VCTPredictor(
            feat_dims=feat_dims,
            c_win=params.c_win,
            p_win=params.p_win
        ).to(self.device)
        self.job_name += '_' + str(encoder) + '_' + str(decoder)
        return encoder, decoder, predictor

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
            # gop_len = np.random.randint(3, 10)
            gop_len = 3
        else:
            gop_len = 3
        return np.arange(0, n_frames+1, 1), gop_len

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

                n_frames = frames.size(1) // 3
                frames = list(torch.chunk(frames.to(self.device), chunks=n_frames, dim=1))
                frames = [torch.zeros_like(frames[0])] * 2 + frames

                gop_struct, gop_len = self._get_gop_struct(n_frames+2)

                decoder_ref = [torch.zeros_like(frames[0])] * 2
                for (f_start, f_end) in zip(gop_struct[:-gop_len], gop_struct[gop_len:]):
                    gop = frames[f_start:f_end]
                    target_frame = gop[-1]
                    code, encoder_aux = self.encoder(gop, self.stage)

                    symbols = self.modem.modulate(code)

                    rx_symbols, channel_aux = self.channel(symbols, snr)
                    epoch_postfix['snr'] = '{:.2f}'.format(channel_aux['channel_snr'])

                    demod_symbols = self.modem.demodulate(rx_symbols)

                    frame_at_rate, _ = self.decoder(demod_symbols, decoder_ref, self.stage)
                    predictions = torch.stack(frame_at_rate, dim=1)
                    target = torch.stack([target_frame]*len(frame_at_rate), dim=1)

                    dist_loss, _ = calc_loss(predictions, target, self.loss, 'none')
                    rate_dist_loss = torch.mean(dist_loss, dim=(4, 3, 2))
                    dist_loss_mean = torch.mean(dist_loss)
                    batch_loss.append(dist_loss_mean.item())

                    # NOTE use a random frame at rate for reference
                    rand_i = np.random.randint(len(frame_at_rate))
                    random_ref = frame_at_rate[rand_i]
                    decoder_ref = [decoder_ref[(i + 1) % len(decoder_ref)]
                                   for i, _ in enumerate(decoder_ref)]
                    decoder_ref[-1] = random_ref.detach()

                    match self.stage:
                        case 'coding':
                            loss = dist_loss_mean
                        case 'prediction':
                            q_pred = self.predictor(encoder_aux['conditional_tokens'])
                            loss = F.mse_loss(q_pred, rate_dist_loss)
                        case _:
                            raise ValueError

                    if self._training:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    else:
                        frame_psnr = calc_psnr(frame_at_rate, [target_frame]*len(frame_at_rate))
                        batch_psnr.extend(frame_psnr)

                        frame_msssim = calc_msssim(frame_at_rate, [target_frame]*len(frame_at_rate))
                        batch_msssim.extend(frame_msssim)

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
                self.predictor.train()
                self.loader = self.train_loader
            case 'val':
                torch.set_grad_enabled(False)
                self.encoder.eval()
                self.decoder.eval()
                self.modem.eval()
                self.channel.eval()
                self.predictor.eval()
                self.loader = self.val_loader
            case 'eval':
                torch.set_grad_enabled(False)
                self.encoder.eval()
                self.decoder.eval()
                self.modem.eval()
                self.channel.eval()
                self.predictor.eval()
                self.loader = self.eval_loader

        self._set_stage()

    def _set_stage(self):
        if self.epoch <= self.coding_stages:
            self.stage = 'coding'
            self.optimizer = self.coding_optimizer
            self.lr_scheduler = self.coding_scheduler

            self.predictor.eval()
        else:
            self.stage = 'prediction'
            self.optimizer = self.predictor_optimizer
            self.lr_scheduler = self.predictor_scheduler

            self.encoder.eval()
            self.decoder.eval()
            self.modem.eval()
            self.channel.eval()

        if self.epoch == self.coding_stages+1: self.es.reset()

    def save_weights(self):
        if not os.path.exists(self.save_dir):
            print('Creating model directory: {}'.format(self.save_dir))
            os.makedirs(self.save_dir)

        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'modem': self.modem.state_dict(),
            'channel': self.channel.state_dict(),
            'coding_optimizer': self.coding_optimizer.state_dict(),
            'predictor_optimizer': self.predictor_optimizer.state_dict(),
            'coding_scheduler': self.coding_scheduler.state_dict(),
            'predictor_scheduler': self.predictor_scheduler.state_dict(),
            'es': self.es.state_dict(),
            'epoch': self.epoch
        }, f'{self.save_dir}/{self.job_name}.pth')

    def load_weights(self):
        cp = torch.load(f'{self.save_dir}/{self.job_name}.pth')
        self.encoder.load_state_dict(cp['encoder'])
        self.decoder.load_state_dict(cp['decoder'])
        self.modem.load_state_dict(cp['modem'])
        self.channel.load_state_dict(cp['channel'])
        self.coding_optimizer.load_state_dict(cp['coding_optimizer'])
        self.predictor_optimizer.load_state_dict(cp['predictor_optimizer'])
        self.coding_scheduler.load_state_dict(cp['coding_scheduler'])
        self.predictor_scheduler.load_state_dict(cp['predictor_scheduler'])
        self.es.load_state_dict(cp['es'])
        self.epoch = cp['epoch']
        print('Loaded weights from epoch {}'.format(self.epoch))

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--save_dir', type=str, help='directory to save checkpoints')
        parser.add_argument('--coding_stages', type=int, help='feature stage training epochs')

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
        parser.add_argument('--encoder.tf_layers', type=list, help='encoder: number of attention layers')
        parser.add_argument('--encoder.tf_heads', type=int, help='encoder: number of attention heads')
        parser.add_argument('--encoder.tf_ff', type=int, help='encoder: number of attention dense layers')
        parser.add_argument('--encoder.tf_dropout', type=float, help='encoder: transformer dropout prob')
        parser.add_argument('--encoder.c_win', type=int, help='encoder: current frame window size')
        parser.add_argument('--encoder.p_win', type=int, help='encoder: past frame window size')

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
