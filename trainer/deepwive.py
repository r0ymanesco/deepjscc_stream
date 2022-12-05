import ipdb
import os
import numpy as np
from pytorch_msssim import ms_ssim

from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer

from modules.modem import Modem
from modules.channel import Channel
from modules.scheduler import EarlyStopping
from modules.feature_encoder_AF import FeatureEncoderAF, FeatureDecoderAF
from modules.scale_space_flow import ScaledSpaceFlow, ss_warp
from modules.deepwive.bandwidth_agent import BWAllocator

from utils import calc_loss, calc_msssim, calc_psnr, as_img_array
from utils import get_dataloader, get_optimizer, get_scheduler


class DeepWiVe(BaseTrainer):
    def __init__(self, dataset, loss, params, resume=False):
        super().__init__('DeepWiVe', dataset, loss, resume, params.device)

        self.epoch = 0
        self.params = params
        self.save_dir = params.save_dir

        self.key_stage = params.key_stage
        self.interp_stage = params.interp_stage
        self.bw_stage = params.bw_stage

        self._get_config(params)

    def _get_config(self, params):
        self.job_name = f'{self.trainer}({self.loss})'

        (self.train_loader,
         self.val_loader,
         self.eval_loader), dataset_aux = self._get_data(params.dataset)
        self.frame_dim = dataset_aux['frame_sizes']

        (self.key_encoder, self.key_decoder,
         self.interp_encoder, self.interp_decoder,
         self.ssf_net, self.bw_allocator) = self._get_encoder(params.encoder, self.frame_dim)

        self.modem = self._get_modem(params.modem)

        self.channel = self._get_channel(params.channel)

        key_modules = [self.key_encoder, self.key_decoder, self.modem, self.channel]
        interp_modules = key_modules + [self.interp_encoder, self.interp_decoder, self.ssf_net]
        bw_modules = [self.bw_allocator,]
        self.key_optimizer, optimizer_aux = get_optimizer(params.optimizer, key_modules)
        self.interp_optimizer, _ = get_optimizer(params.optimizer, interp_modules)
        self.bw_optimizer, _ = get_optimizer(params.optimizer, bw_modules)
        self.job_name += '_' + optimizer_aux['str']

        self.key_scheduler, scheduler_aux = get_scheduler(self.key_optimizer, params.scheduler)
        self.interp_scheduler, _ = get_scheduler(self.interp_optimizer, params.scheduler)
        self.bw_scheduler, _ = get_scheduler(self.bw_optimizer, params.scheduler)
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

    def _get_encoder(self, params, frame_sizes):
        self.gop_size = params.gop_size
        self.feat_dims = (params.c_out, *[dim // 16 for dim in frame_sizes[1:]])
        self.n_bw_chunks = params.n_bw_chunks
        self.ch_uses_per_gop = np.prod(self.feat_dims) // 2
        self.chunk_size = self.feat_dims[0] // params.n_bw_chunks
        key_encoder = FeatureEncoderAF(
            c_in=params.c_in,
            c_feat=params.c_feat,
            c_out=params.c_out
        ).to(self.device)
        key_decoder = FeatureDecoderAF(
            c_in=params.c_out,
            c_feat=params.c_feat,
            c_out=params.c_in,
            feat_dims=self.feat_dims,
        ).to(self.device)
        interp_encoder = FeatureEncoderAF(
            c_in=params.c_in*7,
            c_feat=params.c_feat,
            c_out=params.c_out
        ).to(self.device)
        interp_decoder = FeatureDecoderAF(
            c_in=params.c_out,
            c_feat=params.c_feat,
            c_out=params.c_in*4,
            feat_dims=self.feat_dims,
        ).to(self.device)
        ssf_net = ScaledSpaceFlow(
            c_in=params.c_in,
            c_feat=params.c_feat,
            ss_sigma=params.ss_sigma,
            ss_levels=params.ss_levels,
            kernel_size=3
        ).to(self.device)
        bw_allocator = BWAllocator(
            gop_size=params.gop_size,
            n_chunks=params.n_bw_chunks,
            actor_feat=params.c_feat,
            batch_size=params.policy_batch_size,
            max_memory_size=params.max_memory_size,
            device=self.device
        ).to(self.device)
        self.job_name += '_' + str(key_encoder) + '_' + str(bw_allocator)
        return key_encoder, key_decoder, interp_encoder, interp_decoder, ssf_net, bw_allocator

    def _get_modem(self, params):
        modem = Modem(params.modem).to(self.device)
        self.job_name += '_' + str(modem)
        return modem

    def _get_channel(self, params):
        channel = Channel(params.model, params).to(self.device)
        self.job_name += '_' + str(channel)
        return channel

    def _key_process(self, key_frame, batch_snr, batch_bandwidth):
        code = self.key_encoder(key_frame, batch_snr)
        code[:, batch_bandwidth:] = 0

        symbols = self.modem.modulate(code)

        rx_symbols, _ = self.channel(symbols, [batch_snr.item()])

        demod_symbols = self.modem.demodulate(rx_symbols)

        prediction = self.key_decoder(demod_symbols, batch_snr)
        return prediction

    def _interp_encode(self, target, ref, batch_bandwidth, batch_snr):
        ssf_vol1 = self.ssf_net.generate_ss_volume(ref[0])
        ssf_vol2 = self.ssf_net.generate_ss_volume(ref[1])

        flow1 = self.ssf_net(torch.cat((target, ref[0]), dim=1))
        flow2 = self.ssf_net(torch.cat((target, ref[1]), dim=1))

        w1 = ss_warp(ssf_vol1, flow1.unsqueeze(2), False if self.stage == 'bw' else True)
        w2 = ss_warp(ssf_vol2, flow2.unsqueeze(2), False if self.stage == 'bw' else True)

        r1 = target - w1
        r2 = target - w2
        interp_input = torch.cat((target, w1, w2, r1, r2, flow1, flow2), dim=1)

        code = self.interp_encoder(interp_input, batch_snr)
        code[:, batch_bandwidth:] = 0
        return code, interp_input

    def _interp_decode(self, code, ref, batch_snr):
        decoder_out = self.interp_decoder(code, batch_snr)

        f1, f2, a, r = torch.chunk(decoder_out, chunks=4, dim=1)
        a = F.softmax(a, dim=1)
        a1, a2, a3 = torch.chunk(a, chunks=3, dim=1)
        r = torch.sigmoid(r)

        a1 = a1.repeat_interleave(3, dim=1)
        a2 = a2.repeat_interleave(3, dim=1)
        a3 = a3.repeat_interleave(3, dim=1)

        ssf_vol1 = self.ssf_net.generate_ss_volume(ref[0])
        ssf_vol2 = self.ssf_net.generate_ss_volume(ref[1])

        pred_1 = ss_warp(ssf_vol1, f1.unsqueeze(2), False if self.stage == 'bw' else True)
        pred_2 = ss_warp(ssf_vol2, f2.unsqueeze(2), False if self.stage == 'bw' else True)
        prediction = a1 * pred_1 + a2 * pred_2 + a3 * r
        return prediction

    # def _bw_process(self, frame_state, batch_snr):
    #     allocation, policy = self.bw_allocator.get_action((frame_state, batch_snr), self.mode)
    #     return allocation, policy

    def _get_gop_struct(self, n_frames):
        match self.gop_size:
            case 4:
                interp_struct = [2, 1, 3]
                interp_dist = [2, 1, 1]
                gop_idxs = np.arange(1, n_frames+1, self.gop_size)
            case _:
                raise NotImplementedError
        return interp_struct, interp_dist, gop_idxs

    def __call__(self, snr, *args, **kwargs):
        self.check_mode_set()

        terminate = False
        epoch_trackers = {
            'loss_hist': [],
            'dist_loss_hist': [],
            'psnr_hist': [],
            'msssim_hist': [],
        }

        with tqdm(self.loader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
            for batch_idx, (frames, vid_fns) in enumerate(tepoch):
                pbar_desc = f'epoch: {self.epoch}, {self.mode} [{self.stage}]'
                tepoch.set_description(pbar_desc)

                epoch_postfix = OrderedDict()
                batch_trackers = {
                    'batch_loss': [],
                    'batch_dist_loss': [],
                    'batch_psnr': [],
                    'batch_msssim': [],
                }

                n_frames = frames.size(1) // 3
                frames = list(torch.chunk(frames.to(self.device), chunks=n_frames, dim=1))
                frames = [frame.squeeze(1) for frame in frames]

                match self.stage:
                    case 'key':
                        batch_snr = (snr[1] - snr[0]) * torch.rand((1, 1), device=self.device) + snr[0]
                        epoch_postfix['snr'] = '{:.2f}'.format(batch_snr.item())

                        batch_bandwidth = int(np.random.randint(self.n_bw_chunks+1) * self.chunk_size)
                        _, _, gop_idxs = self._get_gop_struct(n_frames)
                        rand_gop = np.random.randint(len(gop_idxs)-1)
                        i_start = gop_idxs[rand_gop]
                        i_end = gop_idxs[rand_gop+1]
                        key_frame = torch.cat(frames[i_start:i_end], dim=0)

                        predicted_frame = self._key_process(key_frame, batch_snr, batch_bandwidth)
                        loss, batch_trackers = self._get_loss([predicted_frame], [key_frame], batch_trackers)
                        if self._training: self._update_params(loss)
                    case 'interp':
                        batch_bandwidth = int(np.random.randint(self.n_bw_chunks+1) * self.chunk_size)
                        interp_struct, interp_dist, gop_idxs = self._get_gop_struct(n_frames)
                        predictions = []

                        for gop_idx, (i_start, i_end) in enumerate(zip(gop_idxs[:-1], gop_idxs[1:])):
                            batch_snr = (snr[1] - snr[0]) * torch.rand((1, 1), device=self.device) + snr[0]
                            epoch_postfix['snr'] = '{:.2f}'.format(batch_snr.item())

                            gop = frames[i_start-1:i_end]
                            gop_predictions = [torch.zeros(1) for _ in range(len(gop))]

                            if gop_idx == 0:
                                # NOTE init frame uses full bw
                                init_frame = frames[0]
                                init_prediction = self._key_process(init_frame, batch_snr, int(self.n_bw_chunks*self.chunk_size))
                                loss, batch_trackers = self._get_loss([init_prediction], [init_frame], batch_trackers)
                                if self._training: self._update_params(loss)
                                predictions.append(init_prediction.detach())

                            first_key = predictions[i_start-1]
                            gop_predictions[0] = first_key

                            last_key = gop[-1]
                            last_key_prediction = self._key_process(last_key, batch_snr, batch_bandwidth)
                            gop_predictions[-1] = last_key_prediction

                            for (pred_idx, t) in zip(interp_struct, interp_dist):
                                target_frame = gop[pred_idx]
                                ref = (gop_predictions[pred_idx-t], gop_predictions[pred_idx+t])
                                code, _ = self._interp_encode(target_frame, ref, batch_bandwidth, batch_snr)
                                symbols = self.modem.modulate(code)
                                rx_symbols, _ = self.channel(symbols, [batch_snr.item()])
                                demod_symbols = self.modem.demodulate(rx_symbols)
                                prediction = self._interp_decode(demod_symbols, ref, batch_snr)
                                gop_predictions[pred_idx] = prediction

                            loss, batch_trackers = self._get_loss(gop_predictions[1:], gop[1:], batch_trackers)
                            if self._training: self._update_params(loss)
                            predictions.extend([pred.detach() for pred in gop_predictions[1:]])
                    case 'bw':
                        interp_struct, interp_dist, gop_idxs = self._get_gop_struct(n_frames)
                        predictions = []

                        experience = {}
                        for gop_idx, (i_start, i_end) in enumerate(zip(gop_idxs[:-1], gop_idxs[1:])):
                            batch_snr = (snr[1] - snr[0]) * torch.rand((1, 1), device=self.device) + snr[0]
                            epoch_postfix['snr'] = '{:.2f}'.format(batch_snr.item())

                            loss = 0
                            gop = frames[i_start-1:i_end]
                            gop_predictions = [[] for _ in range(len(gop))]
                            emulated_predictions = [torch.zeros(1) for _ in range(len(gop))]
                            state_inputs = [torch.zeros(1) for _ in range(len(gop))]

                            if gop_idx == 0:
                                init_frame = frames[0]
                                init_prediction = self._key_process(init_frame, batch_snr, int(self.n_bw_chunks*self.chunk_size))
                                predictions.append(init_prediction.detach())

                            first_key = predictions[i_start-1]
                            gop_predictions[0] = torch.chunk(first_key, chunks=first_key.size(0), dim=0)

                            # Channel emulation assuming full bw
                            emulated_predictions[0] = first_key
                            state_inputs[0] = first_key
                            last_key = gop[-1]
                            last_key_emulate = self._key_process(last_key, batch_snr, int(self.n_bw_chunks*self.chunk_size))
                            emulated_predictions[-1] = last_key_emulate
                            state_inputs[-1] = last_key_emulate

                            for (pred_idx, t) in zip(interp_struct, interp_dist):
                                target_frame = gop[pred_idx]
                                ref = (emulated_predictions[pred_idx-t], emulated_predictions[pred_idx+t])
                                code, interp_input = self._interp_encode(target_frame, ref, int(self.n_bw_chunks*self.chunk_size), batch_snr)
                                symbols = self.modem.modulate(code)
                                rx_symbols, _ = self.channel(symbols, [batch_snr.item()])
                                demod_symbols = self.modem.demodulate(rx_symbols)
                                prediction = self._interp_decode(demod_symbols, ref, batch_snr)
                                emulated_predictions[pred_idx] = prediction
                                state_inputs[pred_idx] = interp_input

                            frame_state = torch.cat(state_inputs, dim=1)
                            allocation, action_aux = self.bw_allocator.get_action((frame_state, batch_snr), self.mode)
                            epoch_postfix['eps'] = action_aux['eps_threshold']
                            # FIXME these for loops are bad for performance
                            for b_idx, b_alloc in enumerate(allocation):
                                b_gop_predictions = [torch.zeros(1)] * len(gop)
                                b_gop_predictions[0] = gop_predictions[0][b_idx]

                                last_key_prediction = self._key_process(last_key[b_idx].unsqueeze(0), batch_snr,
                                                                        int(b_alloc[0].item()*self.chunk_size))
                                b_gop_predictions[-1] = last_key_prediction
                                gop_predictions[-1].append(last_key_prediction)
                                for (pred_idx, t) in zip(interp_struct, interp_dist):
                                    target_frame = gop[pred_idx][b_idx].unsqueeze(0)
                                    ref = (b_gop_predictions[pred_idx-t], b_gop_predictions[pred_idx+t])
                                    code, _ = self._interp_encode(target_frame, ref,
                                                                  int(b_alloc[pred_idx].item()*self.chunk_size), batch_snr)
                                    symbols = self.modem.modulate(code)
                                    rx_symbols, _ = self.channel(symbols, [batch_snr.item()])
                                    demod_symbols = self.modem.demodulate(rx_symbols)
                                    prediction = self._interp_decode(demod_symbols, ref, batch_snr)
                                    b_gop_predictions[pred_idx] = prediction
                                    gop_predictions[pred_idx].append(prediction)

                            gop_predictions = [torch.cat(pred, dim=0) for pred in gop_predictions]
                            predictions.extend([pred.detach() for pred in gop_predictions[1:]])

                            loss, batch_trackers = self._get_loss(gop_predictions, gop, batch_trackers)

                            if gop_idx > 0:
                                experience['next_state'] = (frame_state.clone(), batch_snr.clone())
                                self.bw_allocator.push_experience(experience)

                            experience['state'] = (frame_state.clone(), batch_snr.clone())
                            experience['action'] = action_aux['policy'].clone()
                            experience['reward'] = batch_trackers['reward'].clone()

                            if self._training and batch_trackers['buffer_ready']: self._update_params(loss)
                    case _:
                        raise ValueError

                epoch_trackers, epoch_postfix = self._update_epoch_postfix(batch_trackers,
                                                                           epoch_trackers,
                                                                           epoch_postfix)
                tepoch.set_postfix(**epoch_postfix)

            loss_mean, return_aux = self._get_return_aux(epoch_trackers)
            if self._validate: terminate = self._update_es(loss_mean)

            self.reset()
            return loss_mean, terminate, return_aux

    def _get_reward(self, predicted_frames, target_frames):
        match self.loss:
            case 'l2':
                predictions = torch.stack(predicted_frames, dim=1)
                target = torch.stack(target_frames, dim=1)
                batch_mse = torch.mean((as_img_array(predictions) - as_img_array(target)) ** 2.,
                                       dim=(4, 3, 2, 1),
                                       dtype=torch.float32)
                reward = 20 * torch.log10(255. / torch.sqrt(batch_mse))
            case 'msssim':
                metric = []
                for (pred, targ) in zip(predicted_frames, target_frames):
                    original = as_img_array(targ)
                    prediction = as_img_array(pred)
                    msssim = ms_ssim(original, prediction,
                                     data_range=255, size_average=False)
                    metric.append(msssim)
                reward = (sum(metric) / len(metric)) * 10
            case _:
                raise ValueError
        return reward

    def _get_return_aux(self, epoch_trackers):
        return_aux = {}
        loss_mean = np.nanmean(epoch_trackers['loss_hist'])

        if self.stage == 'bw':
            return_aux['dist_loss'] = np.nanmean(epoch_trackers['dist_loss_hist'])

        if not self._training:
            psnr_mean = np.nanmean(epoch_trackers['psnr_hist'])
            msssim_mean = np.nanmean(epoch_trackers['msssim_hist'])

            if self._validate:
                return_aux['psnr_mean'] = psnr_mean
                return_aux['msssim_mean'] = msssim_mean

            elif self._evaluate:
                psnr_std = np.sqrt(np.var(epoch_trackers['psnr_hist']))
                msssim_std = np.sqrt(np.var(epoch_trackers['msssim_hist']))

                return_aux['psnr_mean'] = psnr_mean
                return_aux['psnr_std'] = psnr_std
                return_aux['msssim_mean'] = msssim_mean
                return_aux['msssim_std'] = msssim_std
        return loss_mean, return_aux

    def _update_epoch_postfix(self, batch_trackers, epoch_trackers, epoch_postfix):
        if self.stage == 'bw':
            if len(batch_trackers['batch_loss']) > 0:
                epoch_trackers['loss_hist'].append(np.nanmean(batch_trackers['batch_loss']))
                epoch_postfix['Q loss'] = '{:.5f}'.format(epoch_trackers['loss_hist'][-1])

            epoch_trackers['dist_loss_hist'].append(np.nanmean(batch_trackers['batch_dist_loss']))
            epoch_postfix[f'{self.loss} loss'] = '{:.5f}'.format(epoch_trackers['dist_loss_hist'][-1])
        else:
            epoch_trackers['loss_hist'].append(np.nanmean(batch_trackers['batch_loss']))
            epoch_postfix[f'{self.loss} loss'] = '{:.5f}'.format(epoch_trackers['loss_hist'][-1])

        if not self._training:
            epoch_trackers['psnr_hist'].extend(batch_trackers['batch_psnr'])
            batch_psnr_mean = np.nanmean(batch_trackers['batch_psnr'])
            epoch_postfix['psnr'] = '{:.5f}'.format(batch_psnr_mean)

            epoch_trackers['msssim_hist'].extend(batch_trackers['batch_msssim'])
            batch_msssim_mean = np.nanmean(batch_trackers['batch_msssim'])
            epoch_postfix['msssim'] = '{:.5f}'.format(batch_msssim_mean)
        return epoch_trackers, epoch_postfix

    def _get_loss(self, predicted_frames, target_frames, batch_trackers):
        predictions = torch.stack(predicted_frames, dim=1)
        target = torch.stack(target_frames, dim=1)

        match self.stage:
            case 'key':
                loss, _ = calc_loss(predictions, target, self.loss)
                batch_trackers['batch_loss'].append(loss.item())

                if not self._training:
                    frame_psnr = calc_psnr(predicted_frames, target_frames)
                    batch_trackers['batch_psnr'].extend(frame_psnr)

                    frame_msssim = calc_msssim(predicted_frames, target_frames)
                    batch_trackers['batch_msssim'].extend(frame_msssim)
            case 'interp':
                loss, _ = calc_loss(predictions, target, self.loss)
                batch_trackers['batch_loss'].append(loss.item())

                if not self._training:
                    frame_psnr = calc_psnr(predicted_frames, target_frames)
                    batch_trackers['batch_psnr'].extend(frame_psnr)

                    frame_msssim = calc_msssim(predicted_frames, target_frames)
                    batch_trackers['batch_msssim'].extend(frame_msssim)
            case 'bw':
                flag, loss = self.bw_allocator.get_loss()
                batch_trackers['buffer_ready'] = flag
                if flag: batch_trackers['batch_loss'].append(loss.item())

                dist_loss, _ = calc_loss(predictions, target, self.loss)
                batch_trackers['batch_dist_loss'].append(dist_loss.item())

                frame_psnr = calc_psnr(predicted_frames, target_frames)
                batch_trackers['batch_psnr'].extend(frame_psnr)

                frame_msssim = calc_msssim(predicted_frames, target_frames)
                batch_trackers['batch_msssim'].extend(frame_msssim)

                batch_trackers['reward'] = self._get_reward(predicted_frames, target_frames)
            case _:
                raise ValueError
        return loss, batch_trackers

    def _update_es(self, loss):
        flag, best_loss, best_epoch, bad_epochs = self.es.step(torch.Tensor([loss]), self.epoch)
        if flag:
            match self.stage:
                case 'key':
                    flag = False
                    self.load_weights()

                    self.key_stage = self.epoch
                    self.stage = 'interp'
                    self.es.reset()
                case 'interp':
                    flag = False
                    self.load_weights()

                    self.interp_stage = self.epoch
                    self.stage = 'bw'
                    self.es.reset()
                case _:
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
                self.key_encoder.train()
                self.key_encoder.requires_grad_(True)

                self.key_decoder.train()
                self.key_decoder.requires_grad_(True)

                self.interp_encoder.train()
                self.interp_encoder.requires_grad_(True)

                self.interp_decoder.train()
                self.interp_decoder.requires_grad_(True)

                self.ssf_net.train()
                self.ssf_net.requires_grad_(True)

                self.bw_allocator.train()
                self.bw_allocator.requires_grad_(True)

                self.modem.train()
                self.modem.requires_grad_(True)

                self.channel.train()
                self.channel.requires_grad_(True)

                self.loader = self.train_loader
            case 'val':
                torch.set_grad_enabled(False)
                self.key_encoder.eval()
                self.key_encoder.requires_grad_(False)

                self.key_decoder.eval()
                self.key_decoder.requires_grad_(False)

                self.interp_encoder.eval()
                self.interp_encoder.requires_grad_(False)

                self.interp_decoder.eval()
                self.interp_decoder.requires_grad_(False)

                self.ssf_net.eval()
                self.ssf_net.requires_grad_(False)

                self.bw_allocator.eval()
                self.bw_allocator.requires_grad_(False)

                self.modem.eval()
                self.modem.requires_grad_(False)

                self.channel.eval()
                self.channel.requires_grad_(False)

                self.loader = self.val_loader
            case 'eval':
                torch.set_grad_enabled(False)
                self.key_encoder.eval()
                self.key_encoder.requires_grad_(False)

                self.key_decoder.eval()
                self.key_decoder.requires_grad_(False)

                self.interp_encoder.eval()
                self.interp_encoder.requires_grad_(False)

                self.interp_decoder.eval()
                self.interp_decoder.requires_grad_(False)

                self.ssf_net.eval()
                self.ssf_net.requires_grad_(False)

                self.bw_allocator.eval()
                self.bw_allocator.requires_grad_(False)

                self.modem.eval()
                self.modem.requires_grad_(False)

                self.channel.eval()
                self.channel.requires_grad_(False)

                self.loader = self.eval_loader

        self._set_stage()

    def _set_stage(self):
        if self.epoch <= self.key_stage:
            self.stage = 'key'
            self.optimizer = self.key_optimizer
            self.lr_scheduler = self.key_scheduler

            self.interp_encoder.eval()
            self.interp_encoder.requires_grad_(False)

            self.interp_decoder.eval()
            self.interp_decoder.requires_grad_(False)

            self.ssf_net.eval()
            self.ssf_net.requires_grad_(False)

            self.bw_allocator.eval()
            self.bw_allocator.requires_grad_(False)
        elif self.epoch <= self.interp_stage:
            self.stage = 'interp'
            self.optimizer = self.interp_optimizer
            self.lr_scheduler = self.interp_scheduler
            if self.epoch == self.key_stage+1: self.es.reset()

            self.bw_allocator.eval()
            self.bw_allocator.requires_grad_(False)
        else:
            self.stage = 'bw'
            self.optimizer = self.bw_optimizer
            self.lr_scheduler = self.bw_scheduler
            if self.epoch == self.interp_stage+1: self.es.reset()

            self.key_encoder.eval()
            self.key_encoder.requires_grad_(False)

            self.key_decoder.eval()
            self.key_decoder.requires_grad_(False)

            self.interp_encoder.eval()
            self.interp_encoder.requires_grad_(False)

            self.interp_decoder.eval()
            self.interp_decoder.requires_grad_(False)

            self.ssf_net.eval()
            self.ssf_net.requires_grad_(False)

            self.modem.eval()
            self.modem.requires_grad_(False)

            self.channel.eval()
            self.channel.requires_grad_(False)

    def save_weights(self):
        if not os.path.exists(self.save_dir):
            print('Creating model directory: {}'.format(self.save_dir))
            os.makedirs(self.save_dir)

        torch.save({
            'stage': self.stage,
            'key_encoder': self.key_encoder.state_dict(),
            'key_decoder': self.key_decoder.state_dict(),
            'interp_encoder': self.interp_encoder.state_dict(),
            'interp_decoder': self.interp_decoder.state_dict(),
            'ssf_net': self.ssf_net.state_dict(),
            'bw_allocator': self.bw_allocator.state_dict(),
            'modem': self.modem.state_dict(),
            'channel': self.channel.state_dict(),
            'key_optimizer': self.key_optimizer.state_dict(),
            'interp_optimizer': self.interp_optimizer.state_dict(),
            'bw_optimizer': self.bw_optimizer.state_dict(),
            'key_scheduler': self.key_scheduler.state_dict(),
            'interp_scheduler': self.interp_scheduler.state_dict(),
            'bw_scheduler': self.bw_scheduler.state_dict(),
            'es': self.es.state_dict(),
            'epoch': self.epoch
        }, '{}/{}.pth'.format(self.save_dir, self.job_name))

    def load_weights(self):
        cp = torch.load('{}/{}.pth'.format(self.save_dir, self.job_name), map_location='cpu')

        self.stage = cp['stage']

        self.key_encoder.load_state_dict(cp['key_encoder'])
        self.key_decoder.load_state_dict(cp['key_decoder'])
        self.interp_encoder.load_state_dict(cp['interp_encoder'])
        self.interp_decoder.load_state_dict(cp['interp_decoder'])
        self.ssf_net.load_state_dict(cp['ssf_net'])
        self.bw_allocator.load_state_dict(cp['bw_allocator'])
        self.modem.load_state_dict(cp['modem'])
        self.channel.load_state_dict(cp['channel'])

        self.key_optimizer.load_state_dict(cp['key_optimizer'])
        self.interp_optimizer.load_state_dict(cp['interp_optimizer'])
        self.bw_optimizer.load_state_dict(cp['bw_optimizer'])

        self.key_scheduler.load_state_dict(cp['key_scheduler'])
        self.interp_scheduler.load_state_dict(cp['interp_scheduler'])
        self.bw_scheduler.load_state_dict(cp['bw_scheduler'])

        self.es.load_state_dict(cp['es'])
        self.epoch = cp['epoch']
        print('Loaded weights from epoch {}'.format(self.epoch))

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--save_dir', type=str, help='directory to save checkpoints')
        parser.add_argument('--key_stage', type=int, help='key frame stage training epochs')
        parser.add_argument('--interp_stage', type=int, help='interpolation stage training epochs')
        parser.add_argument('--bw_stage', type=int, help='bandwidth stage training epochs')

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
        parser.add_argument('--encoder.ss_sigma', type=float, help='encoder: standard deviation of the Gaussian kernel for ssflow')
        parser.add_argument('--encoder.ss_levels', type=int, help='encoder: number of levels for ssflow')
        parser.add_argument('--encoder.gop_size', type=int, help='encoder: number frames in a GoP')
        parser.add_argument('--encoder.n_bw_chunks', type=int, help='encoder: number of chunks to allocate in a GoP')
        parser.add_argument('--encoder.policy_batch_size', type=int, help='encoder: policy training batch size')
        parser.add_argument('--encoder.max_memory_size', type=int, help='encoder: policy experience buffer size')

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


def print_len(gop_predictions):
    for buff in gop_predictions:
        print(len(buff))
