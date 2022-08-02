import ipdb
import numpy as np

import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, channel_model, params):
        super().__init__()
        self.channel_model = channel_model
        self.channel = self._get_channel(params)

    def _get_channel(self, params):
        match self.channel_model:
            case 'awgn':
                channel = AWGN(params.train_snr, params.eval_snr)
            case _:
                raise NotImplementedError
        return channel

    def __call__(self, symbols, *args, **kwargs):
        return self.channel(symbols, *args, **kwargs)

    def __str__(self):
        return str(self.channel)


class AWGN(nn.Module):
    def __init__(self, train_snr, eval_snr):
        super().__init__()
        self.train_snr = train_snr
        self.eval_snr = eval_snr

    def _get_snr(self, snr_in):
        match len(snr_in):
            case 1:
                snr = torch.tensor(snr_in, dtype=torch.float)
            case 2:
                snr_lower = snr_in[0]
                snr_higher = snr_in[1]
                snr = (snr_higher - snr_lower) * torch.rand(1, 1) + snr_lower
            case _:
                raise ValueError
        return snr

    def __call__(self, symbols, snr, *args, **kwargs):
        channel_aux = {}
        snr_val = self._get_snr(snr).to(symbols.device)
        channel_aux['channel_snr'] = snr_val.item()

        noise_shape = [*symbols.shape, 2]
        Es = torch.mean(symbols.abs() ** 2)
        No = (Es / (10 ** (snr_val/10))) / 2
        awgn = torch.view_as_complex(torch.randn(noise_shape, device=symbols.device) * torch.sqrt(No))
        noisy = symbols + awgn
        return noisy, channel_aux

    def __str__(self):
        return f'AWGN({self.train_snr},{self.eval_snr})'
