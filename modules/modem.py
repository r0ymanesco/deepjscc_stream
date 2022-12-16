import ipdb
import numpy as np
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.quantizers import SoftHardQuantize, LearnedQuantize


def get_constellation(M):
        bits_per_symbol = int(np.log2(M))
        if (M == 1) or ((np.log2(M) % 2 != 0) and M != 2):
            raise ValueError('M must be even power of 2')

        if M == 2:
            constellation = np.array([[1., -1.], [0., 0.]], dtype=np.float32)
            nGray = np.array([0, 1])
        else:
            n = np.arange(0, M)  # Sequential address from 0 to M-1 (1xM dimension)
            a = np.asarray([i ^ (i >> 1) for i in n])  # convert linear addresses to Gray code
            D = np.sqrt(M).astype(int)  # Dimension of K-Map - N x N matrix
            a = np.reshape(a, (D, D))  # NxN gray coded matrix
            # oddRows = np.arange(start=1, stop = D ,step=2)  # identify alternate rows

            nGray = np.reshape(a, (M))  # reshape to 1xM - Gray code walk on KMap
            # Construction of ideal M-QAM constellation from sqrt(M)-PAM
            (x, y) = np.divmod(nGray, D)  # element-wise quotient and remainder
            Ax = 2*x+1-D  # PAM Amplitudes 2d+1-D - real axis
            Ay = 2*y+1-D  # PAM Amplitudes 2d+1-D - imag axis
            constellation = np.array([Ax, Ay], dtype=np.single) / ((np.sqrt(M) - 1) * np.sqrt(2))

        constellation, gray_idx = torch.from_numpy(constellation).transpose(1, 0), torch.from_numpy(nGray)
        _, gray_order = torch.sort(gray_idx)
        gray_constellation = torch.index_select(constellation, 0, gray_order)
        constellation_bin = torch.tensor(list(product([0, 1], repeat=bits_per_symbol)), dtype=torch.float)
        return gray_constellation.transpose(1, 0), constellation_bin


def LLR_AWGN(symbols, noise_var, constellation, constellation_bin):
    B, N, _ = symbols.size()
    input_cplx = torch.view_as_complex(symbols).view(1, -1).repeat_interleave(constellation.size(0), dim=0)

    const_real = constellation[:, 0].unsqueeze(0)
    const_imag = constellation[:, 1].unsqueeze(0)
    const_cplx = (const_real + const_imag*1j).view(-1, 1).repeat_interleave(input_cplx.size(1), dim=1)

    LLR_mtrx = torch.exp((-torch.abs(input_cplx - const_cplx) ** 2) / noise_var)

    llr_num = constellation_bin
    llr_num = torch.matmul(LLR_mtrx.transpose(1, 0), llr_num)

    llr_den = (torch.ones_like(constellation_bin) - constellation_bin)
    llr_den = torch.matmul(LLR_mtrx.transpose(1, 0), llr_den)

    LLR_out = torch.log(llr_num / llr_den)
    LLR_out = LLR_out.view(B, N, -1)
    return LLR_out


class Modem(nn.Module):
    def __init__(self, modem_type, *args, **kwargs):
        super().__init__()
        self.modem_type = modem_type
        self.modem = self._get_modem(*args, **kwargs)

    def _get_modem(self, *args, **kwargs):
        match self.modem_type:
            case 'continuous':
                modem = ContinuousModem()
            case 'qam':
                modem = QAM(*args, **kwargs)
            case 'lm':
                modem = LearnedConstellation(*args, **kwargs)
            case _:
                raise NotImplementedError
        return modem

    def modulate(self, message, *args, **kwargs):
        return self.modem.modulate(message, *args, **kwargs)

    def demodulate(self, symbols, *args, **kwargs):
        return self.modem.demodulate(symbols, *args, **kwargs)

    def __str__(self):
        return str(self.modem)


class ContinuousModem(nn.Module):
    def __init__(self, avg_power=1.):
        super().__init__()
        self.avg_power = avg_power

    def modulate(self, message, channel_uses=None):
        B = message.size(0)
        x = torch.view_as_complex(message.view(B, -1, 2))

        if channel_uses is None:
            k = torch.tensor([x.size(1)]).to(x.device)
        else:
            k = channel_uses.to(x.device)
        modulated = F.normalize(x) * torch.sqrt(k*self.avg_power)
        return modulated

    def demodulate(self, symbols):
        demod = torch.view_as_real(symbols)
        return demod

    def __str__(self):
        return f'ContModem({self.avg_power})'


class QAM(nn.Module):
    def __init__(self, mod_order, return_likelihoods,
                 commitment, anneal, sigma_start, sigma_max, sigma_period, sigma_scale):
        super().__init__()
        self.mod_order = mod_order
        self.return_likelihoods = return_likelihoods

        self.anneal = anneal
        self.sigma_start = sigma_start
        self.sigma_max = sigma_max

        constellation, constellation_bin = get_constellation(mod_order)
        self.modulator = SoftHardQuantize(mod_order, 2, constellation,
                                          commitment, anneal,
                                          sigma_start, sigma_max, sigma_period, sigma_scale)
        self.constellation_bin = constellation_bin

    def modulate(self, code):
        real_modulated, mod_aux = self.modulator(code)
        modulated = torch.view_as_complex(real_modulated)
        return modulated, mod_aux

    def demodulate(self, symbols, channel_model=None, noise_var=0):
        symbols = torch.view_as_real(symbols)
        if self.return_likelihoods:
            match channel_model:
                case 'awgn':
                    constellation = self.modulator.embed
                    return LLR_AWGN(symbols, noise_var, constellation, self.constellation_bin)
                case _:
                    raise NotImplementedError
        else:
            return symbols

    def __str__(self):
        return f'QAM({self.mod_order},{self.anneal},{self.sigma_start},{self.sigma_max})'


class LearnedConstellation(nn.Module):
    def __init__(self, mod_order, return_likelihoods,
                 commitment, anneal, sigma_start, sigma_max, sigma_period, sigma_scale):
        super().__init__()
        self.mod_order = mod_order
        self.return_likelihoods = return_likelihoods

        self.anneal = anneal
        self.sigma_start = sigma_start
        self.sigma_max = sigma_max

        constellation, constellation_bin = get_constellation(mod_order)
        self.modulator = LearnedQuantize(mod_order, 2, constellation, True,
                                         commitment, anneal,
                                         sigma_start, sigma_max, sigma_period, sigma_scale)
        self.constellation_bin = constellation_bin

    def modulate(self, code):
        real_modulated, mod_aux = self.modulator(code)
        modulated = torch.view_as_complex(real_modulated)
        return modulated, mod_aux

    def demodulate(self, symbols, channel_model=None, noise_var=0):
        symbols = torch.view_as_real(symbols)
        if self.return_likelihoods:
            match channel_model:
                case 'awgn':
                    constellation = self.modulator.embed
                    return LLR_AWGN(symbols, noise_var, constellation, self.constellation_bin)
                case _:
                    raise NotImplementedError
        else:
            return symbols

    def __str__(self):
        return f'LM({self.mod_order},{self.anneal},{self.sigma_start},{self.sigma_max})'
