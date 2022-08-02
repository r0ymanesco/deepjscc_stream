import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Modem(nn.Module):
    def __init__(self, modem_type):
        super().__init__()
        self.modem_type = modem_type

        self.modem = self._get_modem()

    def _get_modem(self):
        match self.modem_type:
            case 'continuous':
                modem = ContinuousModem()
            case _:
                raise NotImplementedError
        return modem

    def modulate(self, message):
        return self.modem.modulate(message)

    def demodulate(self, symbols):
        return self.modem.demodulate(symbols)

    def __str__(self):
        return str(self.modem)


class ContinuousModem(nn.Module):
    def __init__(self, avg_power=1.):
        super().__init__()
        self.avg_power = avg_power

    def modulate(self, message):
        B = message.size(0)
        x = torch.view_as_complex(message.view(B, -1, 2))
        k = x.size(1)
        modulated = F.normalize(x) * np.sqrt(k*self.avg_power)
        return modulated

    def demodulate(self, symbols):
        demod = torch.view_as_real(symbols)
        return demod

    def __str__(self):
        return f'ContModem({self.avg_power})'
