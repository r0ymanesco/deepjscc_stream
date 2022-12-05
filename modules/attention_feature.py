import torch
import torch.nn as nn


class AFModule(nn.Module):
    def __init__(self, c_in):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=c_in+1,
                      out_features=c_in),

            nn.LeakyReLU(),

            nn.Linear(in_features=c_in,
                      out_features=c_in),

            nn.Sigmoid()
        )

    def forward(self, x, snr):
        B, _, H, W = x.size()
        context = torch.mean(x, dim=(2, 3))
        snr_context = snr.repeat_interleave(B // snr.size(0), dim=0)

        context_input = torch.cat((context, snr_context), dim=1)
        atten_weights = self.layers(context_input).view(B, -1, 1, 1)
        atten_mask = torch.repeat_interleave(atten_weights, H, dim=2)
        atten_mask = torch.repeat_interleave(atten_mask, W, dim=3)
        out = atten_mask * x
        return out
