import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from compressai.layers import ResidualBlockUpsample, AttentionBlock
from compressai.layers import ResidualBlock, ResidualBlockWithStride

from modules.attention_feature import AFModule


class FeatureEncoderAF(nn.Module):
    def __init__(self, c_in, c_feat, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        self.layers = nn.ModuleDict({
            'rbws1': ResidualBlockWithStride(
                in_ch=c_in,
                out_ch=c_feat,
                stride=2),

            'af1': AFModule(c_in=c_feat),

            'rb1': ResidualBlock(
                in_ch=c_feat,
                out_ch=c_feat),

            'rbws2': ResidualBlockWithStride(
                in_ch=c_feat,
                out_ch=c_feat,
                stride=2),

            'af2': AFModule(c_in=c_feat),

            'a1': AttentionBlock(c_feat),

            'rb2': ResidualBlock(
                in_ch=c_feat,
                out_ch=c_feat),

            'rbws3': ResidualBlockWithStride(
                in_ch=c_feat,
                out_ch=c_feat,
                stride=2),

            'af3': AFModule(c_in=c_feat),

            'rb3': ResidualBlock(
                in_ch=c_feat,
                out_ch=c_feat),

            'rbws4': ResidualBlockWithStride(
                in_ch=c_feat,
                out_ch=c_out,
                stride=2),

            'af4': AFModule(c_in=c_out),

            'a2': AttentionBlock(c_out),
        })

    def run_fn(self, module_key):
        def custom_forward(*inputs):
            if module_key[:2] == 'af':
                x, snr = inputs
                x = self.layers[module_key](x, snr)
            else:
                x = inputs[0]
                x = self.layers[module_key](x)
            return x
        return custom_forward

    def forward(self, x, snr):
        # x = x.requires_grad_()
        # for key in self.layers:
        #     if key[:2] == 'af':
        #         x = checkpoint.checkpoint(
        #             self.run_fn(key), x, snr)
        #     else:
        #         x = checkpoint.checkpoint(
        #             self.run_fn(key), x)

        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr)
            else:
                x = self.layers[key](x)
        return x

    def __str__(self):
        return f'FeatureEncoderAF({self.c_in},{self.c_feat},{self.c_out})'


class FeatureDecoderAF(nn.Module):
    def __init__(self, c_in, c_feat, c_out, feat_dims):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out
        self.feat_dims = feat_dims

        self.layers = nn.ModuleDict({
            'a1': AttentionBlock(c_in),

            'rb1': ResidualBlock(
                in_ch=c_in,
                out_ch=c_in),

            'rbu1': ResidualBlockUpsample(
                in_ch=c_in,
                out_ch=c_feat,
                upsample=2),

            'af1': AFModule(c_in=c_feat),

            'rb2': ResidualBlock(
                in_ch=c_feat,
                out_ch=c_feat),

            'rbu2': ResidualBlockUpsample(
                in_ch=c_feat,
                out_ch=c_feat,
                upsample=2),

            'af2': AFModule(c_in=c_feat),

            'a2': AttentionBlock(c_feat),

            'rb3': ResidualBlock(
                in_ch=c_feat,
                out_ch=c_feat),

            'rbu3': ResidualBlockUpsample(
                in_ch=c_feat,
                out_ch=c_feat,
                upsample=2),

            'af3': AFModule(c_in=c_feat),

            'rb4': ResidualBlock(
                in_ch=c_feat,
                out_ch=c_feat),

            'rbu4': ResidualBlockUpsample(
                in_ch=c_feat,
                out_ch=c_out,
                upsample=2),

            'af4': AFModule(c_in=c_out),

            'sigmoid': nn.Sigmoid()
        })

    def run_fn(self, module_key):
        def custom_forward(*inputs):
            if module_key[:2] == 'af':
                x, snr = inputs
                x = self.layers[module_key](x, snr)
            else:
                x = inputs[0]
                x = self.layers[module_key](x)
            return x
        return custom_forward

    def forward(self, x, snr):
        # x = x.requires_grad_()
        # for key in self.layers:
        #     if key[:2] == 'af':
        #         x = checkpoint.checkpoint(
        #             self.run_fn(key), x, snr)
        #     else:
        #         x = checkpoint.checkpoint(
        #             self.run_fn(key), x)

        B = x.size(0)
        x = x.view((B, *self.feat_dims))
        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr)
            else:
                x = self.layers[key](x)
        return x

    def __str__(self):
        return f'FeatureDecoderAF({self.c_in},{self.c_feat},{self.c_out})'
