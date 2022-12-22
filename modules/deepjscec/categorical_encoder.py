import ipdb

import torch.nn as nn

from compressai.layers import AttentionBlock
from compressai.layers import ResidualBlock, ResidualBlockWithStride


class CategoricalEncoder(nn.Module):
    def __init__(self, c_in, c_feat, c_out, reduced, classes):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out
        self.classes = classes
        self.reduced = reduced

        if reduced:
            self._reduced_arch(c_in, c_feat, c_out)
        else:
            self._regular_arch(c_in, c_feat, c_out)

    def _regular_arch(self, c_in, c_feat, c_out):
        self.encoder = nn.Sequential(
            ResidualBlockWithStride(
                in_ch=c_in,
                out_ch=c_feat,
                stride=2),

            ResidualBlockWithStride(
                in_ch=c_feat,
                out_ch=c_feat,
                stride=2),

            AttentionBlock(c_feat),

            ResidualBlockWithStride(
                in_ch=c_feat,
                out_ch=c_feat,
                stride=2),

            ResidualBlockWithStride(
                in_ch=c_feat,
                out_ch=c_out,
                stride=2),

            AttentionBlock(c_out),
        )

    def _reduced_arch(self, c_in, c_feat, c_out):
        self.encoder = nn.Sequential(
            ResidualBlockWithStride(
                in_ch=c_in,
                out_ch=c_feat,
                stride=2),

            ResidualBlock(
                in_ch=c_feat,
                out_ch=c_feat),

            AttentionBlock(c_feat),

            ResidualBlockWithStride(
                in_ch=c_feat,
                out_ch=c_feat,
                stride=2),

            ResidualBlock(
                in_ch=c_feat,
                out_ch=c_out),

            AttentionBlock(c_out),
        )

    def forward(self, x):
        return self.encoder(x)

    def __str__(self):
        return f'CategoricalEncoder({self.c_in},{self.c_feat},{self.c_out},{self.reduced})'


class CategoricalDecoder(nn.Module):
    def __init__(self, c_out, c_feat, feat_dims, reduced, classes):
        super().__init__()
        self.c_out = c_out
        self.c_feat = c_feat
        self.feat_dims = feat_dims
        self.classes = classes
        self.reduced = reduced

        if reduced:
            self._reduced_arch(c_out, c_feat, classes)
        else:
            self._regular_arch(c_out, c_feat, classes)

    def _regular_arch(self, c_out, c_feat, classes):
        self.jscc_decoder = nn.Sequential(
            ResidualBlockWithStride(
                in_ch=c_out,
                out_ch=c_feat,
                stride=2),

            ResidualBlock(
                in_ch=c_feat,
                out_ch=2*c_feat),

            ResidualBlockWithStride(
                in_ch=2*c_feat,
                out_ch=4*c_feat,
                stride=2),

            AttentionBlock(4*c_feat),

            ResidualBlock(
                in_ch=4*c_feat,
                out_ch=8*c_feat),

            ResidualBlockWithStride(
                in_ch=8*c_feat,
                out_ch=16*c_feat,
                stride=2),

            ResidualBlock(
                in_ch=16*c_feat,
                out_ch=16*c_feat),

            AttentionBlock(16*c_feat),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*c_feat, 16*c_feat),
            nn.LeakyReLU(),
            nn.Linear(16*c_feat, classes),
        )

    def _reduced_arch(self, c_out, c_feat, classes):
        self.jscc_decoder = nn.Sequential(
            ResidualBlock(
                in_ch=c_out,
                out_ch=c_feat),

            ResidualBlock(
                in_ch=c_feat,
                out_ch=2*c_feat),

            AttentionBlock(2*c_feat),

            ResidualBlock(
                in_ch=2*c_feat,
                out_ch=2*c_feat),

            ResidualBlockWithStride(
                in_ch=2*c_feat,
                out_ch=4*c_feat,
                stride=2),

            ResidualBlock(
                in_ch=4*c_feat,
                out_ch=8*c_feat),

            ResidualBlock(
                in_ch=8*c_feat,
                out_ch=8*c_feat),

            AttentionBlock(8*c_feat),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(8*c_feat, 8*c_feat),
            nn.LeakyReLU(),
            nn.Linear(8*c_feat, classes),
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view((B, *self.feat_dims))
        return self.classifier(self.jscc_decoder(x).view(B, -1))

    def __str__(self):
        return f'CategoricalEncoder({self.c_feat},{self.c_out},{self.reduced})'
