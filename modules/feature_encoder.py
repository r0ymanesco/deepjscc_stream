import torch.nn as nn

from compressai.layers import ResidualBlockUpsample, AttentionBlock
from compressai.layers import ResidualBlock, ResidualBlockWithStride


class FeatureEncoder(nn.Module):
    def __init__(self, c_in, c_feat, c_out, reduced):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out
        self.reduced = reduced

        self._get_arch(reduced)

    @staticmethod
    def get_config(reduced):
        match reduced:
            case True:
                down_factor = 4
            case False:
                down_factor = 16
            case _:
                raise ValueError
        return down_factor

    def _get_arch(self, reduced):
        match reduced:
            case True:
                self._reduced_arch()
            case False:
                self._regular_arch()
            case _:
                raise ValueError

    def _regular_arch(self):
        self.layers = nn.Sequential(
            ResidualBlockWithStride(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                stride=2),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                stride=2),

            AttentionBlock(self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                stride=2),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                stride=2),

            AttentionBlock(self.c_out),
        )

    def _reduced_arch(self):
        self.layers = nn.Sequential(
            ResidualBlockWithStride(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                stride=2),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            AttentionBlock(self.c_feat),

            ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                stride=2),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),

            AttentionBlock(self.c_out),
        )

    def forward(self, x):
        return self.layers(x)

    def __str__(self):
        return f'FeatureEncoder({self.c_in},{self.c_feat},{self.c_out},{self.reduced})'


class FeatureDecoder(nn.Module):
    def __init__(self, c_in, c_feat, c_out, feat_dims, reduced):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out
        self.feat_dims = feat_dims
        self.reduced = reduced

        self._get_arch(reduced)

    def _get_arch(self, reduced):
        match reduced:
            case True:
                self._reduced_arch()
            case False:
                self._regular_arch()

    def _regular_arch(self):
        self.layers = nn.Sequential(
            AttentionBlock(self.c_in),

            ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                upsample=2),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                upsample=2),

            AttentionBlock(self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                upsample=2),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                upsample=2),
        )

    def _reduced_arch(self):
        self.layers = nn.Sequential(
            AttentionBlock(self.c_in),

            ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                upsample=2),

            AttentionBlock(self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                upsample=2),
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view((B, *self.feat_dims))
        return self.layers(x)

    def __str__(self):
        return f'FeatureDecoder({self.c_in},{self.c_feat},{self.c_out},{self.reduced})'
