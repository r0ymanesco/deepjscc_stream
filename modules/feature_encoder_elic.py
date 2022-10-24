from copy import deepcopy

import torch.nn as nn

from compressai.layers import AttentionBlock


class ResidualBottleneck(nn.Module):
    def __init__(self, c_in, c_feat=64, n_blocks=3):
        super().__init__()

        layers = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_feat,
                kernel_size=1,
                stride=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=c_feat,
                out_channels=c_feat,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=c_feat,
                out_channels=c_in,
                kernel_size=1,
                stride=1,
            ),
        )

        self.blocks = nn.ModuleList([deepcopy(layers) for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x
        return x


class FeatureEncoderELIC(nn.Module):
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
            nn.Conv2d(
                in_channels=self.c_in,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
            ),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
            ),

            AttentionBlock(self.c_feat),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
            ),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_out,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            AttentionBlock(self.c_out),
        )

    def _reduced_arch(self):
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=self.c_in,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
                n_blocks=1,
            ),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
                n_blocks=1,
            ),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
                n_blocks=1,
            ),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_out,
                kernel_size=5,
                stride=1,
                padding=2
            )
        )

    def forward(self, x):
        return self.layers(x)

    def __str__(self):
        return f'FeatureEncoderELIC({self.c_in},{self.c_feat},{self.c_out},{self.reduced})'


class FeatureDecoderELIC(nn.Module):
    def __init__(self, c_in, c_feat, c_out, reduced):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out
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

            nn.ConvTranspose2d(
                in_channels=self.c_in,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
            ),

            nn.ConvTranspose2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            AttentionBlock(self.c_feat),

            ResidualBottleneck(
                c_in=self.c_feat,
            ),

            nn.ConvTranspose2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
            ),

            nn.ConvTranspose2d(
                in_channels=self.c_feat,
                out_channels=self.c_out,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            )
        )

    def _reduced_arch(self):
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.c_in,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
                n_blocks=1,
            ),

            nn.ConvTranspose2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
                n_blocks=1,
            ),

            nn.ConvTranspose2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            ResidualBottleneck(
                c_in=self.c_feat,
                n_blocks=1,
            ),

            nn.ConvTranspose2d(
                in_channels=self.c_feat,
                out_channels=self.c_out,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            )
        )

    def forward(self, x):
        return self.layers(x)

    def __str__(self):
        return f'FeatureDecoderELIC({self.c_in},{self.c_feat},{self.c_out},{self.reduced})'
