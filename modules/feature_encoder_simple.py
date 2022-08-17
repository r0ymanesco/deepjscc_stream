import torch.nn as nn

from compressai.layers import GDN


class FeatureEncoderSimple(nn.Module):
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

            GDN(
                in_channels=self.c_feat
            ),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            GDN(
                in_channels=self.c_feat
            ),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            GDN(
                in_channels=self.c_feat
            ),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_out,
                kernel_size=5,
                stride=2,
                padding=2
            )
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

            GDN(
                in_channels=self.c_feat
            ),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            GDN(
                in_channels=self.c_feat
            ),

            nn.Conv2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            GDN(
                in_channels=self.c_feat
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
        return f'FeatureEncoderSimple({self.c_in},{self.c_feat},{self.c_out},{self.reduced})'


class FeatureDecoderSimple(nn.Module):
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
            nn.ConvTranspose2d(
                in_channels=self.c_in,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            GDN(
                in_channels=self.c_feat,
                inverse=True
            ),

            nn.ConvTranspose2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            GDN(
                in_channels=self.c_feat,
                inverse=True
            ),

            nn.ConvTranspose2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            GDN(
                in_channels=self.c_feat,
                inverse=True
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

            GDN(
                in_channels=self.c_feat,
                inverse=True
            ),

            nn.ConvTranspose2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1
            ),

            GDN(
                in_channels=self.c_feat,
                inverse=True
            ),

            nn.ConvTranspose2d(
                in_channels=self.c_feat,
                out_channels=self.c_feat,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            GDN(
                in_channels=self.c_feat,
                inverse=True
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
        return f'FeatureDecoderSimple({self.c_in},{self.c_feat},{self.c_out},{self.reduced})'
