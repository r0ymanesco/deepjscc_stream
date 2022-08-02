import ipdb
import numpy as np

import torch
import torch.nn as nn

from modules.transformer import TFEncoder
from modules.transformer2d import TFEncoder2D
from modules.feature_encoder import FeatureEncoder, FeatureDecoder


class TFRecursiveEncoder(nn.Module):
    def __init__(self, c_in, c_feat, feat_dims, reduced,
                 tf_layers, tf_heads, tf_ff, max_seq_len, tf_dropout=0.):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.feat_dims = feat_dims
        self.reduced = reduced

        self.tf_layers = tf_layers
        self.tf_heads = tf_heads
        self.tf_ff = tf_ff
        self.tf_dropout = tf_dropout

        c_out = feat_dims[0]
        self.key_feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)
        self.int_feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)

        # d_feat = np.prod(feat_dims)
        # self.tf_encoder = TFEncoder(d_feat, tf_layers, tf_heads, tf_ff, max_seq_len, tf_dropout)
        self.tf_encoder = TFEncoder2D(c_out*2, tf_layers, tf_heads, tf_ff, tf_dropout,
                                      feat_dims, c_feat, max_seq_len)
        # self.tf_decoder = TFDecoder(d_feat, tf_layers, tf_heads, tf_ff, tf_dropout)

    def forward(self, gop):
        key_frame = gop[0]
        int_frames = gop[1:]

        z_0 = self.key_feature_encoder(key_frame)
        seq_list = [z_0]

        for i, frame in enumerate(int_frames):
            z = self.int_feature_encoder(frame)
            z_prev = seq_list[i]
            seq_list[i] = torch.cat((z_prev, z), dim=1)

            tf_input = torch.stack(seq_list, dim=1)
            mask = torch.tril(torch.ones(i+1, i+1, device=z.device))
            # NOTE mask does nothing in TF2D
            tf_output = self.tf_encoder(tf_input, mask)
            z_next = torch.chunk(tf_output, chunks=i+1, dim=1)[-1]  # NOTE: entropy est this val
            z_next = torch.chunk(z_next, chunks=2, dim=2)[-1]
            seq_list.append(z_next.squeeze(1))

        z_int = seq_list[-1]
        return (z_0, z_int), {}

    def __str__(self):
        return f'TFRecursiveEncoder({self.c_in},{self.c_feat},\
        {self.tf_layers},{self.tf_heads},{self.tf_ff},{self.tf_dropout})'


class TFRecursiveDecoder(nn.Module):
    def __init__(self, c_in, c_feat, feat_dims, reduced,
                 tf_layers, tf_heads, tf_ff, max_seq_len, tf_dropout=0.):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.feat_dims = feat_dims

        self.tf_layers = tf_layers
        self.tf_heads = tf_heads
        self.tf_ff = tf_ff
        self.tf_dropout = tf_dropout

        c_out = feat_dims[0]
        self.key_feature_decoder = FeatureDecoder(c_out, c_feat, c_in, reduced)
        self.int_feature_decoder = FeatureDecoder(c_out, c_feat, c_in, reduced)
        self.auto_feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)

        # d_feat = np.prod(feat_dims)
        # self.tf_encoder = TFEncoder(d_feat, tf_layers, tf_heads, tf_ff, max_seq_len, tf_dropout)
        self.tf_encoder = TFEncoder2D(c_out*2, tf_layers, tf_heads, tf_ff, tf_dropout,
                                      feat_dims, c_feat, max_seq_len)
        # self.tf_decoder = TFDecoder(d_feat, tf_layers, tf_heads, tf_ff, tf_dropout)

    def forward(self, codes, gop_len):
        gop = []
        z_0, z_int = codes

        B = z_0.size(0)
        z_0 = z_0.view((B, *self.feat_dims))
        z_int = z_int.view((B, *self.feat_dims))
        seq_list = [z_int]

        key_frame = self.key_feature_decoder(z_0)
        gop.append(key_frame)

        for i in range(gop_len-1):
            z_prev = seq_list[i]
            z_bar = self.auto_feature_encoder(gop[i])
            seq_list[i] = torch.cat((z_prev, z_bar), dim=1)

            tf_input = torch.stack(seq_list, dim=1)
            mask = torch.tril(torch.ones(i+1, i+1, device=z_0.device))
            tf_output = self.tf_encoder(tf_input, mask)
            z_next = torch.chunk(tf_output, chunks=i+1, dim=1)[-1]
            z_next = torch.chunk(z_next, chunks=2, dim=2)[-1]
            seq_list.append(z_next.squeeze(1))

            frame = self.int_feature_decoder(z_next.squeeze(1))
            gop.append(frame)

        return gop, {}

    def __str__(self):
        return f'TFRecursiveDecoder({self.c_in},{self.c_feat},\
        {self.tf_layers},{self.tf_heads},{self.tf_ff},{self.tf_dropout})'
