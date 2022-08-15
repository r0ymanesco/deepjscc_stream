import ipdb
import numpy as np

import torch
import torch.nn as nn

from modules.transformer import TFEncoder
from modules.transformer2d import TFEncoder2D, TFDecoder2D
from modules.feature_encoder import FeatureEncoder, FeatureDecoder


def get_mask(q_len, k_len, n_heads, feat_dims):
    idxs = torch.tril_indices(q_len, k_len)
    mask = torch.zeros(1, q_len, k_len, n_heads, *feat_dims)
    mask[:, idxs[0], idxs[1]] = 1.
    return mask


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
        self.c_out = c_out
        self.feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)

        self.tf_encoder = TFEncoder2D(c_out, tf_layers, tf_heads, tf_ff, tf_dropout,
                                      feat_dims, c_feat, max_seq_len)
        self.tf_decoder = TFDecoder2D(c_out, tf_layers, tf_heads, tf_ff, tf_dropout,
                                      feat_dims, c_feat, max_seq_len)

    def _feature_train(self, gop):
        frames = torch.cat(gop, dim=0)
        z_0 = self.feature_encoder(frames)
        return (z_0, torch.ones_like(z_0)), {}

    def _joint_train(self, gop):
        key_frame = gop[0]
        int_frames = gop[1:]
        B = key_frame.size(0)

        z_0 = self.feature_encoder(key_frame)
        seq_list = [z_0] + [None] * len(int_frames)

        batch_int_frames = torch.cat(int_frames, dim=0)
        batch_z_int = self.feature_encoder(batch_int_frames)
        batch_z_int = torch.chunk(batch_z_int, chunks=len(int_frames), dim=0)

        # enc_mask = torch.repeat_interleave(
        #     get_mask(len(int_frames), len(int_frames), self.tf_heads, self.feat_dims[1:]), B, dim=0)
        enc_input = torch.stack(batch_z_int, dim=1)
        enc_output = self.tf_encoder(enc_input, None)# enc_mask.to(enc_input.device))

        for i in range(len(int_frames)):
            # trg_mask = torch.repeat_interleave(
            #     get_mask(i+1, i+1, self.tf_heads, self.feat_dims[1:]), B, dim=0)
            # src_mask = torch.repeat_interleave(
            #     get_mask(i+1, len(int_frames), self.tf_heads, self.feat_dims[1:]), B, dim=0)
            target = torch.stack(seq_list[:i+1], dim=1)
            tf_output = self.tf_decoder(target, enc_output,
                                        src_mask=None, #src_mask.to(enc_output.device),
                                        trg_mask=None) #trg_mask.to(tf_input.device))
            z_next = torch.chunk(tf_output, chunks=i+1, dim=1)[-1]
            seq_list[i+1] = z_next.squeeze(1)

        z_int = seq_list[-1]
        assert None not in seq_list
        return (z_0, z_int), {}

    def forward(self, gop, stage):
        match stage:
            case 'feature':
                return self._feature_train(gop)
            case 'joint':
                return self._joint_train(gop)
            case _:
                raise ValueError

    def __str__(self):
        return f'TFRecursiveEncoder({self.c_in},{self.c_feat},{self.c_out},{self.tf_layers},{self.tf_heads},{self.tf_ff},{self.tf_dropout})'


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
        self.c_out = c_out
        self.feature_decoder = FeatureDecoder(c_out, c_feat, c_in, reduced)
        self.auto_feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)

        self.tf_encoder = TFEncoder2D(c_out, tf_layers, tf_heads, tf_ff, tf_dropout,
                                      feat_dims, c_feat, max_seq_len)
        self.tf_decoder = TFDecoder2D(c_out, tf_layers, tf_heads, tf_ff, tf_dropout,
                                      feat_dims, c_feat, max_seq_len)

    def _feature_train(self, codes, gop_len):
        z_0, _ = codes
        B = z_0.size(0)
        z_0 = z_0.view((B, *self.feat_dims))
        frames = self.feature_decoder(z_0)
        gop = torch.chunk(frames, chunks=gop_len, dim=0)
        return gop, {}

    def _joint_train(self, codes, gop_len):
        z_0, z_int = codes
        gop = [None] * gop_len

        B = z_0.size(0)
        z_0 = z_0.view((B, *self.feat_dims))
        z_int = z_int.view((B, *self.feat_dims))

        key_frame = self.feature_decoder(z_0)
        gop[0] = torch.sigmoid(key_frame)

        trg_list = [None] * (gop_len-1)
        seq_list = [z_int] + [None] * (gop_len-1)
        for i in range(gop_len-1):
            seq_input = torch.stack(seq_list[:i+1], dim=1)
            # seq_mask = torch.repeat_interleave(
            #     get_mask(len(seq_list), len(seq_list), self.tf_heads, self.feat_dims[1:]), B, dim=0)
            enc_output = self.tf_encoder(seq_input, None) # seq_mask.to(seq_input.device))

            z_bar = self.auto_feature_encoder(gop[i])
            trg_list[i] = z_bar
            target = torch.stack(trg_list[:i+1], dim=1)
            # trg_mask = torch.repeat_interleave(
            #     get_mask(len(trg_list), len(trg_list), self.tf_heads, self.feat_dims[1:]), B, dim=0)
            tf_output = self.tf_decoder(target, enc_output,
                                        src_mask=None, #seq_mask.to(enc_output.device),
                                        trg_mask=None) #trg_mask.to(target.device))

            z_next = torch.chunk(tf_output, chunks=i+1, dim=1)[-1]
            seq_list[i+1] = z_next.squeeze(1)

            frame = self.feature_decoder(z_next.squeeze(1))
            gop[i+1] = torch.sigmoid(frame)

        assert None not in gop
        assert None not in seq_list
        assert None not in trg_list
        return gop, {}

    def forward(self, codes, gop_len, stage):
        match stage:
            case 'feature':
                return self._feature_train(codes, gop_len)
            case 'joint':
                return self._joint_train(codes, gop_len)
            case _:
                raise ValueError

    def __str__(self):
        return f'TFRecursiveDecoder({self.c_in},{self.c_feat},{self.c_out},{self.tf_layers},{self.tf_heads},{self.tf_ff},{self.tf_dropout})'
