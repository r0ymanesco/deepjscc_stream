import ipdb

import torch
import torch.nn as nn

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

        self.codeword_query = nn.parameter.Parameter(torch.rand(1, 1, *feat_dims))

    def _feature_train(self, gop):
        frames = torch.cat(gop, dim=0)
        z_0 = self.feature_encoder(frames)
        return z_0, {}

    def _joint_train(self, gop):
        B = gop[0].size(0)

        batch_frames = torch.cat(gop, dim=0)
        batch_y = self.feature_encoder(batch_frames)
        batch_y = torch.chunk(batch_y, chunks=len(gop), dim=0)

        enc_input = torch.stack(batch_y, dim=1)
        enc_output = self.tf_encoder(enc_input, None)

        batch_codeword_query = self.codeword_query.tile((B, 1, 1, 1, 1))
        codeword = self.tf_decoder(batch_codeword_query, enc_output, src_mask=None, trg_mask=None)
        return codeword, {}

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
        # self.auto_feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)

        # self.tf_encoder = TFEncoder2D(c_out, tf_layers, tf_heads, tf_ff, tf_dropout,
        #                               feat_dims, c_feat, max_seq_len)
        self.tf_decoder = TFDecoder2D(c_out, tf_layers, tf_heads, tf_ff, tf_dropout,
                                      feat_dims, c_feat, max_seq_len)

        self.codeword_query = nn.parameter.Parameter(torch.rand(1, 1, *feat_dims))

    def _feature_train(self, codes, gop_len):
        B = codes.size(0)
        z_0 = codes.view((B, *self.feat_dims))
        frames = self.feature_decoder(z_0)
        gop = torch.chunk(frames, chunks=gop_len, dim=0)
        return gop, {}

    def _joint_train(self, codeword, gop_len):
        B = codeword.size(0)
        codeword = codeword.view((B, *self.feat_dims)).unsqueeze(1)
        gop = []

        codeword_query_start = self.codeword_query.tile((B, 1, 1, 1, 1))
        trg_list = [codeword_query_start]
        for i in range(gop_len):
            target = torch.cat(trg_list, dim=1)
            trg_mask = torch.repeat_interleave(
                get_mask(len(trg_list), len(trg_list), self.tf_heads, self.feat_dims[1:]), B, dim=0)
            tf_output = self.tf_decoder(target, codeword,
                                        src_mask=None,
                                        trg_mask=trg_mask.to(target.device))

            y_next = torch.chunk(tf_output, chunks=i+1, dim=1)[-1]
            trg_list.append(y_next)

            frame = self.feature_decoder(y_next.squeeze(1))
            gop.append(torch.sigmoid(frame))

        assert len(gop) == gop_len
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
