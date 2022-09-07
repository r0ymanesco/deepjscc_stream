import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.feature_encoder_simple import FeatureEncoderSimple, FeatureDecoderSimple
from modules.feature_encoder import FeatureEncoder, FeatureDecoder


def get_mask(q_len, k_len, dims):
    block = torch.ones(dims, dims)
    loc = torch.ones(q_len, k_len)
    loc = torch.tril(loc, diagonal=0)
    mask = ~torch.kron(loc, block).to(torch.bool)
    return mask


def get_pad(H, W, c_win):
    W_r = W % c_win
    W_pad = [W_r // 2, W_r //2]
    if W_r % 2 != 0: W_pad[0] -= 1

    H_r = H % c_win
    H_pad = [H_r // 2, H_r // 2]
    if H_r % 2 != 0: H_pad[0] -= 1

    assert (W + sum(W_pad)) % c_win == 0
    assert (H + sum(H_pad)) % c_win == 0
    return H_pad, W_pad


def get_src_tokens(z_int, padding, p_win, p_stride):
    z_int_pad = F.pad(z_int, padding)

    z_src = z_int_pad.unfold(2, p_win, p_stride).unfold(3, p_win, p_stride)
    # (B, C, n_patches_h, n_patches_w, p_win, p_win)
    z_src = z_src.permute(0, 2, 3, 4, 5, 1)
    # (B, n_patches_h, n_patches_w, p_win, p_win, C)
    return z_src.contiguous()


def get_trg_tokens(z_0, padding, c_win):
    z_pad = F.pad(z_0, padding)

    z_trg = z_pad.unfold(2, c_win, c_win).unfold(3, c_win, c_win)
    # (B, C, n_patches_h, n_patches_w, c_win, c_win)
    z_trg = z_trg.permute(0, 2, 3, 4, 5, 1)
    # (B, n_patches_h, n_patches_w, c_win, c_win, C)
    return z_trg.contiguous()


def restore_shape(patches, output_shape, stride):
    # NOTE assumes patches in (B, n_patches_h, n_patches_w, c_win, c_win, C)
    patches = patches.permute(0, 5, 1, 2, 3, 4)
    B, C, _, _, win, _ = patches.size()
    patches = patches.contiguous().view(B, C, -1, win*win)
    patches = patches.permute(0, 1, 3, 2)
    patches = patches.contiguous().view(B, C*win*win, -1)
    restored = F.fold(
        patches, output_size=output_shape, kernel_size=win, stride=stride
    )
    return restored.contiguous()


class VCTRecursiveEncoder(nn.Module):
    def __init__(self, c_in, c_feat, feat_dims, reduced, c_win, p_win,
                 tf_layers, tf_heads, tf_ff, tf_dropout=0.):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.feat_dims = feat_dims
        self.reduced = reduced

        self.c_win = c_win
        self.p_win = p_win

        self.tf_layers = tf_layers
        self.tf_heads = tf_heads
        self.tf_ff = tf_ff
        self.tf_dropout = tf_dropout

        self.trg_h_pad, self.trg_w_pad = get_pad(self.feat_dims[1], self.feat_dims[2], c_win)
        self.src_h_pad = [pad + ((p_win - c_win) // 2) for pad in self.trg_h_pad]
        self.src_w_pad = [pad + ((p_win - c_win) // 2) for pad in self.trg_w_pad]
        n_patches_h = (self.feat_dims[1] + sum(self.trg_h_pad)) // c_win
        n_patches_w = (self.feat_dims[2] + sum(self.trg_w_pad)) // c_win

        c_out = feat_dims[0]
        self.c_out = c_out
        self.feature_encoder = FeatureEncoderSimple(c_in, c_feat, c_out, reduced)
        # self.feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)

        tf_encoder_layer = nn.TransformerEncoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_encoder_sep = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[0])
        self.tf_encoder_joint = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[1])

        tf_decoder_layer = nn.TransformerDecoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_decoder = nn.TransformerDecoder(tf_decoder_layer, num_layers=tf_layers[2])

        self.codeword_query = nn.parameter.Parameter(torch.rand(n_patches_h*n_patches_w, c_win*c_win, c_out))

    def _feature_train(self, gop):
        frames = torch.cat(gop, dim=0)
        z_0 = self.feature_encoder(frames)
        return z_0, {}

    def _joint_train(self, gop):
        B = gop[0].size(0)

        # TODO do channel emulation
        # the frames in batch_int_frames should be expected reconstruction
        batch_int_frames = torch.cat(gop, dim=0)
        batch_y = self.feature_encoder(batch_int_frames)
        y_tokens = get_src_tokens(batch_y, (*self.src_w_pad, *self.src_h_pad), self.p_win, self.c_win)
        # (B, n_patches_h, n_patches_w, p_win, p_win, C)
        n_patches_h, n_patches_w = y_tokens.size(1), y_tokens.size(2)

        sep_input = y_tokens.view(B*len(gop)*n_patches_h*n_patches_w, -1, self.c_out)
        sep_output = self.tf_encoder_sep(sep_input)

        joint_input = sep_output.view(B, len(gop), n_patches_h, n_patches_w, -1, self.c_out)
        joint_input = joint_input.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, n_patches_h, n_patches_w, len(gop), -1, c_out)
        joint_input = joint_input.view(B*n_patches_h*n_patches_w, -1, self.c_out)
        joint_output = self.tf_encoder_joint(joint_input)

        batch_codeword_query = self.codeword_query.tile((B, 1, 1))
        codeword = self.tf_decoder(batch_codeword_query, joint_output)

        H_out, W_out = (self.feat_dims[1] + sum(self.trg_h_pad)), (self.feat_dims[2] + sum(self.trg_w_pad))
        codeword = codeword.view(B, n_patches_h, n_patches_w, self.c_win, self.c_win, self.c_out)
        codeword = restore_shape(codeword, (H_out, W_out), self.c_win)
        codeword = torch.split(codeword, [self.feat_dims[1], H_out - self.feat_dims[1]], dim=2)[0]
        codeword = torch.split(codeword, [self.feat_dims[2], W_out - self.feat_dims[2]], dim=3)[0]
        return codeword.contiguous(), {}

    def forward(self, gop, stage):
        match stage:
            case 'feature':
                return self._feature_train(gop)
            case 'joint':
                return self._joint_train(gop)
            case _:
                raise ValueError

    def __str__(self):
        return f'VCTReEncoder({self.c_in},{self.c_feat},{self.c_out},{self.tf_layers},{self.tf_heads},{self.tf_ff},{self.tf_dropout})'


class VCTRecursiveDecoder(nn.Module):
    def __init__(self, c_in, c_feat, feat_dims, reduced, c_win, p_win,
                 tf_layers, tf_heads, tf_ff, tf_dropout=0.):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.feat_dims = feat_dims

        self.c_win = c_win
        self.p_win = p_win

        self.tf_layers = tf_layers
        self.tf_heads = tf_heads
        self.tf_ff = tf_ff
        self.tf_dropout = tf_dropout

        self.trg_h_pad, self.trg_w_pad = get_pad(self.feat_dims[1], self.feat_dims[2], c_win)
        self.src_h_pad = [pad + ((p_win - c_win) // 2) for pad in self.trg_h_pad]
        self.src_w_pad = [pad + ((p_win - c_win) // 2) for pad in self.trg_w_pad]
        n_patches_h = (self.feat_dims[1] + sum(self.trg_h_pad)) // c_win
        n_patches_w = (self.feat_dims[2] + sum(self.trg_w_pad)) // c_win

        c_out = feat_dims[0]
        self.c_out = c_out
        self.feature_decoder = FeatureDecoderSimple(c_out, c_feat, c_in, reduced)
        # self.auto_feature_encoder = FeatureEncoderSimple(c_in, c_feat, c_out, reduced)
        # self.feature_decoder = FeatureDecoder(c_out, c_feat, c_in, reduced)
        # self.auto_feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)

        # tf_encoder_layer = nn.TransformerEncoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        # self.tf_encoder_sep = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[0])
        # self.tf_encoder_joint = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[1])

        tf_decoder_layer = nn.TransformerDecoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_decoder = nn.TransformerDecoder(tf_decoder_layer, num_layers=tf_layers[2])

        self.codeword_query = nn.parameter.Parameter(torch.rand(n_patches_h*n_patches_w, c_win*c_win, c_out))

    def _feature_train(self, codes, gop_len):
        z_0 = codes
        B = z_0.size(0)
        z_0 = z_0.view((B, *self.feat_dims))
        frames = self.feature_decoder(z_0)
        gop = torch.chunk(frames, chunks=gop_len, dim=0)
        return gop, {}

    def _joint_train(self, codeword, gop_len):
        B = codeword.size(0)
        H_out, W_out = (self.feat_dims[1] + sum(self.trg_h_pad)), (self.feat_dims[2] + sum(self.trg_w_pad))
        gop = []

        src_codeword = codeword.view((B, *self.feat_dims))
        y_tokens = get_trg_tokens(src_codeword, (*self.trg_w_pad, *self.trg_h_pad), self.c_win)
        n_patches_h, n_patches_w = y_tokens.size(1), y_tokens.size(2)
        src_tokens = y_tokens.view(B*n_patches_h*n_patches_w, -1, self.c_out)

        start_query = self.codeword_query.tile((B, 1, 1))
        trg_list = [start_query]
        for i in range(gop_len):
            target = torch.cat(trg_list, dim=1)
            trg_mask = get_mask(len(trg_list), len(trg_list), self.c_win**2).to(target.device)
            tf_output = self.tf_decoder(target, src_tokens, tgt_mask=trg_mask)

            y_next = torch.chunk(tf_output, chunks=i+1, dim=1)[-1]
            trg_list.append(y_next)

            y_feat = y_next.view(B, n_patches_h, n_patches_w, self.c_win, self.c_win, self.c_out)
            y_feat = restore_shape(y_feat, (H_out, W_out), self.c_win)
            y_feat = torch.split(y_feat, [self.feat_dims[1], H_out - self.feat_dims[1]], dim=2)[0]
            y_feat = torch.split(y_feat, [self.feat_dims[2], W_out - self.feat_dims[2]], dim=3)[0]
            frame = self.feature_decoder(y_feat)
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
        return f'VCTReDecoder({self.c_in},{self.c_feat},{self.c_out},{self.tf_layers},{self.tf_heads},{self.tf_ff},{self.tf_dropout})'
