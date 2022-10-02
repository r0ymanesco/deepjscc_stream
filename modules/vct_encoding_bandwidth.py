import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from modules.feature_encoder_simple import FeatureEncoderSimple, FeatureDecoderSimple
from modules.feature_encoder import FeatureEncoder, FeatureDecoder


def get_block_mask(q_len, k_len, dims):
    block = torch.ones(dims, dims)
    loc = torch.ones(q_len, k_len)
    loc = torch.tril(loc, diagonal=0)
    mask = ~torch.kron(loc, block).to(torch.bool)
    return mask

def get_mask(q_len, k_len):
    loc = torch.ones(q_len, k_len)
    mask = torch.triu(loc, diagonal=1).to(torch.bool)
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


class VCTEncoderBandwidth(nn.Module):
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
        # n_patches_h = (self.feat_dims[1] + sum(self.trg_h_pad)) // c_win
        # n_patches_w = (self.feat_dims[2] + sum(self.trg_w_pad)) // c_win

        c_out = feat_dims[0]
        self.c_out = c_out
        self.feature_encoder = FeatureEncoderSimple(c_in, c_feat, c_out, reduced)
        # self.feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)

        tf_encoder_layer = nn.TransformerEncoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_encoder_sep = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[0])
        self.tf_encoder_joint = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[1])

        tf_decoder_layer = nn.TransformerDecoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_decoder = nn.TransformerDecoder(tf_decoder_layer, num_layers=tf_layers[2])

        # self.distribution_mapper = nn.Linear(c_out, 2*c_out)

    # def _compute_likelihood(self, decoder_in, decoder_out):
    #     ipdb.set_trace()
    #     B, seq_len, _ = decoder_out.shape
    #     decoder_in = decoder_in.view(-1)

    #     distribution_params = self.distribution_mapper(decoder_out)
    #     dist_mean, dist_std = torch.chunk(distribution_params, chunks=2, dim=-1)
    #     dist_var = dist_std.pow(2)
    #     dist_std = torch.sqrt(dist_var)

    #     dist_mean = dist_mean.view(-1)
    #     dist_std = torch.diag(dist_std.view(-1))
    #     distribution = Normal(dist_mean, dist_std)

    #     likelihood = distribution.cdf(decoder_in + 0.5) - distribution.cdf(decoder_in - 0.5)
    #     likelihood = likelihood.view(B, seq_len, -1)
    #     return likelihood

    def _feature_train(self, gop):
        frames = torch.cat(gop, dim=0)
        z_0 = self.feature_encoder(frames)
        return z_0, {}

    def _coding_train(self, gop):
        B = gop[0].size(0)
        n_prev_tokens = len(gop) - 1
        H_out, W_out = (self.feat_dims[1] + sum(self.trg_h_pad)), (self.feat_dims[2] + sum(self.trg_w_pad))

        batch_int_frames = torch.cat(gop, dim=0)
        batch_y = self.feature_encoder(batch_int_frames)
        prev_y, int_y = torch.split(batch_y, [B*n_prev_tokens, B], dim=0)
        # FIXME do channel emulation for reference frames

        prev_tokens = get_src_tokens(prev_y, (*self.src_w_pad, *self.src_h_pad), self.p_win, self.c_win)
        # (B*n_prev_tokens, n_patches_h, n_patches_w, p_win, p_win, C)

        int_tokens = get_trg_tokens(int_y, (*self.trg_w_pad, *self.trg_h_pad), self.c_win)
        # (B, n_patches_h, n_patches_w, c_win, c_win, C)
        n_patches_h, n_patches_w = int_tokens.size(1), int_tokens.size(2)
        int_tokens = int_tokens.view(B*n_patches_h*n_patches_w, -1, self.c_out)
        # (B*n_patches_h*n_patches_w, c_win*c_win, C)
        int_tokens = torch.chunk(int_tokens, chunks=self.c_win**2, dim=1)
        # [(B*n_patches_h*n_patches_w, 1, C)] * (c_win*c_win)

        sep_input = prev_tokens.view(B*n_prev_tokens*n_patches_h*n_patches_w, -1, self.c_out)
        sep_output = self.tf_encoder_sep(sep_input)

        joint_input = sep_output.view(B, n_prev_tokens, n_patches_h, n_patches_w, -1, self.c_out)
        joint_input = joint_input.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, n_patches_h, n_patches_w, n_prev_tokens, p_win*p_win, c_out)
        joint_input = joint_input.view(B*n_patches_h*n_patches_w, -1, self.c_out)
        # (B*n_patches_h*n_patches_w, n_prev_tokens*p_win*p_win, c_out)
        tf_encoder_out = self.tf_encoder_joint(joint_input)

        trg_in = []
        conditional_tokens = []
        for i, token in enumerate(int_tokens):
            trg_in.append(token)
            tf_decoder_in = torch.cat(trg_in, dim=1)
            trg_mask = get_mask(i+1, i+1).to(token.device)
            tf_decoder_out = self.tf_decoder(tf_decoder_in, tf_encoder_out, tgt_mask=trg_mask)
            # (B*n_patches_h*n_patches_w, i+1, C)

            out_token = torch.chunk(tf_decoder_out, chunks=i+1, dim=1)[-1]
            out_token = out_token.view(B, n_patches_h, n_patches_w, 1, -1)
            conditional_tokens.append(out_token)

        codeword = torch.cat(conditional_tokens, dim=3)
        # codeword = torch.cat(int_tokens, dim=1)
        codeword = codeword.view(B, n_patches_h, n_patches_w, self.c_win, self.c_win, self.c_out)
        codeword = restore_shape(codeword, (H_out, W_out), self.c_win)
        codeword = torch.split(codeword, [self.feat_dims[1], H_out - self.feat_dims[1]], dim=2)[0]
        codeword = torch.split(codeword, [self.feat_dims[2], W_out - self.feat_dims[2]], dim=3)[0]
        return codeword.contiguous(), {'conditional_tokens': conditional_tokens}

    def forward(self, gop, stage):
        match stage:
            # case 'feature':
            #     return self._feature_train(gop)
            case 'coding':
                return self._coding_train(gop)
            case 'prediction':
                return self._coding_train(gop)
            case _:
                raise ValueError

    def __str__(self):
        return f'VCTBWEncoder({self.c_in},{self.c_feat},{self.c_out},{self.tf_layers},{self.tf_heads},{self.tf_ff},{self.tf_dropout})'


class VCTDecoderBandwidth(nn.Module):
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
        # n_patches_h = (self.feat_dims[1] + sum(self.trg_h_pad)) // c_win
        # n_patches_w = (self.feat_dims[2] + sum(self.trg_w_pad)) // c_win

        c_out = feat_dims[0]
        self.c_out = c_out
        self.feature_decoder = FeatureDecoderSimple(c_out, c_feat, c_in, reduced)
        self.auto_feature_encoder = FeatureEncoderSimple(c_in, c_feat, c_out, reduced)
        # self.feature_decoder = FeatureDecoder(c_out, c_feat, c_in, reduced)
        # self.auto_feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)

        tf_encoder_layer = nn.TransformerEncoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_encoder_sep = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[0])
        self.tf_encoder_joint = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[1])

        tf_decoder_layer = nn.TransformerDecoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_decoder = nn.TransformerDecoder(tf_decoder_layer, num_layers=tf_layers[2])

    def _feature_train(self, codes, gop_len):
        z_0 = codes
        B = z_0.size(0)
        z_0 = z_0.view((B, *self.feat_dims))
        frames = self.feature_decoder(z_0)
        gop = torch.chunk(frames, chunks=gop_len, dim=0)
        return gop, {}

    def _coding_train(self, codeword, prev_frames):
        B = codeword.size(0)
        n_prev_tokens = len(prev_frames)
        H_out, W_out = (self.feat_dims[1] + sum(self.trg_h_pad)), (self.feat_dims[2] + sum(self.trg_w_pad))
        frame_at_rate = []

        codeword = codeword.view((B, *self.feat_dims))
        int_tokens = get_trg_tokens(codeword, (*self.trg_w_pad, *self.trg_h_pad), self.c_win)
        n_patches_h, n_patches_w = int_tokens.size(1), int_tokens.size(2)
        int_tokens = int_tokens.view(B*n_patches_h*n_patches_w, -1, self.c_out)
        # (B*n_patches_h*n_patches_w, c_win*c_win, C)
        int_tokens = torch.chunk(int_tokens, chunks=self.c_win**2, dim=1)
        # [(B*n_patches_h*n_patches_w, 1, C)] * (c_win*c_win)

        batch_prev_frames = torch.cat(prev_frames, dim=0)
        prev_y = self.auto_feature_encoder(batch_prev_frames)

        prev_tokens = get_src_tokens(prev_y, (*self.src_w_pad, *self.src_h_pad), self.p_win, self.c_win)
        # (B*n_prev_tokens, n_patches_h, n_patches_w, p_win, p_win, C)
        sep_input = prev_tokens.view(B*n_prev_tokens*n_patches_h*n_patches_w, -1, self.c_out)
        sep_output = self.tf_encoder_sep(sep_input)

        joint_input = sep_output.view(B, n_prev_tokens, n_patches_h, n_patches_w, -1, self.c_out)
        joint_input = joint_input.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, n_patches_h, n_patches_w, n_prev_tokens, p_win*p_win, c_out)
        joint_input = joint_input.view(B*n_patches_h*n_patches_w, -1, self.c_out)
        # (B*n_patches_h*n_patches_w, n_prev_tokens*p_win*p_win, c_out)
        tf_encoder_out = self.tf_encoder_joint(joint_input)

        trg_list = []
        decoded_tokens = []
        padding_tokens = [torch.zeros_like(int_tokens[0])] * len(int_tokens)
        for i, token in enumerate(int_tokens):
            trg_list.append(token)
            tf_decoder_in = torch.cat(trg_list, dim=1)
            trg_mask = get_mask(i+1, i+1).to(token.device)
            tf_decoder_out = self.tf_decoder(tf_decoder_in, tf_encoder_out, tgt_mask=trg_mask)
            # (B*n_patches_h*n_patches_w, i+1, C)
            y_next = torch.chunk(tf_decoder_out, chunks=i+1, dim=1)[-1]
            decoded_tokens.append(y_next)

            padded_tokens = decoded_tokens + padding_tokens[i+1:]
            curr_y = torch.cat(padded_tokens, dim=1)
            y_feat = curr_y.view(B, n_patches_h, n_patches_w, self.c_win, self.c_win, self.c_out)
            y_feat = restore_shape(y_feat, (H_out, W_out), self.c_win)
            y_feat = torch.split(y_feat, [self.feat_dims[1], H_out - self.feat_dims[1]], dim=2)[0]
            y_feat = torch.split(y_feat, [self.feat_dims[2], W_out - self.feat_dims[2]], dim=3)[0]
            frame = self.feature_decoder(y_feat)
            frame_at_rate.append(torch.sigmoid(frame))
            # NOTE this list contains a single frame decoded at different bw usages (n tokens)

        return frame_at_rate, {}

    def forward(self, codes, prev_frames, stage):
        match stage:
            # case 'feature':
            #     return self._feature_train(codes, gop_len)
            case 'coding':
                return self._coding_train(codes, prev_frames)
            case 'prediction':
                return self._coding_train(codes, prev_frames)
            case _:
                raise ValueError

    def __str__(self):
        return f'VCTBWDecoder({self.c_in},{self.c_feat},{self.c_out},{self.tf_layers},{self.tf_heads},{self.tf_ff},{self.tf_dropout})'


class VCTPredictor(nn.Module):
    def __init__(self, feat_dims, c_win, p_win):
        super().__init__()
        c_out = feat_dims[0]
        self.trg_h_pad, self.trg_w_pad = get_pad(feat_dims[1], feat_dims[2], c_win)
        self.src_h_pad = [pad + ((p_win - c_win) // 2) for pad in self.trg_h_pad]
        self.src_w_pad = [pad + ((p_win - c_win) // 2) for pad in self.trg_w_pad]
        n_patches_h = (feat_dims[1] + sum(self.trg_h_pad)) // c_win
        n_patches_w = (feat_dims[2] + sum(self.trg_w_pad)) // c_win

        self.quality_predictor = nn.Sequential(
            nn.Linear(n_patches_h*n_patches_w*c_out, n_patches_h*n_patches_w*c_out),
            nn.LeakyReLU(),
            nn.Linear(n_patches_h*n_patches_w*c_out, n_patches_h*n_patches_w*c_out),
            nn.LeakyReLU(),
            nn.Linear(n_patches_h*n_patches_w*c_out, 1)
        )

    def forward(self, conditional_tokens):
        ipdb.set_trace()
        B = conditional_tokens[0].size(0)
        batched_tokens = torch.cat(conditional_tokens, dim=0)
        prediction = self.quality_predictor(
            batched_tokens.view(B*len(conditional_tokens), -1)
        )
        return prediction
