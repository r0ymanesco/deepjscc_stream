import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from modules.feature_encoder_simple import FeatureEncoderSimple, FeatureDecoderSimple
from modules.feature_encoder import FeatureEncoder, FeatureDecoder
from modules.feature_encoder_elic import FeatureEncoderELIC, FeatureDecoderELIC
from modules.feature_encoder_elicSM import FeatureEncoderELICSM, FeatureDecoderELICSM


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


def get_rate_index(q_pred, target_q, codeword_at_rate):
    rate_indices = []
    target_rate_codewords = []
    max_idx = torch.tensor([q_pred.size(1)-1]).to(q_pred.device)
    for batch_idx, pred in enumerate(q_pred):
        comp = torch.ge(pred, torch.ones_like(pred) * target_q)
        if torch.any(comp):
            _, idx = torch.max(comp.to(torch.float), dim=-1)
            rate_indices.append(idx.unsqueeze(0))
            target_rate_codewords.append(codeword_at_rate[idx][batch_idx])
        else:
            rate_indices.append(max_idx)
            target_rate_codewords.append(codeword_at_rate[max_idx][batch_idx])
    rate_indices = torch.stack(rate_indices, dim=0)
    target_rate_codewords = torch.stack(target_rate_codewords, dim=0)
    return rate_indices, target_rate_codewords


class VCTEncoderBandwidth(nn.Module):
    def __init__(self, c_in, c_feat, feat_dims, reduced, c_win, p_win,
                 tf_layers, tf_heads, tf_ff, tf_dropout=0.,
                 target_quality=np.inf):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.feat_dims = feat_dims
        self.reduced = reduced
        self.target_quality = target_quality

        self.c_win = c_win
        self.p_win = p_win

        self.tf_layers = tf_layers
        self.tf_heads = tf_heads
        self.tf_ff = tf_ff
        self.tf_dropout = tf_dropout

        self.trg_h_pad, self.trg_w_pad = get_pad(self.feat_dims[1], self.feat_dims[2], c_win)
        self.src_h_pad = [pad + ((p_win - c_win) // 2) for pad in self.trg_h_pad]
        self.src_w_pad = [pad + ((p_win - c_win) // 2) for pad in self.trg_w_pad]
        # self.n_patches_h = (self.feat_dims[1] + sum(self.trg_h_pad)) // c_win
        # self.n_patches_w = (self.feat_dims[2] + sum(self.trg_w_pad)) // c_win

        c_out = feat_dims[0]
        self.c_out = c_out
        # self.feature_encoder = FeatureEncoderSimple(c_in, c_feat, c_out, reduced)
        # self.feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)
        self.feature_encoder = FeatureEncoderELIC(c_in, c_feat, c_out, reduced)

        tf_encoder_layer = nn.TransformerEncoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_encoder_sep = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[0])
        self.tf_encoder_joint = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[1])

        tf_decoder_layer = nn.TransformerDecoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_decoder = nn.TransformerDecoder(tf_decoder_layer, num_layers=tf_layers[2])

        # self.rate_tokens = self._get_rate_tokens(c_win**2, c_out)

    def _get_rate_tokens(self, L, L_e):
        rate_tokens = []
        for level in range(L):
            s = torch.tensor([level * (L_e - 1) / (L - 1)])
            u = torch.floor(s).to(torch.int64)
            v = ((u + 1) % L_e).to(torch.int64)
            d_u = s - u
            d_v = v - s
            alpha = d_v / (d_u + d_v)
            beta = 1 - alpha
            v_l = alpha * F.one_hot(u, num_classes=L_e) + beta * F.one_hot(v, num_classes=L_e)
            rate_tokens.append(v_l)
        rate_tokens = torch.cat(rate_tokens, dim=0)
        return rate_tokens

    def _feature_train(self, gop):
        frames = torch.cat(gop, dim=0)
        z_0 = self.feature_encoder(frames)
        return z_0, {}

    def _coding_train(self, gop):
        B = gop[0].size(0)
        n_prev_tokens = len(gop) - 1

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

        sep_input = prev_tokens.view(B*n_prev_tokens*n_patches_h*n_patches_w, -1, self.c_out)
        sep_output = self.tf_encoder_sep(sep_input)

        joint_input = sep_output.view(B, n_prev_tokens, n_patches_h, n_patches_w, -1, self.c_out)
        joint_input = joint_input.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, n_patches_h, n_patches_w, n_prev_tokens, p_win*p_win, c_out)
        joint_input = joint_input.view(B*n_patches_h*n_patches_w, -1, self.c_out)
        # (B*n_patches_h*n_patches_w, n_prev_tokens*p_win*p_win, c_out)
        tf_encoder_out = self.tf_encoder_joint(joint_input)

        trg_mask = get_mask(self.c_win**2, self.c_win**2).to(int_tokens.device)
        tf_decoder_out = self.tf_decoder(int_tokens, tf_encoder_out, tgt_mask=trg_mask)
        # (B*n_patches_h*n_patches_w, c_win*c_win, C)
        restored_decoder_out = tf_decoder_out.view(B, n_patches_h, n_patches_w, self.c_win*self.c_win, self.c_out)
        conditional_tokens = restored_decoder_out.permute(0, 3, 1, 2, 4).contiguous()
        # (B, c_win*c_win, n_patches_h, n_patches_w, C)

        rated_codewords, channel_uses = self._rated_codeword(restored_decoder_out)
        # return codeword.contiguous(), {'prev_tokens': prev_tokens,
        #                                'int_tokens': int_tokens}
        return rated_codewords, {'conditional_tokens': conditional_tokens,
                                 'channel_uses': channel_uses}

    def _rated_codeword(self, restored_decoder_out):
        B = restored_decoder_out.size(0)
        n_patches_h, n_patches_w = restored_decoder_out.size(1), restored_decoder_out.size(2)

        rate_tokens = list(torch.chunk(restored_decoder_out, chunks=self.c_win**2, dim=3))
        padding_tokens = [torch.zeros_like(rate_tokens[0])] * (self.c_win**2)
        padded_tokens = [torch.cat(rate_tokens[:i+1] + padding_tokens[:(self.c_win**2-i-1)], dim=3)
                         for i in range(self.c_win**2)]
        rated_tokens = torch.cat(padded_tokens, dim=0)
        rated_codewords = rated_tokens.view(B*(self.c_win**2), n_patches_h, n_patches_w,
                                            self.c_win, self.c_win, self.c_out)

        ch_uses_per_token = n_patches_h * n_patches_w * self.c_out // 2
        channel_uses = torch.arange(ch_uses_per_token, ch_uses_per_token*(self.c_win**2)+1, ch_uses_per_token)
        batched_channel_uses = torch.repeat_interleave(channel_uses, B).view(-1, 1)
        return rated_codewords.contiguous(), batched_channel_uses

    def _predicted_codeword(self, rated_codewords, q_pred_scaled):
        # B = q_pred_scaled.size(0)
        n_patches_h, n_patches_w = rated_codewords.size(1), rated_codewords.size(2)
        ch_uses_per_token = n_patches_h * n_patches_w * self.c_out // 2

        codewords_at_rate = torch.chunk(rated_codewords, chunks=self.c_win**2, dim=0)
        rate_indices, target_rate_codewords = get_rate_index(q_pred_scaled, self.target_quality, codewords_at_rate)
        batched_channel_uses = rate_indices * ch_uses_per_token
        return target_rate_codewords.contiguous(), batched_channel_uses, rate_indices

    def forward(self, gop, predictor, stage):
        # H_out, W_out = (self.feat_dims[1] + sum(self.trg_h_pad)), (self.feat_dims[2] + sum(self.trg_w_pad))
        rated_codewords, code_aux = self._coding_train(gop)
        match stage:
            case 'coding':
                # rated_codewords = restore_shape(rated_codewords, (H_out, W_out), self.c_win)
                # rated_codewords = torch.split(rated_codewords, [self.feat_dims[1], H_out - self.feat_dims[1]], dim=2)[0]
                # rated_codewords = torch.split(rated_codewords, [self.feat_dims[2], W_out - self.feat_dims[2]], dim=3)[0]
                return rated_codewords.contiguous(), {'channel_uses': code_aux['channel_uses']}
            case 'prediction':
                q_pred, _ = predictor(code_aux['conditional_tokens'])
                return rated_codewords.contiguous(), {'q_pred': q_pred,
                                                      'channel_uses': code_aux['channel_uses']}
            case 'fine_tune':
                q_pred, q_pred_scaled = predictor(code_aux['conditional_tokens'])
                rated_codewords, batched_channel_uses, rate_indices = self._predicted_codeword(rated_codewords, q_pred_scaled)
                return rated_codewords.contiguous(), {'q_pred': q_pred,
                                                      'q_pred_scaled': q_pred_scaled,
                                                      'channel_uses': batched_channel_uses,
                                                      'rate_indices': rate_indices}

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

        c_out = feat_dims[0]
        self.c_out = c_out
        # self.feature_decoder = FeatureDecoderSimple(c_out, c_feat, c_in, reduced)
        # self.auto_feature_encoder = FeatureEncoderSimple(c_in, c_feat, c_out, reduced)
        # self.feature_decoder = FeatureDecoder(c_out, c_feat, c_in, reduced)
        # self.auto_feature_encoder = FeatureEncoder(c_in, c_feat, c_out, reduced)
        self.feature_decoder = FeatureDecoderELIC(c_out, c_feat, c_in, reduced)
        self.auto_feature_encoder = FeatureEncoderELIC(c_in, c_feat, c_out, reduced)

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

    def _coding_train(self, int_tokens, prev_frames):
        B = int_tokens.size(0)
        _B = prev_frames[0].size(0)
        repeat_factor = B // _B
        n_prev_tokens = len(prev_frames)
        H_out, W_out = (self.feat_dims[1] + sum(self.trg_h_pad)), (self.feat_dims[2] + sum(self.trg_w_pad))
        n_patches_h, n_patches_w = int_tokens.size(1), int_tokens.size(2)
        decoder_input = int_tokens.view(B*n_patches_h*n_patches_w, self.c_win**2, self.c_out)

        batch_prev_frames = torch.cat(prev_frames, dim=0)
        prev_y = self.auto_feature_encoder(batch_prev_frames)

        prev_tokens = get_src_tokens(prev_y, (*self.src_w_pad, *self.src_h_pad), self.p_win, self.c_win)
        # (B*n_prev_tokens, n_patches_h, n_patches_w, p_win, p_win, C)
        sep_input = prev_tokens.view(-1, self.p_win**2, self.c_out)
        sep_output = self.tf_encoder_sep(sep_input)

        joint_input = sep_output.view(_B, n_prev_tokens, n_patches_h, n_patches_w, -1, self.c_out)
        joint_input = joint_input.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, n_patches_h, n_patches_w, n_prev_tokens, p_win*p_win, c_out)
        joint_input = joint_input.view(_B*n_patches_h*n_patches_w, -1, self.c_out)
        # (B*n_patches_h*n_patches_w, n_prev_tokens*p_win*p_win, c_out)
        tf_encoder_out = self.tf_encoder_joint(joint_input)
        batched_tf_encoder_out = torch.tile(tf_encoder_out, (repeat_factor, 1, 1))

        trg_mask = get_mask(self.c_win**2, self.c_win**2).to(int_tokens.device)
        tf_decoder_out = self.tf_decoder(decoder_input, batched_tf_encoder_out, tgt_mask=trg_mask)
        y_feat = tf_decoder_out.view(B, n_patches_h, n_patches_w, self.c_win, self.c_win, self.c_out)
        y_feat = restore_shape(y_feat, (H_out, W_out), self.c_win)
        y_feat = torch.split(y_feat, [self.feat_dims[1], H_out - self.feat_dims[1]], dim=2)[0]
        y_feat = torch.split(y_feat, [self.feat_dims[2], W_out - self.feat_dims[2]], dim=3)[0]
        frames = torch.sigmoid(self.feature_decoder(y_feat))
        return frames, {}

    def _process_codeword(self, codes, batch_channel_uses):
        # NOTE it is more efficient to concatenate the tokens from all rates instead of creating separate batches
        # The problem with this is that the power scaling for each rate is different so hard to do
        B = codes.size(0)
        assert B == batch_channel_uses.size(0)
        n_patches_h = (self.feat_dims[1] + sum(self.trg_h_pad)) // self.c_win
        n_patches_w = (self.feat_dims[2] + sum(self.trg_w_pad)) // self.c_win

        ch_uses_per_token = n_patches_h * n_patches_w * self.c_out // 2
        batch_rate_uses = (batch_channel_uses / ch_uses_per_token).to(torch.int)
        int_tokens = codes.view(B, n_patches_h, n_patches_w, self.c_win**2, self.c_out)
        # (B, n_patches_h, n_patches_w, c_win*c_win, C)

        padding_tokens = torch.zeros_like(int_tokens[:1])
        padding_tokens = list(torch.chunk(padding_tokens, chunks=self.c_win**2, dim=3))

        int_tokens = torch.chunk(int_tokens, chunks=B, dim=0)
        rate_uses = torch.chunk(batch_rate_uses, chunks=B, dim=0)

        processed_codewords = []
        for (tokens, rate) in zip(int_tokens, rate_uses):
            rate = int(rate.view(-1).item())
            code = torch.split(tokens, [rate, self.c_win**2-rate], dim=3)[0]
            padding = padding_tokens[:(self.c_win**2-rate)]
            codeword = torch.cat((code, *padding), dim=3)
            processed_codewords.append(codeword)
        processed_codewords = torch.cat(processed_codewords, dim=0)
        return processed_codewords.contiguous()

    def forward(self, codes, prev_frames, batch_channel_uses, stage):
        processed_codewords = self._process_codeword(codes, batch_channel_uses)
        frames, _ = self._coding_train(processed_codewords, prev_frames)
        match stage:
            case 'coding':
                frames_at_rate = torch.chunk(frames, chunks=self.c_win**2, dim=0)
                return frames_at_rate, {}
            case 'prediction':
                frames_at_rate = torch.chunk(frames, chunks=self.c_win**2, dim=0)
                return frames_at_rate, {}
            case 'fine_tune':
                return [frames], {}
            case _:
                raise ValueError

    def __str__(self):
        return f'VCTBWDecoder({self.c_in},{self.c_feat},{self.c_out},{self.tf_layers},{self.tf_heads},{self.tf_ff},{self.tf_dropout})'


class VCTPredictor(nn.Module):
    def __init__(self, loss, feat_dims, c_win, p_win,
                 tf_layers, tf_heads, tf_ff, tf_dropout=0.):
        super().__init__()
        self.loss = loss
        self.c_win = c_win

        c_out = feat_dims[0]
        self.c_out = c_out
        self.trg_h_pad, self.trg_w_pad = get_pad(feat_dims[1], feat_dims[2], c_win)
        self.src_h_pad = [pad + ((p_win - c_win) // 2) for pad in self.trg_h_pad]
        self.src_w_pad = [pad + ((p_win - c_win) // 2) for pad in self.trg_w_pad]
        self.n_patches_h = (feat_dims[1] + sum(self.trg_h_pad)) // c_win
        self.n_patches_w = (feat_dims[2] + sum(self.trg_w_pad)) // c_win
        self.ch_uses_per_token = self.n_patches_h * self.n_patches_w * c_out // 2
        vals_per_token = self.n_patches_h * self.n_patches_w * c_out
        self.n_rates = self.c_win ** 2

        tf_encoder_layer = nn.TransformerEncoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_encoder_sep = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[0])
        self.tf_encoder_joint = nn.TransformerEncoder(tf_encoder_layer, num_layers=tf_layers[1])

        tf_decoder_layer = nn.TransformerDecoderLayer(c_out, tf_heads, tf_ff, tf_dropout, batch_first=True)
        self.tf_decoder = nn.TransformerDecoder(tf_decoder_layer, num_layers=tf_layers[2])

        self.quality_predictor = nn.Sequential(
            nn.Linear(vals_per_token, vals_per_token),
            nn.LeakyReLU(),
            nn.Linear(vals_per_token, vals_per_token),
            nn.LeakyReLU(),
            nn.Linear(vals_per_token, vals_per_token),
            nn.LeakyReLU(),
            nn.Linear(vals_per_token, vals_per_token),
            nn.LeakyReLU(),
            nn.Linear(vals_per_token, 1),
        )

    def _scale_for_metric(self, q_pred):
        match self.loss:
            case 'l2':
                return q_pred, 10 * torch.log10(1. / q_pred)
            case 'msssim':
                return q_pred.clamp(0, 1), q_pred.clamp(0, 1)
            case _:
                raise ValueError

    def forward(self, int_tokens, prev_tokens):
        B = int_tokens.size(0) // (self.n_patches_h * self.n_patches_w)
        n_prev_tokens = prev_tokens.size(0) // B

        sep_input = prev_tokens.view(B*n_prev_tokens*self.n_patches_h*self.n_patches_w, -1, self.c_out)
        sep_output = self.tf_encoder_sep(sep_input)

        joint_input = sep_output.view(B, n_prev_tokens, self.n_patches_h, self.n_patches_w, -1, self.c_out)
        joint_input = joint_input.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, n_patches_h, n_patches_w, n_prev_tokens, p_win*p_win, c_out)
        joint_input = joint_input.view(B*self.n_patches_h*self.n_patches_w, -1, self.c_out)
        # (B*n_patches_h*n_patches_w, n_prev_tokens*p_win*p_win, c_out)
        tf_encoder_out = self.tf_encoder_joint(joint_input)

        trg_mask = get_mask(self.c_win**2, self.c_win**2).to(int_tokens.device)
        tf_decoder_out = self.tf_decoder(int_tokens, tf_encoder_out, tgt_mask=trg_mask)
        restored_decoder_out = tf_decoder_out.view(B, self.n_patches_h, self.n_patches_w, self.c_win*self.c_win, self.c_out)
        batched_tokens = restored_decoder_out.permute(0, 3, 1, 2, 4).contiguous()
        # (B, c_win*c_win, n_patches_h, n_patches_w, C)

        predictor_out = self.quality_predictor(
            batched_tokens.view(B, (self.c_win**2), -1)
        ).view(B, self.c_win**2)
        prediction = torch.cumsum(predictor_out.pow(2), dim=1)
        prediction = torch.fliplr(prediction)
        return self._scale_for_metric(prediction)
