import ipdb
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import ResidualBlock, AttentionBlock


class MultiHeadAttention2D(nn.Module):
    def __init__(self, n_heads, c_in, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = c_in // n_heads

        self.q_conv = nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=1, stride=1)
        self.k_conv = nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=1, stride=1)
        self.v_conv = nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=1, stride=1)

        self.att_net = nn.Sequential(
            AttentionBlock(c_in),
            ResidualBlock(in_ch=c_in, out_ch=n_heads)
        )
        self.out = nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=1, stride=1)

    def attention(self, q, k, v, mask=None):
        bs, q_seq_len, k_seq_len, c, h, w = q.size()

        att_input = (q + k).view(bs*q_seq_len*k_seq_len, c, h, w)
        scores = self.att_net(att_input) / math.sqrt(self.d_k)
        scores = scores.view(bs, q_seq_len, k_seq_len, self.n_heads, h, w)
        scores = scores.transpose(2, 3)

        if mask is not None:
            mask = mask.transpose(2, 3)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = F.softmax(scores, dim=3).unsqueeze(4)
        # repeat to get dims (bs, q_seq_len, n_heads, k_seq_len, c//n_heads, h, w)
        scores = torch.repeat_interleave(scores, c//self.n_heads, dim=4)

        # dims (bs, q_seq_len, n_heads, c//n_heads, h, w)
        output = torch.sum(scores * v, dim=3)
        return output

    def forward(self, query, k, v, mask=None):
        # assumes input shape (bs, seq_len, c, h, w)
        (bs, q_seq_len, c, h, w) = query.size()
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)

        # perform linear operation and split into n_heads
        query = query.view(bs * q_seq_len, c, h, w)
        query = self.q_conv(query)
        query = query.view(bs, q_seq_len, 1, c, h, w)
        # dims (bs, q_seq_len, k_seq_len, c, h, w)
        query = torch.repeat_interleave(query, k_seq_len, dim=2)

        k = k.view(bs * k_seq_len, c, h, w)
        k = self.k_conv(k)
        k = k.view(bs, 1, k_seq_len, c, h, w)
        # dims (bs, q_seq_len, k_seq_len, c, h, w)
        k = torch.repeat_interleave(k, q_seq_len, dim=1)

        v = v.view(bs * v_seq_len, c, h, w)
        v = self.v_conv(v)
        v = v.view(bs, 1, v_seq_len, c, h, w)
        # dims (bs, q_seq_len, v_seq_len, c, h, w)
        v = torch.repeat_interleave(v, q_seq_len, dim=1)
        v = v.view(bs, q_seq_len, k_seq_len, self.n_heads, c//self.n_heads, h, w)
        # dims (bs, q_seq_len, n_heads, v_seq_len, c//n_heads, h, w)
        v = v.transpose(2, 3)

        # calculate attention
        # scores = self.attention(q, k, v, d_k, mask, self.dropout)
        # dims (bs, q_seq_len, k_seq_len, c, h, w)
        v_out = self.attention(query, k, v, mask)

        v_out = v_out.view(bs*q_seq_len, c, h, w)
        # dims (bs, q_seq_len, k_seq_len, c, h, w)
        output = self.out(v_out).view(bs, q_seq_len, c, h, w)
        # returns input shape (bs, seq_len, c, h, w)
        return output


class PositionalEncoder2D(nn.Module):
    def __init__(self, feat_dims, num_features_max, max_seq_len=80):
        super().__init__()

        # create constant 'pe' matrix with values dependant on
        # pos and i
        _, h, w = feat_dims
        pe = torch.zeros(max_seq_len, num_features_max, requires_grad=False)
        for pos in range(max_seq_len):
            for i in range(0, num_features_max, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/num_features_max)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/num_features_max)))

        pe = pe.unsqueeze(0)
        pe = pe.unsqueeze(3)
        pe = pe.unsqueeze(4)
        pe = torch.repeat_interleave(pe, h, dim=3)
        pe= torch.repeat_interleave(pe, w, dim=4)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # assumes input shape (bs, seq_len, c, h, w)
        _, seq_len, c, h, w = x.size()

        # make embeddings relatively larger
        x = x * math.sqrt(c)
        # add constant to embedding
        x = x + self.pe[:, :seq_len, :c, :, :]
        # returns shape (bs, seq_len, c, h, w)
        return x


class FeedForward2D(nn.Module):
    def __init__(self, c_in, c_feat, dropout=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=c_in,
                               out_channels=c_feat,
                               kernel_size=1,
                               stride=1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(in_channels=c_feat,
                               out_channels=c_in,
                               kernel_size=1,
                               stride=1)

    def forward(self, x):
        # assumes input shape (bs, seq_len, c, h, w)
        bs, seq_len, c, h, w = x.size()

        x = x.view(bs * seq_len, c, h, w)
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.conv2(x)
        x = x.view(bs, seq_len, c, h, w)
        # returns shape (bs, seq_len, c, h, w)
        return x


class EncoderLayer2D(nn.Module):
    def __init__(self, c_in, n_heads, c_ff, dropout=0.):
        super().__init__()
        self.norm_1 = nn.InstanceNorm2d(c_in)
        self.norm_2 = nn.InstanceNorm2d(c_in)

        self.attn = MultiHeadAttention2D(n_heads, c_in, dropout)
        self.ff = FeedForward2D(c_in, c_ff, dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # assumes input shape (bs, seq_len, c, h, w)
        bs, seq_len, c, h, w = x.size()

        x2 = x.view(bs * seq_len, c, h, w)
        x2 = self.norm_1(x2)
        x2 = x2.view(bs, seq_len, c, h, w)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))

        x2 = x.view(bs * seq_len, c, h, w)
        x2 = self.norm_2(x2)
        x2 = x2.view(bs, seq_len, c, h, w)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer2D(nn.Module):
    def __init__(self, c_in, n_heads, c_ff, dropout=0.):
        super().__init__()
        self.norm_1 = nn.InstanceNorm2d(c_in)
        self.norm_2 = nn.InstanceNorm2d(c_in)
        self.norm_3 = nn.InstanceNorm2d(c_in)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention2D(n_heads, c_in, dropout)
        self.attn_2 = MultiHeadAttention2D(n_heads, c_in, dropout)
        self.ff = FeedForward2D(c_in, c_ff, dropout)

    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        # assumes input shape (bs, seq_len, c, h, w)
        bs, seq_len, c, h, w = x.size()

        x2 = x.view(bs * seq_len, c, h, w)
        x2 = self.norm_1(x2)
        x2 = x2.view(bs, seq_len, c, h, w)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))

        x2 = x.view(bs * seq_len, c, h, w)
        x2 = self.norm_2(x2)
        x2 = x2.view(bs, seq_len, c, h, w)
        x = x + self.dropout_2(self.attn_2(x2, enc_output, enc_output, src_mask))

        x2 = x.view(bs * seq_len, c, h, w)
        x2 = self.norm_3(x2)
        x2 = x2.view(bs, seq_len, c, h ,w)
        x = x + self.dropout_3(self.ff(x2))
        # returns shape (bs, seq_len, c, h, w)
        return x


class TFEncoder2D(nn.Module):
    def __init__(self, c_in, n_layers, n_heads, c_ff, dropout,
                 feat_dims, num_features_max=1e5, max_seq_len=80):
        super().__init__()
        self.n_layers = n_layers
        self.pe = PositionalEncoder2D(feat_dims, num_features_max, max_seq_len)
        self.layers = self.get_clones(EncoderLayer2D(c_in, n_heads, c_ff, dropout), n_layers)
        self.norm = nn.InstanceNorm2d(c_in)

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def run_function(self, module_idx):
        def custom_forward(*inputs):
            x = inputs[0]
            mask = inputs[1]
            x = self.layers[module_idx](x, mask)
            return x
        return custom_forward

    def forward(self, x, mask=None):
        # assumes input shape (bs, seq_len, c, h, w)
        bs, seq_len, c, h, w = x.size()

        # x = self.pe(x)

        # for layer_idx in range(self.n_layers):
        #     x = checkpoint.checkpoint(
        #         self.run_function(layer_idx), x, mask)
        for layer in self.layers:
            x = layer(x, mask)

        x = x.view(bs * seq_len, c, h, w)
        x = self.norm(x)
        x = x.view(bs, seq_len, c, h, w)
        # returns shape (bs, seq_len, c, h, w)
        return x


class TFDecoder2D(nn.Module):
    def __init__(self, c_in, n_layers, n_heads, c_ff, dropout,
                 feat_dims, num_features_max=1e5, max_seq_len=80):
        super().__init__()
        self.n_layers = n_layers
        self.pe = PositionalEncoder2D(feat_dims, num_features_max, max_seq_len)
        self.layers = self.get_clones(DecoderLayer2D(c_in, n_heads, c_ff, dropout), n_layers)
        self.norm = nn.InstanceNorm2d(c_in)

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def run_function(self, module_idx):
        def custom_forward(*inputs):
            x = inputs[0]
            enc_output = inputs[1]
            src_mask = inputs[2]
            trg_mask = inputs[3]
            x = self.layers[module_idx](x, enc_output, src_mask, trg_mask)
            return x
        return custom_forward

    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        # assumes input shape (bs, seq_len, c, h, w)
        bs, seq_len, c, h, w = x.size()

        # x = self.pe(x)

        # for layer_idx in range(self.n_layers):
        #     x = checkpoint.checkpoint(
        #         self.run_function(layer_idx), x, enc_output, src_mask, trg_mask)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, trg_mask)

        x = x.view(bs * seq_len, c, h, w)
        x = self.norm(x)
        x = x.view(bs, seq_len, c, h, w)
        # returns shape (bs, seq_len, c, h, w)
        return x
