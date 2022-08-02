import ipdb
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def generate_square_subsequent_mask(self, sz):
        bs = sz.size(0)
        seq_len = sz.size(1)
        mask = (torch.triu(torch.ones(bs, seq_len, seq_len)) == 1).transpose(1, 2)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        ipdb.set_trace()
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into n_heads
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k)

        # transpose to get dims bs * n_heads * seq_len * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concat heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(1, max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[0, pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[0, pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(n_heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(n_heads, d_model, dropout)
        self.attn_2 = MultiHeadAttention(n_heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, enc_output, enc_output, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class TFEncoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = self.get_clones(EncoderLayer(d_model, n_heads, d_ff, dropout), n_layers)
        self.norm = nn.LayerNorm(d_model)

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, src, mask=None):
        x = self.pe(src)
        # x = src
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def __str__(self):
        return f'TFEncoder({self.d_model},{self.n_layers},{self.n_heads},{self.d_ff},{self.dropout})'


class TFDecoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = self.get_clones(DecoderLayer(d_model, n_heads, d_ff, dropout), n_layers)
        self.norm = nn.LayerNorm(d_model)

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, trg, enc_output, src_mask=None, trg_mask=None):
        x = self.pe(trg)
        # x = trg
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, trg_mask)
        return self.norm(x)

    def __str__(self):
        return f'TFDecoder({self.d_model},{self.n_layers},{self.n_heads},{self.d_ff},{self.dropout})'
