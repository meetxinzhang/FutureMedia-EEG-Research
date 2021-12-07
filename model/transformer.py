# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/7/21 1:42 PM
@desc:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import seaborn
seaborn.set_context(context="talk")


def clones(module, N):
    """Produce N identical layers.
    copy.deepcopy(module): new a new object and all its sub-objects in memory.
    copy.copy(module)    : new a new object in memory, but its sub-objects are pointers to their old memory.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """
    'Scaled Dot Product Attention'
    Formulation on paper: Attention(Q,K,V)=softmax(QK^T/√d_k)V
    :params query, key, value are same size as [n_batches, n_head, seq_len, d_k]
    """
    d_k = query.size(-1)  # key.size = query.size
    # [n_batches, n_head, seq_len, d_k] * [n_batches, n_head, d_k, seq_len]
    #   = [n_batches, n_head, seq_len, seq_len]
    # It's the parallel version of vector dot product
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # if element in mask is 0, then set -1e9 at corresponding
    p_attn = F.softmax(scores, dim=-1)  # softmax on columns [n_batches, n_head, seq_len, seq_len]
    if dropout is not None:
        p_attn = dropout(p_attn)  # [n_batches, n_head, seq_len, seq_len]
    # [n_batches, n_head, seq_len, seq_len] * [n_batches, n_head, seq_len, d_k]
    #   = [n_batches, n_head, seq_len, d_k]
    #   Attend at element level of Value rather than at value itself, hard to understand(Took me 2h) but correct.
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_head
        self.n_h = n_head
        self.linear_list = clones(nn.Linear(d_model, d_model), 4)  # 4 linear_layers, one for output
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        query, key, value are one thing, shape=[n_batches, seq_len, d_model]
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch, then divide (q,k,v) into n_head, like multi-channels in CNN.
        #    from [n_batches, seq_len, d_model]  >
        #    [n_batches, seq_len, d_model] > [n_batches, seq_len, n_head, d_k] > [n_batches, n_head, seq_len, d_k]
        #    Advance n_head is for subsequent parallel calculations
        query, key, value = \
            [linear(x).view(nbatches, -1, self.n_h, self.d_k).transpose(1, 2)
             for linear, x in zip(self.linear_list, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        #    [n_batches, n_head, seq_len, d_k], [n_batches, n_head, seq_len, seq_len]
        b, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" heads, let shape=input.shape, like channels combination in CNN. Then apply a final linear.
        #    [n_b, n_h, seq_len, d_k] > [n_b, seq_len, n_h, d_k] > [n_b, seq_len, n_h*d_k]
        b = b.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_h * self.d_k)
        return self.linear_list[-1](b)  # [n_b, seq_len, d_model]


class PositionWiseFeedForward(nn.Module):
    """Two linear to Feed Forward."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  #
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Encoder(nn.Module):
    """Encoder is a stack of N EncoderLayer"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layer_list = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        :param x:     [n_b, seq_len, d_model]
        :param mask:  [n_b, seq_len] like [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        :return:      [n_b, seq_len, d_model]
        """
        for layer in self.layer_list:
            x = layer(x, mask)  # the previous output is the next input
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attention and feed forward"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn        # Multi-Head Attention on citation
        self.feed_forward = feed_forward  # Feed Forward on citation
        self.sublayer = clones(ResidualConnectionNorm(size, dropout), 2)  # 2 Add & Norm on citation
        self.size = size

    def forward(self, x, mask):
        #  [n_b, seq_len, d_model]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # sublayer 1
        return self.sublayer[1](x, self.feed_forward(x))   # sublayer 2  [n_b, seq_len, d_model]


class ResidualConnectionNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    The norm is last in citation, but there are other situations where closer to intuition.
    """
    def __init__(self, size, dropout):
        super(ResidualConnectionNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return self.norm(x + self.dropout(sublayer(x)))


class LayerNorm(nn.Module):
    """Construct a layer-normalization module, often used in RNN. (See citation for details)."""
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    """To embed."""

    def __init__(self, embed_len, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_len)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_len, 2) * -(math.log(10000.0) / embed_len))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
