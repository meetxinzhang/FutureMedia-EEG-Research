# # encoding: utf-8
# """
# @author: Xin Zhang
# @contact: zhangxin@szbl.ac.cn
# @time: 12/6/21 6:16 PM
# @desc:
# """
#
# import numpy as np
# from torch import nn
# import torch.nn.functional as F
# import torch
#
#
# class PositionalEncoding(nn.Module):
#     """
#     PE function in paper
#     Input: x
#     Output: x + position_encoder
#     I don't like the absolute position encoding at all.
#     """
#     def __init__(self, embed_len, max_seq_len, dropout):
#         super(PositionalEncoding, self).__init__()
#         self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed_len)) for i in range(embed_len)]
#                                 for pos in range(max_seq_len)])  # shape=[seq_len, embed_len]
#         self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])   # interval=2, 偶数 sin()
#         self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])   # interval=1, 奇数 cos()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # concat
#         out = x + nn.Parameter(self.pe, requires_grad=False).cuda()
#         out = self.dropout(out)
#         return out
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_head, dropout=0.0):
#         """
#         :param d_model: dim of input in each time-step, also the dimension of feature map
#         :param n_head:
#         :param dropout: rate to dropout
#         """
#         super(MultiHeadAttention, self).__init__()
#         self.num_head = n_head
#         assert d_model % n_head == 0    # head 数必须能够整除隐层大小
#         self.dim_head = d_model // n_head   # 按照 head 数量进行张量均分
#
#         self.fc_Q = nn.Linear(d_model, n_head * self.dim_head)  # Q，matrix multiplication by linear
#         self.fc_K = nn.Linear(d_model, n_head * self.dim_head)  # K
#         self.fc_V = nn.Linear(d_model, n_head * self.dim_head)  # V
#
#         self.attention = ScaledDotProductAttention()
#         self.fc = nn.Linear(n_head * self.dim_head, d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model)   # 自带的 LayerNorm 方法
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         Q = self.fc_Q(x)
#         K = self.fc_K(x)
#         V = self.fc_V(x)
#         Q = Q.view(batch_size * self.num_head, -1, self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
#         K = K.view(batch_size * self.num_head, -1, self.dim_head)
#         V = V.view(batch_size * self.num_head, -1, self.dim_head)
#         # if mask:  # TODO
#         #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
#         scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
#         context = self.attention(Q, K, V, scale)  # Scaled_Dot_Product_Attention计算
#         context = context.view(batch_size, -1, self.dim_head * self.num_head)  # reshape 回原来的形状
#         out = self.fc(context)   # 全连接
#         out = self.dropout(out)
#         out = out + x      # 残差连接,ADD
#         out = self.layer_norm(out)  # 对应Norm
#         return out
#
#
# class ScaledDotProductAttention(nn.Module):
#     """Scaled Dot-Product"""
#     def __init__(self):
#         super(ScaledDotProductAttention, self).__init__()
#
#     def forward(self, Q, K, V, scale=None):
#         attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
#         if scale:
#             attention = attention * scale
#         # if mask:  # TODO change this
#         #     attention = attention.masked_fill_(mask == 0, -1e9)
#         attention = F.softmax(attention, dim=-1)
#         context = torch.matmul(attention, V)
#         return context
#
#
# class PositionWiseFeedForward(nn.Module):
#     def __init__(self, dim_model, hidden, dropout=0.0):
#         super(PositionWiseFeedForward, self).__init__()
#         self.fc1 = nn.Linear(dim_model, hidden)
#         self.fc2 = nn.Linear(hidden, dim_model)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(dim_model)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = F.relu(out)
#         out = self.fc2(out)   # 两层全连接
#         out = self.dropout(out)
#         out = out + x  # 残差连接
#         out = self.layer_norm(out)
#         return out
#
#
# class Encoder(nn.Module):
#     def __init__(self, dim_in, n_head, time_step, dropout):
#         super(Encoder, self).__init__()
#         self.attention = MultiHeadAttention(dim_in, n_head, dropout)
#         self.feed_forward = PositionWiseFeedForward(dim_in, time_step, dropout)
#
#     def forward(self, x):
#         out = self.attention(x)
#         out = self.feed_forward(out)
#         return out
