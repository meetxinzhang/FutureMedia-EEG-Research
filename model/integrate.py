# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/6/21 2:50 PM
@desc:
"""
from torch import nn
from torch.nn import functional as F
from model .transformer import Encoder, EncoderLayer, MultiHeadedAttention, PositionWiseFeedForward, PositionalEncoding


class EEGModel(nn.Module):
    def __init__(self, d_model=96, dropout=0.2, n_layers=6, n_head=8):
        super(EEGModel, self).__init__()
        # Encoder of transformer
        # self.encoder = use_torch_interface()
        # self.encoder = Encoder(dim_in=96, n_head=8, time_step=512, dropout=0.3)
        self.pe = PositionalEncoding(embed_len=96, dropout=0.2, max_seq_len=512)
        attn = MultiHeadedAttention(n_head=8, d_model=96, dropout=0.2)
        ff = PositionWiseFeedForward(d_model=96, d_ff=256, dropout=0.2)
        self.encoder = Encoder(EncoderLayer(96, attn, ff, dropout=0.2), N=6)

        # Classifier
        self.fl = nn.Flatten(start_dim=1, end_dim=-1)
        self.den1 = nn.Linear(in_features=49152, out_features=512, bias=False)
        self.den2 = nn.Linear(in_features=512, out_features=128, bias=False)
        self.den3 = nn.Linear(in_features=128, out_features=40, bias=False)
        self.bn1 = nn.BatchNorm1d(512, affine=False)  # Without Learnable Parameters
        self.bn2 = nn.BatchNorm1d(128, affine=False)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, mask):
        x = self.pe(x)
        h1 = self.encoder(x, mask)  # [bs, time_step, 96]
        # last_step = h1[:, -1, :]  # [bs, 96]
        fl = self.fl(h1)  # [bs, time_step*96]

        h2 = F.leaky_relu(self.den1(fl))  # [bs, 256]
        h2 = self.dropout(self.bn1(h2))
        h3 = F.leaky_relu(self.den2(h2))  # [bs, 128]
        h3 = self.dropout(self.bn2(h3))
        h4 = F.leaky_relu(self.den3(h3))  # [bs, 40]
        logits = F.softmax(h4, dim=-1)
        return logits


# def use_torch_interface():
#     encoder_layer = nn.TransformerEncoderLayer(d_model=96, nhead=8, batch_first=True)
#     encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#     return encoder

# def make_transformer(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
#     """Helper: Construct a model from hyperparameters."""
#     c = copy.deepcopy
#     attn = MultiHeadedAttention(h, d_model)
#     ff = PositionWiseFeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)
#     model = TransformerModel(
#         Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N))
#     # This was important from their code.
#     # Initialize parameters with Glorot / fan_avg.
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform(p)
#     return model

#
# class TransformerModel(nn.Module):
#     """
#     A standard Encoder-Decoder architecture. Base for this and many
#     other models.
#     """
#
#     def __init__(self, encoder):
#         super(TransformerModel, self).__init__()
#         self.encoder = encoder
#
#     def forward(self, src, src_mask):
#         "Take in and process masked src and target sequences."
#         return self.encode(src, src_mask)
#
#     def encode(self, src, src_mask):
#         return self.encoder(src), src_mask
