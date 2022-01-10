# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/6/21 2:50 PM
@desc:
"""
from torch import nn
from torch.nn import functional as F
# from model.transformer import Encoder, EncoderLayer, MultiHeadedAttention, PositionWiseFeedForward, PositionalEncoding
from model.transformer import PositionalEncoding
from einops.layers.torch import Rearrange


class EEGModel(nn.Module):
    def __init__(self, d_model=96, dropout=0.2, n_layers=6, n_head=8):
        super(EEGModel, self).__init__()

        # Vision-Transformer
        self.rearrange = Rearrange('bs (n t) c -> bs n (t c)', c=96, t=16)
        self.pre_linear = nn.Linear(in_features=96 * 16, out_features=64, bias=True)

        self.pe = PositionalEncoding(embed_len=64, dropout=0.2, max_seq_len=64)
        # Encoder of transformer
        # attn = MultiHeadedAttention(n_head=8, d_model=64, dropout=0.2)
        # ff = PositionWiseFeedForward(d_model=64, d_ff=256, dropout=0.2)
        # self.encoder = Encoder(EncoderLayer(64, attn, ff, dropout=0.2), N=5)
        self.encoder = use_torch_interface()

        # Classifier
        self.fl = nn.Flatten(start_dim=1, end_dim=-1)
        self.den1 = nn.Linear(in_features=4096, out_features=512, bias=True)
        self.den2 = nn.Linear(in_features=512, out_features=128, bias=True)
        self.den3 = nn.Linear(in_features=128, out_features=40, bias=True)
        self.bn0 = nn.BatchNorm1d(64, affine=False)
        self.bn1 = nn.BatchNorm1d(512, affine=False)  # Without Learnable Parameters
        self.bn2 = nn.BatchNorm1d(128, affine=False)
        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)

    def forward(self, x, mask):
        patches = self.rearrange(x)  # [bs, 64, 16*96]
        patches_ed = self.bn0(F.leaky_relu(self.pre_linear(patches)))  # [bs, n(time:1024/16=64), 64]

        patches_ed = self.pe.forward(patches_ed)  # [bs, 64, 64]
        h1 = self.encoder.forward(patches_ed, mask)  # [bs, time_step, 64]
        # print(h1.shape)  # [bs, time:1024/16=64, 64]
        # last_step = h1[:, -1, :]  # [bs, 64]
        fl = self.fl(h1)  # [bs, time_step*64]

        h2 = F.leaky_relu(self.den1(fl))  # [bs, 256]
        h2 = self.dp1(self.bn1(h2))
        h3 = F.leaky_relu(self.den2(h2))  # [bs, 128]
        h3 = self.dp2(self.bn2(h3))
        h4 = self.den3(h3)  # [bs, 40]
        logits = F.softmax(h4, dim=-1)
        return logits


def use_torch_interface():
    encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    return encoder

