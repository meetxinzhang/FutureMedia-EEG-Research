# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/6/21 2:50 PM
@desc:
"""
from torch import nn


def use_torch_interface():
    encoder_layer = nn.TransformerEncoderLayer(d_model=96, nhead=8, batch_first=True)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    return encoder


class EEGModel(nn.Module):
    def __init__(self):
        super(EEGModel, self).__init__()
        self.encoder = use_torch_interface()
        self.den1 = nn.Linear(in_features=96, out_features=192)
        self.den2 = nn.Linear(in_features=192, out_features=96)
        self.den3 = nn.Linear(in_features=96, out_features=40)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        h1 = self.encoder(x)  # [bs, time_step, 96]
        last_step = h1[:, -1, :]  # [bs, 96]
        h2 = self.den1(last_step)
        h3 = self.den2(h2)
        logits = self.den3(h3)
        return logits


