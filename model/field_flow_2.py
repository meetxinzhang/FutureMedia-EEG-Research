# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/8 18:29
 @desc:
"""

import torch
from torch import nn
from modules.linear_conv2d import LinearConv2D
import einops


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=.002)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class FieldFlow2(nn.Module):
    def __init__(self, channels=127):
        super().__init__()
        self.c = channels

        self.lc1 = LinearConv2D(input_channels=channels, out_channels=254, groups=channels,
                                embedding=85, kernel_width=15, kernel_stride=3,
                                activate_height=2, activate_stride=2)
        self.lc2 = LinearConv2D(input_channels=254, out_channels=508, groups=channels,
                                embedding=42, kernel_width=7, kernel_stride=2,
                                activate_height=2, activate_stride=2)
        self.lc3 = LinearConv2D(input_channels=508, out_channels=254, groups=channels,
                                embedding=21, kernel_width=7, kernel_stride=2,
                                activate_height=2, activate_stride=1)
        self.lc4 = LinearConv2D(input_channels=254, out_channels=127, groups=channels,
                                embedding=20, kernel_width=7, kernel_stride=2,
                                activate_height=2, activate_stride=1)
        self.l1 = nn.Linear(in_features=50673, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=40)
        self.fla = nn.Flatten(start_dim=1, end_dim=-1)

        # b c f t
        self.tf = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(
            d_model=84, nhead=4, dim_feedforward=128, dropout=0.3, batch_first=True, norm_first=True), num_layers=1)
        self.ch_embed = nn.Parameter(torch.zeros(1, channels, 84))

        self.apply(_init_weights)

    def forward(self, x):
        # [b c=127 f=85 t=500]
        x = self.lc1(x)  # [b 254 42 167]
        x = self.lc2(x)  # [b 508 21 83]
        [b, _, f, _] = x.size()

        _t = x.size()[-1]
        x = einops.rearrange(x, 'b (g d) f t -> (b t) g (f d)', g=self.c, d=4)  # [bt 127 84]
        ch_embed = self.ch_embed.expand(b * _t, -1, -1)
        x = torch.add(x, ch_embed)
        x = self.tf(x)  # (b t) g (f d)
        x = einops.rearrange(x, '(b t) g (f d) -> b (g d) f t', b=b, t=_t, f=f, d=4)  # [2, 508, 21, 84]

        x = self.lc3(x)  # [2, 254, 20, 42]
        x = self.lc4(x)  # [2, 127, 19, 21]

        x = self.fla(x)
        x = self.l1(x)
        x = self.l2(x)

        return x
