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
from modules.nn_lrp import Block
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
    def __init__(self, channels=127, early_drop=0.3, late_drop=0.1):
        super().__init__()
        self.c = channels

        self.lc1 = LinearConv2D(input_channels=channels, out_channels=254, groups=channels,
                                embedding=85, kernel_width=15, kernel_stride=3,
                                activate_height=2, activate_stride=2)
        self.lc2 = LinearConv2D(input_channels=254, out_channels=508, groups=channels,
                                embedding=42, kernel_width=7, kernel_stride=2,
                                activate_height=2, activate_stride=2)
        self.lc3 = LinearConv2D(input_channels=4, out_channels=8, groups=1,
                                embedding=21, kernel_width=7, kernel_stride=2,
                                activate_height=9, activate_stride=3)
        self.lc4 = LinearConv2D(input_channels=8, out_channels=16, groups=1,
                                embedding=5, kernel_width=5, kernel_stride=2,
                                activate_height=5, activate_stride=1)
        self.l1 = nn.Linear(in_features=336, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=40)
        self.fla = nn.Flatten(start_dim=1, end_dim=-1)
        self.drop1 = nn.Dropout(p=early_drop)
        self.drop2 = nn.Dropout(p=late_drop)

        # b c f t
        self.channel_token = nn.Parameter(torch.zeros(1, 1, 84))  # [1, 1, d]
        self.tf_blocks = nn.ModuleList([
            Block(tokens=128, dim=84, num_heads=4, mlp_dilator=2, rel_pos=True, drop=early_drop, attn_drop=0.1)
            for _ in range(1)])

        self.ch_embed = nn.Parameter(torch.zeros(1, channels, 84))
        # self.channel_token = nn.Parameter(torch.zeros(1, 1, self.d))  # [1, 1, d]

        self.apply(_init_weights)

    def forward(self, x):
        # [b c=127 f=85 t=500]
        x = self.lc1(x)  # [b 254 42 167]
        x = self.drop1(x)
        x = self.lc2(x)  # [b 508 21 83]
        x = self.drop1(x)
        [b, _, f, _] = x.size()

        _t = x.size()[-1]
        x = einops.rearrange(x, 'b (g d) f t -> (b t) g (f d)', g=self.c, d=4)  # [bt 127 84]
        channel_tokens = self.channel_token.expand(b*_t, -1, -1)  # [1 1 d] -> [bt 1 d]
        x = torch.cat((channel_tokens, x), dim=1)  # [bt 1 d] + [bt 127 d]  -> [bt 1+127 d]
        for blk in self.tf_blocks:
            x = blk(x)  # (b t) g (f d)

        # x = torch.index_select(x, dim=1, index=torch.LongTensor(range(0, self.c)).cuda())
        # x = einops.rearrange(x, '(b t) g (f d) -> b (g d) f t', b=b, t=_t, f=f, d=4)  # [b, 508, 21, 84]
        x = torch.select(x, dim=1, index=0).squeeze()  # (b t) 1->_ (f d)
        x = einops.rearrange(x, '(b t) (f d) -> b d f t', b=b, t=_t, f=f, d=4)  # [b, 4, 21, 84]

        x = self.lc3(x)  # [b, 8, 5, 42]
        x = self.drop2(x)
        x = self.lc4(x)  # [b, 16, 1, 21]

        x = self.fla(x)
        x = self.l1(self.drop2(x))
        x = self.l2(x)

        return x
