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


grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


class FieldFlow2(nn.Module):
    def __init__(self, channels=127, early_drop=0.3, late_drop=0.1):
        super().__init__()
        self.c = channels

        self.lc1 = LinearConv2D(input_channels=channels, out_channels=channels*4, groups=channels,
                                embedding=30, kernel_width=7, kernel_stride=1,
                                activate_height=2, activate_stride=2, padding=[3, 3, 1, 1])  # b o f t
        self.p1 = nn.AvgPool2d(kernel_size=(1, 6), stride=(1, 6))  # b o f/2 t/3

        # self.lc2 = LinearConv2D(input_channels=channels*2, out_channels=channels*4, groups=channels,
        #                         embedding=15, kernel_width=7, kernel_stride=1,
        #                         activate_height=3, activate_stride=1, padding=[3, 3, 1, 1])
        # self.p2 = nn.AvgPool2d(kernel_size=(2, 3), stride=(2, 3))  # b o f/4 t/9

        self.lc3 = LinearConv2D(input_channels=channels*4, out_channels=channels*12, groups=channels,
                                embedding=16, kernel_width=5, kernel_stride=1,
                                activate_height=16, activate_stride=1, padding=[2, 2, 0, 0])  # b o 1 t/9
        self.p3 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))  # b o f/4 t/18

        self.l1 = nn.Linear(in_features=1152, out_features=256)
        self.l2 = nn.Linear(in_features=256, out_features=40)

        self.drop1 = nn.Dropout(p=early_drop)
        self.drop2 = nn.Dropout(p=late_drop)
        self.bn1 = nn.BatchNorm2d(num_features=channels*4)
        # self.bn2 = nn.BatchNorm2d(num_features=channels*4)

        # b c f t
        self.time_token = nn.Parameter(torch.zeros(1, 1, channels*12))  # [1, 1, d]
        self.tf_blocks = nn.ModuleList([
            Block(tokens=57, dim=1152, num_heads=8, mlp_dilator=2, rel_pos=True, drop=late_drop, attn_drop=0)
            for _ in range(1)])

        self.apply(_init_weights)

    def forward(self, x):
        # Our:[b c=127 f=85 t=500]  Purdue:[b 96 30 1024]
        x = self.lc1(x)
        x = self.p1(x)  # [b 384 16 170]
        x = self.bn1(self.drop1(x))

        x = self.lc3(x)  # [b 1152 1 56]
        x = self.p3(x)
        x = self.drop1(x)

        x = x.squeeze()  # [b 768 56]
        b = x.size()[0]

        # blank noise introduced if self-attend at electrode dimension.
        x = einops.rearrange(x, 'b (g d) t -> b t (g d)', g=self.c)  # [b 56 1152]

        time_tokens = self.time_token.expand(b, -1, -1)  # [1 1 d] -> [b 1 d]
        x = torch.cat((time_tokens, x), dim=1)  # [b 1 d] + [b 127 d]  -> [b 1+127 d]

        # print(x.size(), 'tsfm')
        for blk in self.tf_blocks:
            x = blk(x)  # b t d
        x = torch.select(x, dim=1, index=0).squeeze()  # b 1 d -> b d

        x = self.l1(x)
        x = self.l2(self.drop2(x))

        return x
