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
    if isinstance(m, LinearConv2D):
        nn.init.xavier_uniform_(m.weight)
        if isinstance(m, LinearConv2D) and m.add_bias is not None:
            nn.init.xavier_uniform_(m.add_bias, 0)
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

        self.lc1 = LinearConv2D(input_channels=channels, out_channels=channels*2, groups=channels,
                                embedding=30, kernel_width=15, kernel_stride=1,
                                activate_height=2, activate_stride=1)  # b o f t
        self.p1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))  # b o f/2 t/3

        self.lc2 = LinearConv2D(input_channels=channels*2, out_channels=channels*4, groups=channels,
                                embedding=15, kernel_width=7, kernel_stride=1,
                                activate_height=2, activate_stride=1)
        self.p2 = nn.AvgPool2d(kernel_size=(2, 3), stride=(2, 3))  # b o f/4 t/9

        self.lc3 = LinearConv2D(input_channels=channels*4, out_channels=channels*12, groups=channels,
                                embedding=8, kernel_width=5, kernel_stride=1,
                                activate_height=8, activate_stride=1, padding=False)  # b o 1 t/9
        self.p3 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))  # b o f/4 t/18

        self.l1 = nn.Linear(in_features=1152, out_features=256)
        self.l2 = nn.Linear(in_features=256, out_features=40)

        self.drop1 = nn.Dropout(p=early_drop)
        self.drop2 = nn.Dropout(p=late_drop)
        self.bn1 = nn.BatchNorm2d(num_features=channels*2)
        self.bn2 = nn.BatchNorm2d(num_features=channels*4)

        # b c f t
        self.time_token = nn.Parameter(torch.zeros(1, 1, channels*12))  # [1, 1, d]
        self.tf_blocks = nn.ModuleList([
            Block(tokens=56, dim=1152, num_heads=8, mlp_dilator=2, rel_pos=True, drop=late_drop, attn_drop=0)
            for _ in range(2)])

        self.apply(_init_weights)

    def forward(self, x):
        # Our:[b c=127 f=85 t=500]  Purdue:[b 96 30 1024]
        x = self.lc1(x)
        x = self.p1(x)  # [b 192 15 512]
        x = self.bn1(self.drop1(x))

        x = self.lc2(x)
        x = self.p2(x)  # [b 384 8 170]
        x = self.bn2(self.drop1(x))

        x = self.lc3(x)  # [b 1152 1 56]
        x = self.p3(x)
        x = self.drop1(x)

        x = x.squeeze()  # [b 1152 56]
        [b, _, _t] = x.size()

        # blank noise introduced if self-attend at electrode dimension.
        x = einops.rearrange(x, 'b (g d) t -> b t (g d)', g=self.c)  # [b 56 1152]

        time_tokens = self.time_token.expand(b, -1, -1)  # [1 1 d] -> [b 1 d]
        x = torch.cat((time_tokens, x), dim=1)  # [b 1 d] + [b 127 d]  -> [b 1+127 d]

        for blk in self.tf_blocks:
            x = blk(x)  # b t d
        x = torch.select(x, dim=1, index=0).squeeze()  # b 1 d -> b d

        x = self.l1(self.drop2(x))
        x = self.l2(x)

        return x
