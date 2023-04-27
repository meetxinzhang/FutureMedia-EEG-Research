# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/4/10 15:40
 @desc:
"""
import torch
from torch import nn
from modules.linear_conv2d import LinearConv2D
from modules.nn_lrp import Block
import einops


class FieldFlow1p2(nn.Module):
    def __init__(self, channels=30, electrodes=127, time=512, early_drop=0.3, late_drop=0.1):
        super().__init__()
        # [b f e t]
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels*4, kernel_size=(1, 1), groups=10, bias=False),
            nn.BatchNorm2d(num_features=channels * 4),
            nn.ELU(),

            nn.Conv2d(in_channels=channels*4, out_channels=channels*4, kernel_size=(1, 3), groups=3),
            nn.BatchNorm2d(num_features=channels*4),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=early_drop),

            nn.Conv2d(in_channels=channels * 4, out_channels=channels * 4, kernel_size=(1, 3), groups=1),
            nn.BatchNorm2d(num_features=channels * 4),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))
        )
        n_time = self.extra(torch.zeros(1, channels, electrodes, time)).contiguous().size()[-1]

        # [b f e t]
        self.ele_token = nn.Parameter(torch.zeros(1, 1, channels * 4))  # [1, 1, f]
        self.electro_tfs = nn.ModuleList([
            Block(tokens=electrodes+1, dim=channels*4, num_heads=12, mlp_dilator=2, rel_pos=True, drop=late_drop, attn_drop=0)
            for _ in range(2)])

        # b t f
        self.time_token = nn.Parameter(torch.zeros(1, 1, channels*4))  # [1, 1, f]
        self.time_tfs = nn.ModuleList([
            Block(tokens=n_time+1, dim=channels*4, num_heads=12, mlp_dilator=2, rel_pos=True, drop=late_drop, attn_drop=0)
            for _ in range(2)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=channels*4, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=40),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # [b f c t]
        x = self.extra(x)
        b, _, _, t = x.size()

        ele_tokens = self.ele_token.expand(b*t, -1, -1)
        x = einops.rearrange(x, "b f e t -> (b t) e f")
        x = torch.cat([ele_tokens, x], dim=1)  # [bt 1+e f]
        for blk in self.electro_tfs:
            x = blk(x)
        x = torch.select(x, dim=1, index=0).squeeze()  # bt 1 f -> bt f
        x = einops.rearrange(x, "(b t) f  -> b t f", b=b)

        time_tokens = self.time_token.expand(b, -1, -1)  # [1 1 f] -> [b 1 f]
        x = torch.cat((time_tokens, x), dim=1)  # [b 1+t f]
        for blk in self.time_tfs:
            x = blk(x)  # b t d
        x = torch.select(x, dim=1, index=0).squeeze()  # b 1 f -> b f

        return self.classifier(x)
