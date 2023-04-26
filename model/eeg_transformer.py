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


class EEGTransformer(nn.Module):
    def __init__(self, in_channels=127, electrodes=96, early_drop=0.3, late_drop=0.1):
        super().__init__()
        self.c = in_channels

        # self.extra = nn.Sequential(
        #     LinearConv2D(input_channels=channels, out_channels=channels * 6, groups=channels,
        #                  embedding=30, kernel_width=1, kernel_stride=1,
        #                  activate_height=2, activate_stride=2, padding=[0, 0, 1, 1]),  # b o f t
        #     nn.BatchNorm2d(num_features=channels * 6),
        #     nn.ELU(),
        #     nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3)),  # b o f/2 t/3
        #     nn.Dropout(p=early_drop),
        #
        #     LinearConv2D(input_channels=channels * 6, out_channels=channels * 96, groups=channels,
        #                  embedding=16, kernel_width=3, kernel_stride=1,
        #                  activate_height=16, activate_stride=1, padding=[1, 1, 0, 0]),  # b o 1 t/9
        #     nn.BatchNorm2d(num_features=channels * 96),
        #     nn.ELU(),
        #     nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3)),
        #     nn.Dropout(p=early_drop)
        # )
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),  # left, right, top, bottom of 2D img
            nn.Conv2d(
                in_channels=in_channels,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 25),  # filter size
                # kernel_size=(1, 3),  # filter size 1111111111111 short T
                bias=False
            ),  # output shape (b, 8, C, T)
            # nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters  (16, C, T)
                kernel_size=(electrodes, 1),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            # nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.AvgPool2d((1, 2)),  # short T
            nn.Dropout(early_drop)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=32,  # num_filters
                kernel_size=(1, 15),  # filter size
                # kernel_size=(1, 3),  # short T
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=32,  # input shape (16, 1, T//4)
                out_channels=32,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(32),  # output shape (16, 1, T//4)
            nn.ELU(),
            # nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.AvgPool2d((1, 4)),  # short T
            nn.Dropout(early_drop)
        )

        # b c f t
        self.time_token = nn.Parameter(torch.zeros(1, 1, 32))  # [1, 1, d]
        self.tf_blocks = nn.ModuleList([
            Block(tokens=59, dim=32, num_heads=4, mlp_dilator=2, rel_pos=True, drop=late_drop, attn_drop=0)
            for _ in range(2)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1888, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=40),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Our:[b c=127 f=85 t=500]  Purdue:[b 96 30 1024]
        # x = self.extra(x)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        x = x.squeeze()  # [b 32 58]
        b = x.size()[0]

        # blank noise introduced if self-attend at electrode dimension.
        x = einops.rearrange(x, 'b c t -> b t c')  # [b 64 32]

        time_tokens = self.time_token.expand(b, -1, -1)  # [1 1 d] -> [b 1 d]
        x = torch.cat((time_tokens, x), dim=1)  # [b 1 d] + [b 127 d]  -> [b 1+127 d]

        # print(x.size(), 'tsfm')
        for blk in self.tf_blocks:
            x = blk(x)  # b t d
        # x = torch.select(x, dim=1, index=0).squeeze()  # b 1 d -> b d
        x = x.view(x.size(0), -1)
        return self.classifier(x)
