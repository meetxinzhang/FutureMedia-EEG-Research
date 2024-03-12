# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/8 18:29
 @desc:
"""

import torch
from torch import nn
import torch_dct as dct

from modules.linear_conv2d import LinearConv2D
from modules.nn_lrp import Block
import einops
from einops.layers.torch import Rearrange as EinRearrange


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

        self.block_1 = nn.Sequential(
            nn.BatchNorm2d(1),  # output shape (8, C, T)
            nn.ZeroPad2d((1, 2, 0, 0)),  # left, right, top, bottom of 2D img
            nn.Conv2d(
                in_channels=in_channels,  # input shape (1, C, T)
                out_channels=32,  # num_filters
                kernel_size=(1, 8),  # filter size
                stride=(1, 8),
                bias=False
            ),  # output shape (b, 8, C, T)
            # nn.AvgPool2d(kernel_size=(1, 2),  stride=(1, 2)),  # output shape (16, 1, T//4)
            # nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # self.extra = nn.Sequential(
        #     LinearConv2D(input_channels=self.c, out_channels=self.c * 6, groups=self.c,
        #                  embedding=16, kernel_width=1, kernel_stride=1,
        #                  activate_height=1, activate_stride=2, padding=[0, 0, 1, 1]),  # b o f t
        #     nn.BatchNorm2d(num_features=self.c * 6),
        #     nn.ELU(),
        #     nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3)),  # b o f/2 t/3
        #     nn.Dropout(p=early_drop),
        #
        #     LinearConv2D(input_channels=self.c * 6, out_channels=self.c * 96, groups=self.c,
        #                  embedding=16, kernel_width=3, kernel_stride=1,]),  # b o 1 t/9
        #     nn.BatchNorm2d(num_features=self.c * 96),
        #                  activate_height=16, activate_stride=1, padding=[1, 1, 0, 0

        #     nn.ELU(),
        #     nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3)),
        #     nn.Dropout(p=early_drop)
        # )

        self.mapping = nn.Sequential(
            EinRearrange('b c e t -> b t c e'),
            nn.Flatten(start_dim=2, end_dim=-1),
            nn.Dropout(early_drop),
            nn.Linear(in_features=3072, out_features=512),
            nn.ReLU(),
            nn.Dropout(early_drop),
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        # self.block_2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=8,  # input shape (8, C, T)
        #         out_channels=16,  # num_filters  (16, C, T)
        #         kernel_size=(electrodes, 1),  # filter size
        #         groups=8,
        #         bias=False
        #     ),  # output shape (16, 1, T)
        #     nn.BatchNorm2d(16),  # output shape (16, 1, T)
        #     nn.ELU(),
        #     # nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
        #     nn.AvgPool2d((1, 2)),  # short T
        #     nn.Dropout(early_drop)  # output shape (16, 1, T//4)
        # )

        # self.block_3 = nn.Sequential(
        #     nn.ZeroPad2d((1, 2, 0, 0)),
        #     nn.Conv2d(
        #         in_channels=16,  # input shape (16, 1, T//4)
        #         out_channels=32,  # num_filters
        #         kernel_size=(1, 15),  # filter size
        #         # kernel_size=(1, 3),  # short T
        #         groups=16,
        #         bias=False
        #     ),  # output shape (16, 1, T//4)
        #     nn.Conv2d(
        #         in_channels=32,  # input shape (16, 1, T//4)
        #         out_channels=32,  # num_filters
        #         kernel_size=(1, 1),  # filter size
        #         bias=False
        #     ),  # output shape (16, 1, T//4)
        #     nn.BatchNorm2d(32),  # output shape (16, 1, T//4)
        #     nn.ELU(),
        #     # nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
        #     nn.AvgPool2d((1, 4)),  # short T
        #     nn.Dropout(early_drop)
        # )

        # b c f t
        self.layer_norm = nn.LayerNorm(512)
        self.time_token = nn.Parameter(torch.zeros(1, 1, 512))  # [1, 1, d]
        self.tf_blocks = nn.ModuleList([
            Block(tokens=65, dim=512, num_heads=8, mlp_dilator=1, rel_pos=True, drop=late_drop, attn_drop=0.1)
            for _ in range(3)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=40),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # [b 1 c t] needed
        # x = self.extra(x)
        # x = dct.dct(x)

        x = self.block_1(x)  # [b, c, e, t]
        # x = self.block_2(x)
        # x = self.block_3(x)
        # print(x.shape, 'conv')

        x = self.mapping(x)  # [b, t, c]
        x = self.layer_norm(x)
        b = x.size()[0]
        # blank noise introduced if self-attend at electrode dimension.
        time_tokens = self.time_token.expand(b, -1, -1)  # [1 1 d] -> [b 1 d]
        x = torch.cat((time_tokens, x), dim=1)  # [b 1 d] + [b 127 d]  -> [b 1+127 d]

        # print(x.shape, 'cat')
        for blk in self.tf_blocks:
            x = blk(x)  # b t d
        x = torch.select(x, dim=1, index=0).squeeze()  # b 1 d -> b d
        x = x.view(b, -1)
        return self.classifier(x)
