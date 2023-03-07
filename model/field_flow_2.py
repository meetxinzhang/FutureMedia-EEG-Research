# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/6 13:06
 @desc:
   forked from https://github.com/MeetXinZhang/Spectrogram_frame-linear-network/blob/master/SFLN/linear_3d_layer.py
   https://doi.org/10.1016/j.ecoinf.2019.101009
"""

import torch
import torch.nn as nn
import einops


class LinearConv2DLayer(nn.Module):
    """
    see https://doi.org/10.1016/j.ecoinf.2019.101009 for the details
    """

    def __init__(self, input_channels, out_channels, groups, embedding,
                 kernel_width, kernel_stride, activate_height=2, activate_stride=2, bias=False):
        super().__init__()
        self.c = input_channels
        self.e = embedding
        self.w = kernel_width
        self.ks = kernel_stride
        self.g = groups
        assert input_channels % groups == 0
        kernel_depth = input_channels // groups
        self.d = kernel_depth
        self.o = out_channels
        assert out_channels % groups == 0
        n_filters1group = out_channels // groups
        self.n = n_filters1group
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, self.d, self.e, self.w))  # [o d e w]
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.add_bias = nn.Parameter(torch.empty(out_channels, self.d, self.e, self.w))
            nn.init.xavier_uniform_(self.add_bias)

        self.conv2d = nn.Conv2d(in_channels=self.d, out_channels=1,
                                kernel_size=(activate_height, self.w),
                                stride=(activate_stride, 1), padding='valid', bias=False).requires_grad_(False)
        nn.init.constant_(self.conv2d.weight, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        [b, c, f, t] = x.size()  # [b t c f]
        assert c == self.c
        assert f == self.e

        pad_1 = torch.zeros(b, c, f, self.w // 2).cuda()
        x = torch.concat([pad_1, x, pad_1], dim=-1)  # [b c f t++]

        x = x.unfold(dimension=-1, size=self.w, step=self.ks)  # [b c f t w]
        print(x.size(), 'tttttttt')
        x = einops.rearrange(x, 'b (g d) f t w -> b t g d f w', g=self.g, d=self.d)
        w = einops.rearrange(self.weight, '(g n) d f w -> g n d f w', g=self.g, n=self.n)
        if self.bias:
            bias = einops.rearrange(self.add_bias, '(g n) d f w -> g n d f w', g=self.g, n=self.n)

        try:
            x = x.unsqueeze(3).expand(-1, -1, -1, self.n, -1, -1, -1)  # [b t g 1->n d f w]
            x = torch.mul(x, w)  # [b t g n d f w] ele-wise* [g n d f w] with broadcast of low memory cost
            if self.bias:
                x = torch.add(x, bias)

        except MemoryError:
            print(' Out of memory, For loop replaced.')
            x = x.unsqueeze(3)  # [b t g 1 d f w]
            # [t g 1->n d f w] ele-wise* [g n d f w] -> [t g n d f w] with broadcast of low memory cost
            y = torch.mul(x[0].expand(-1, -1, self.n, -1, -1, -1), w)  # [t g n d f w]
            if self.bias:
                y = torch.add(y, bias)
            y.unsqueeze(0)  # [1 t g n d f w]
            for i in range(1, b):
                temp = torch.add(y, torch.mul(x[i].expand(-1, -1, self.n, -1, -1, -1), w))  # [t g n d f w]
                if self.bias:
                    temp = torch.add(y, bias)  # [t g n d f w]
                temp.unsqueeze(0)  # [1 t g n d f w]
                y = torch.cat((y, temp), dim=0)  # [1++ t g n d f w]
                del temp

        x = einops.rearrange(x, 'b t g n d f w -> b t (g n) d f w')  # [b t out_c d f w]
        print(x.size(), 'qqqqqqqq')
        x = einops.rearrange(x, 'b t o d f w -> (b t o) d f w')  # [m d f w]
        print(x.size(), 'aaa')
        x = self.conv2d(x)  # [m 1 1+f-height//s w/w]  [m 1 f' 1]
        x = self.relu(x).squeeze()  # [m f']
        print(x.size(), 'bbb')
        x = einops.rearrange(x, '(b t o) f -> b o f t', b=b, t=t, o=self.o)

        return x
