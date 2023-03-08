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
        else:
            self.add_bias = None

        self.conv2d = nn.Conv2d(in_channels=self.d, out_channels=1,
                                kernel_size=(activate_height, self.w),
                                stride=(activate_stride, 1), padding='valid', bias=False).requires_grad_(False)
        nn.init.constant_(self.conv2d.weight, 1)
        self.relu = nn.LeakyReLU()

    def _linear_conv_broadcasting(self, x, w, b=None):
        x = x.unsqueeze(-4).expand(-1, -1, -1, self.n, -1, -1, -1)  # [b t g 1->n d f w]
        if not self.bias:
            return torch.mul(x, w)  # [b t g n d f w] ele-wise* [g n d f w] with broadcast of low memory cost
        else:
            return torch.add(torch.mul(x, w), b)  # addition with broadcast

    def forward(self, x):
        [b, c, f, t] = x.size()  # [b t c f]
        assert c == self.c
        assert f == self.e

        pad_width = ((t-1)*self.ks-t+self.w)  # calculate the padding
        pad_1 = torch.zeros(b, c, f, pad_width//2).cuda()
        pad_2 = torch.zeros(b, c, f, (pad_width+1)//2).cuda()
        x = torch.concat([pad_1, x, pad_2], dim=-1)  # [b c f t++]

        x = x.unfold(dimension=-1, size=self.w, step=self.ks)  # [b c f t w]
        x = einops.rearrange(x, 'b (g d) f t w -> b t g d f w', g=self.g, d=self.d)
        w = einops.rearrange(self.weight, '(g n) d f w -> g n d f w', g=self.g, n=self.n)
        the_bias = None
        if self.bias:
            the_bias = einops.rearrange(self.add_bias, '(g n) d f w -> g n d f w', g=self.g, n=self.n)

        try:
            y = self._linear_conv_broadcasting(x, w, b=the_bias)
            del x

        except MemoryError:
            print(' Out of memory, For loop replaced.')
            y = self._linear_conv_broadcasting(x[0].unsqueeze(0), w, b=the_bias)
            for i in range(1, b):  # one or multi x depends on the memory
                temp = self._linear_conv_broadcasting(x[i].unsqueeze(0), w, b=the_bias)
                y = torch.cat((y, temp), dim=0)  # [1++ t g n d f w]
                del temp, x

        y = einops.rearrange(y, 'b t g n d f w -> b t (g n) d f w')  # [b t out_c d f w]
        y = einops.rearrange(y, 'b t o d f w -> (b t o) d f w')  # [m d f w]
        y = self.conv2d(y)  # [m 1 1+f-height//s w/w]  [m 1 f' 1]
        y = self.relu(y).squeeze()  # [m f']
        y = einops.rearrange(y, '(b t o) f -> b o f t', b=b, t=t, o=self.o)

        return y
