# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/6 13:06
 @desc:
   forked from https://github.com/MeetXinZhang/Spectrogram_frame-linear-network/blob/master/SFLN/linear_conv3d_layer.py
   It was first proposed in my research https://doi.org/10.1016/j.ecoinf.2019.101009
"""

import torch
import torch.nn as nn
import einops


class LinearConv2D(nn.Module):
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

        self.add_bias = None
        if bias:
            self.add_bias = nn.Parameter(torch.empty(out_channels, self.d, self.e, self.w))
            nn.init.xavier_uniform_(self.add_bias)

        self.conv2d = nn.Conv2d(in_channels=self.d, out_channels=1,
                                kernel_size=(activate_height, self.w),
                                stride=(activate_stride, 1), padding='valid', bias=False).requires_grad_(False)
        nn.init.constant_(self.conv2d.weight, 1)
        self.bn = nn.BatchNorm3d(num_features=self.o, momentum=0.05)
        self.relu = nn.LeakyReLU()

    def _linear_mul_broadcasting(self, x, w, b=None):
        x = x.unsqueeze(-4).expand(-1, -1, -1, self.n, -1, -1, -1)  # [b t g 1->n d f w]
        if not self.bias:
            return torch.mul(x, w)  # [b t g n d f w] ele-wise* [g n d f w] with broadcast
        else:
            return torch.add(torch.mul(x, w), b)  # addition with broadcast

    def forward(self, x):
        [b, c, f, _] = x.size()
        assert c == self.c
        assert f == self.e

        # pad_width = ((t-1)*self.ks-t+self.w)  # calculate the padding
        pad = torch.zeros(b, c, f, self.w//2).cuda()
        x = torch.concat([pad, x, pad], dim=-1) # [b c f t++]

        x = x.unfold(dimension=-1, size=self.w, step=self.ks)  # [b c f t w]
        t = x.size(-2)
        x = einops.rearrange(x, 'b (g d) f t w -> b t g d f w', g=self.g, d=self.d)
        w = einops.rearrange(self.weight, '(g n) d f w -> g n d f w', g=self.g, n=self.n)
        the_b = einops.rearrange(self.add_bias, '(g n) d f w -> g n d f w', g=self.g, n=self.n) if self.bias else None

        try:
            y = self._linear_mul_broadcasting(x, w, b=the_b)
            del x

        except RuntimeError:
            print(' Out of memory, For loop ops replaced.')
            y = self._linear_mul_broadcasting(x[0].unsqueeze(0), w, b=the_b)
            mini_b = 4  # must keep b % mini_b = 0
            for i in range(1, b, mini_b):  # one or multi x depends on the memory
                temp = self._linear_mul_broadcasting(x[i:i+mini_b-1].unsqueeze(0), w, b=the_b)
                y = torch.cat((y, temp), dim=0)  # [1++ t g n d f w]
                del temp, x

        y = einops.rearrange(y, 'b t g n d f w -> b (g n) d f (t w)')  # [b out_c d f (t w)]
        y = self.bn(y)
        y = einops.rearrange(y, 'b o d f (t w) -> b o d f t w', t=t, w=self.w)  # [m d f w]
        y = einops.rearrange(y, 'b o d f t w-> (b o t) d f w')  # [m d f w]
        y = self.conv2d(y)  # [m 1 1+f-height//s w/w]  [m 1 f' 1]
        y = self.relu(y).squeeze(1).squeeze(-1)  # [m f']
        y = einops.rearrange(y, '(b o t) f -> b o f t', b=b, t=t, o=self.o)

        return y
