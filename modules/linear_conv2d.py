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
import gc


class LinearConv2D(nn.Module):
    """
    see https://doi.org/10.1016/j.ecoinf.2019.101009 for the details
    """

    def __init__(self, input_channels, out_channels, groups, embedding,
                 kernel_width, kernel_stride, activate_height=2, activate_stride=2, padding=True, bias=False):
        super().__init__()
        self.c = input_channels
        self.e = embedding
        self.w = kernel_width
        self.ks = kernel_stride
        self.ah = activate_height
        self.g = groups
        assert input_channels % groups == 0
        kernel_depth = input_channels // groups
        self.d = kernel_depth
        self.o = out_channels
        assert out_channels % groups == 0
        n_filters1group = out_channels // groups
        self.n = n_filters1group
        self.bias = bias
        self.padding = padding

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
        if self.padding:
            pad1 = torch.zeros(b, c, f, self.w // 2).cuda()
            x = torch.concat([pad1, x, pad1], dim=-1)  # [b c f t++]

        x = x.unfold(dimension=-1, size=self.w, step=self.ks)  # [b c f t w]
        t = x.size(-2)
        x = einops.rearrange(x, 'b (g d) f t w -> b t g d f w', g=self.g, d=self.d)
        w = einops.rearrange(self.weight, '(g n) d f w -> g n d f w', g=self.g, n=self.n)
        the_b = einops.rearrange(self.add_bias, '(g n) d f w -> g n d f w', g=self.g, n=self.n) if self.bias else None

        try:
            y = self._linear_mul_broadcasting(x, w, b=the_b)
            y = einops.rearrange(y, 'b t g n d f w -> b t (g n) d f w')  # [b t out_c d f w]
            y = einops.rearrange(y, 'b t o d f w -> (b t o) d f w')  # [m d f w]

            if self.padding:
                pad2 = torch.zeros(y.shape[0], self.d, self.ah//2, self.w).to(y.device)
                y = torch.concat([pad2, y, pad2], dim=-2)  # [m d f++ w]

            y = self.conv2d(y)  # [m 1 f w/w]  [m 1 f 1]
            y = self.relu(y).squeeze(1).squeeze(-1)  # [m f]
            y = einops.rearrange(y, '(b t o) f -> b o f t', b=b, t=t, o=self.o)

        except RuntimeError:  # Out of memory, For loop ops replaced.
            gc.collect()
            mini_b = 4  # must keep b % mini_b = 0
            x = [e for e in torch.split(x, mini_b, dim=0)]  # list(mini_b, t g d f w)
            temp = x.pop()
            y = self._linear_mul_broadcasting(temp, w, b=the_b)
            del temp
            gc.collect()
            y = einops.rearrange(y, 'b t g n d f w -> b t (g n) d f w')  # [b t out_c d f w]
            y = einops.rearrange(y, 'b t o d f w -> (b t o) d f w')  # [m d f w]
            if self.padding:
                pad2 = torch.zeros(y.shape[0], self.d, self.ah//2, self.w).to(y.device)
                y = torch.concat([pad2, y, pad2], dim=-2)  # [m d f++ w]
            y = self.conv2d(y)  # [m 1 f w/w]  [m 1 f 1]
            y = self.relu(y).squeeze(1).squeeze(-1)  # [m f]
            y = einops.rearrange(y, '(b t o) f -> b o f t', b=mini_b, t=t, o=self.o)

            rest = len(x)
            for _ in range(rest):  # one or multi x depends on the memory
                temp = x.pop()
                temp = self._linear_mul_broadcasting(temp, w, b=the_b)
                temp = einops.rearrange(temp, 'b t g n d f w -> b t (g n) d f w')  # [b t out_c d f w]
                temp = einops.rearrange(temp, 'b t o d f w -> (b t o) d f w')  # [m d f w]
                if self.padding:
                    pad2 = torch.zeros(temp.shape[0], self.d, self.ah // 2, self.w).to(y.device)
                    temp = torch.concat([pad2, temp, pad2], dim=-2)  # [m d f++ w]
                temp = self.conv2d(temp)  # [m 1 f w/w]  [m 1 f 1]
                temp = self.relu(temp).squeeze(1).squeeze(-1)  # [m f]
                temp = einops.rearrange(temp, '(b t o) f -> b o f t', b=mini_b, t=t, o=self.o)

                y = torch.cat((y, temp), dim=0)  # [mini_b++ o f t]
                del temp
                gc.collect()

        # if self.padding:
        #     pad2 = torch.zeros(b, self.d, f//2, self.w).cuda()
        #     print(y.size(), pad2.size(), 'qqqqq')
        #     y = torch.concat([pad2, y, pad2], dim=-2)  # [m d f++ w]
        #
        # y = self.conv2d(y)  # [m 1 f w/w]  [m 1 f 1]
        # y = self.relu(y).squeeze(1).squeeze(-1)  # [m f]
        # y = einops.rearrange(y, '(b t o) f -> b o f t', b=b, t=t, o=self.o)
        del x
        gc.collect()
        return y
