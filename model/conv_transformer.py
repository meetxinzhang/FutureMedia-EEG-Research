# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/3 19:11
 @desc:
"""

import torch
import torch.nn as nn
from einops import rearrange
from utils.pos_embed import RelPosEmb1DAISummer


class LocFeaExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c = channels
        self.conv3d_3 = nn.Conv3d(in_channels=1, out_channels=channels // 2, bias=True,
                                  kernel_size=(8, 8, 3), stride=(4, 4, 1), padding=(0, 0, 1))
        self.conv3d_5 = nn.Conv3d(in_channels=1, out_channels=channels // 2, bias=True,
                                  kernel_size=(8, 8, 5), stride=(4, 4, 1), padding=(0, 0, 2))

        self.bn = nn.BatchNorm3d(num_features=channels)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2), padding=0)

    def forward(self, x):
        # [b, 1,  M, M, T]
        x_3 = self.conv3d_3(x)  # [b, c/2, m, m, T]
        x_5 = self.conv3d_5(x)  # [b, c/2, m, m, T]
        x = torch.cat((x_3, x_5), dim=1)  # [b, c, m, m, T]

        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)  # [b, c, m, m, T/2]
        [b, _, _, _, t] = x.shape
        x = torch.reshape(x, [b, self.c, -1, t])  # [b, c, m*m, T/2]
        return x


class CFE(nn.Module):
    def __init__(self, channels, E=16, drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=E // 2,
                               kernel_size=(1, 3), stride=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=E // 2,
                               kernel_size=(1, 5), stride=(1, 1), padding='same')
        self.bn = nn.BatchNorm2d(num_features=E)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(drop)
        self.conv3 = nn.Conv2d(in_channels=E, out_channels=channels,
                               kernel_size=(1, 1), stride=(1, 1), padding='same')

    def forward(self, x):
        # [b, c, p, t]
        x1 = self.conv1(x)  # [b, E/2, p, t]
        x2 = self.conv2(x)  # [b, E/2, p, t]
        x = torch.cat((x1, x2), dim=1)  # [b, E, p, t]
        x = self.bn(x)
        x = self.elu(x)
        x = self.drop(x)
        x = self.conv3(x)  # [b, c, p, t]
        x = self.drop(x)
        return x


class MHA(nn.Module):
    def __init__(self, channels, num_heads=8, drop=0.3):
        super().__init__()
        self.h = num_heads
        self.d = channels // num_heads
        # scale factor
        self.scale = self.d ** -0.5

        self.conv_qkv = nn.Conv2d(in_channels=channels, out_channels=3 * channels, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(drop)

        self.rel_pos_emb = RelPosEmb1DAISummer(tokens=601, dim_head=1000, heads=None)  # print q for size

    def forward(self, x):
        # [b, c, p, t]
        qkv = self.conv_qkv(x)  # [b, c, p, t] -> [b, 3*c, p, t]
        q, k, v = rearrange(qkv, 'b (qkv h d) p t -> qkv b h d p t', qkv=3, h=self.h, d=self.d)
        q = rearrange(q, 'b h d p t -> b h p (d t)')
        k = rearrange(k, 'b h d p t -> b h (d t) p')
        v = rearrange(v, 'b h d p t -> b h p (d t)')

        # print(q.size())  # print q for rel_pos_emb size
        dots = torch.matmul(q, k)  # [b, h, p, p]
        relative_position_bias = self.rel_pos_emb(q)  # [b, h, p, p],  q need [b, h, tokens, dim]
        dots += relative_position_bias

        dots *= self.scale  # [b, h, p, p]
        attn = self.softmax(dots)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)  # [b, h, p, (dt)]
        out = rearrange(out, 'b h p (d t) -> b (h d) p t', h=self.h, d=self.d)
        return out


class CTBlock(nn.Module):
    def __init__(self, channels, num_heads, E, drop=0.3):
        super().__init__()
        self.mha = MHA(channels=channels, num_heads=num_heads, drop=drop)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.cfe = CFE(channels=channels, E=E, drop=drop)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        # [b, c, p=m*m, T]
        x = torch.add(self.mha(x), x)
        x = self.bn1(x)
        x = torch.add(self.cfe(x), x)
        x = self.bn2(x)
        return x


class ConvTransformer(nn.Module):
    def __init__(self, num_classes, channels=8, num_heads=2, E=16, F=64, size=32, T=500, depth=2, drop=0.2):
        super().__init__()
        self.lfe = LocFeaExtractor(channels=channels)

        self.channel_token = nn.Parameter(torch.eye(n=channels, m=T // 2)).unsqueeze(0).unsqueeze(2)  # [1, c, 1, T/2]
        self.blocks = nn.ModuleList([
            CTBlock(channels=channels, num_heads=num_heads, E=E, drop=drop)
            for _ in range(depth)])

        p = ((size - 8 + 2 * 0) // 4 + 1) ** 2
        t = T // 4

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=F // 2,
                               kernel_size=(p, 3), stride=(p, 1), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=F // 2,
                               kernel_size=(p, 5), stride=(p, 1), padding=(0, 2))
        self.bn = nn.BatchNorm3d(num_features=1)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.fla = nn.Flatten(start_dim=1, end_dim=-1)
        self.l1 = nn.Linear(in_features=48000, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=num_classes)
        self.d1 = nn.Dropout(p=drop)
        self.d2 = nn.Dropout(p=drop)

    def forward(self, x):
        # [b, 1, M, M, T]
        x = self.lfe(x)  # [b, c, p=m*m, T/2]
        ct = self.channel_token.expand(x.size()[0], -1, -1, -1).to(x.device)  # [b, c, 1, T/2]
        x = torch.cat((ct, x), dim=2)  # [b, c, 1+p, T/2]

        for blk in self.blocks:
            x = blk(x)  # [b, c, p, T/2]

        x1 = self.conv1(x)  # [b, F/2, 1, T/2]
        x2 = self.conv2(x)  # [b, F/2, 1, T/2]
        x = torch.cat((x1, x2), dim=1)  # [b, F, 1, T/2]
        x = self.bn(x.unsqueeze(1)).squeeze()  # [b, F, T/2]
        x = self.elu(x)
        x = self.pool(x)  # [b, F, t]
        x = self.fla(x)  # [b, F*t]
        x = self.l1(self.d1(x))
        x = self.l2(self.d2(x))  # [b, classes]
        return x
