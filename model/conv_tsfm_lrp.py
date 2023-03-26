# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/26 15:29
 @desc:
"""
import einops
import torch
import torch.nn as nn
from modules import layers_lrp
from einops import rearrange
from utils.pos_embed import RelPosEmb1DAISummer


class LocFeaExtractor(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.c = hid_channels
        self.m_cache = 32
        self.clone = layers_lrp.Clone()
        self.conv3d_3 = layers_lrp.Conv3d(in_channels=in_channels, out_channels=hid_channels // 2, bias=True,
                                          kernel_size=(8, 8, 3), stride=(4, 4, 1), padding=(0, 0, 1))
        self.conv3d_5 = layers_lrp.Conv3d(in_channels=in_channels, out_channels=hid_channels // 2, bias=True,
                                          kernel_size=(8, 8, 5), stride=(4, 4, 1), padding=(0, 0, 2))
        self.cat = layers_lrp.Cat()

        self.bn = layers_lrp.BatchNorm3d(num_features=hid_channels)
        self.elu = layers_lrp.ELU()
        self.pool = layers_lrp.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2), padding=0)

    def forward(self, x):
        # [b, 1, M, M, T]
        [x_3, x_5] = self.clone(x, 2)
        x_3 = self.conv3d_3(x_3)  # [b, c/2, m, m, T]
        x_5 = self.conv3d_5(x_5)  # [b, c/2, m, m, T]
        x = self.cat((x_3, x_5), dim=1)  # [b, c, m, m, T]

        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)  # [b, c, m, m, T/2]
        self.m_cache = x.shape[2]
        x = einops.rearrange(x, 'b c m n t -> b c (m n) t')
        return x

    def rel_prop(self, cam, **kwargs):
        cam = einops.rearrange(cam, 'b c (m n) t -> b c m n t', m=self.m_cache)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.elu.relprop(cam, **kwargs)
        cam = self.bn.relprop(cam, **kwargs)
        [cam3, cam5] = self.cat.relprop(cam, **kwargs)
        cam3 = self.conv3d_3.relprop(cam3, **kwargs)
        cam5 = self.conv3d_5.relprop(cam5, **kwargs)
        cam = self.clone.relprop([cam3, cam5], **kwargs)
        return cam


class CFE(nn.Module):
    def __init__(self, channels, ffd_c=16, drop=0.2):
        super().__init__()
        self.conv1 = layers_lrp.Conv2d(in_channels=channels, out_channels=ffd_c // 2,
                               kernel_size=(1, 3), stride=(1, 1), padding='same')
        self.conv2 = layers_lrp.Conv2d(in_channels=channels, out_channels=ffd_c // 2,
                               kernel_size=(1, 5), stride=(1, 1), padding='same')
        self.bn = layers_lrp.BatchNorm2d(num_features=ffd_c)
        self.elu = layers_lrp.ELU()
        self.drop = layers_lrp.Dropout(drop)
        self.conv3 = layers_lrp.Conv2d(in_channels=ffd_c, out_channels=channels,
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

    def rel_prop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.conv3.relprop(cam, **kwargs)
        


class MHA(nn.Module):
    def __init__(self, channels, num_heads=8, drop=0.3):
        super().__init__()
        self.h = num_heads
        self.d = channels // num_heads
        # scale factor
        self.scale = self.d ** -0.5

        self.conv_qkv = layers_lrp.Conv2d(in_channels=channels, out_channels=3 * channels, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = layers_lrp.Softmax(dim=-1)
        self.drop = layers_lrp.Dropout(drop)

        self.rel_pos_emb = RelPosEmb1DAISummer(tokens=50, dim_head=124, heads=None)  # print q for size at 90 line

    def forward(self, x):
        # [b, c, p, t]
        qkv = self.conv_qkv(x)  # [b, c, p, t] -> [b, 3*c, p, t]
        q, k, v = rearrange(qkv, 'b (qkv h d) p t -> qkv b h d p t', qkv=3, h=self.h, d=self.d)
        q = rearrange(q, 'b h d p t -> b h p (d t)')
        k = rearrange(k, 'b h d p t -> b h (d t) p')
        v = rearrange(v, 'b h d p t -> b h p (d t)')

        # print(q.size(), 'q')  # print q for rel_pos_emb size
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
    def __init__(self, channels, num_heads, ffd_c, drop=0.3):
        super().__init__()
        self.mha = MHA(channels=channels, num_heads=num_heads, drop=drop)
        self.bn1 = layers_lrp.BatchNorm2d(num_features=channels)
        self.cfe = CFE(channels=channels, ffd_c=ffd_c, drop=drop)
        self.bn2 = layers_lrp.BatchNorm2d(num_features=channels)

    def forward(self, x):
        # [b, c, p=m*m, T]
        x = torch.add(self.mha(x), x)
        x = self.bn1(x)
        x = torch.add(self.cfe(x), x)
        x = self.bn2(x)
        return x


class ConvTransformer(nn.Module):
    def __init__(self, num_classes, in_channels, hid_channels=8, num_heads=2,
                 ffd_channels=16, deep_channels=64, size=32, T=500, depth=2, drop=0.2):
        super().__init__()
        self.lfe = LocFeaExtractor(in_channels=in_channels, hid_channels=hid_channels)

        self.channel_token = nn.Parameter(torch.eye(n=hid_channels, m=T // 2)).unsqueeze(0).unsqueeze(
            2)  # [1, c, 1, T/2]
        self.blocks = nn.ModuleList([
            CTBlock(channels=hid_channels, num_heads=num_heads, ffd_c=ffd_channels, drop=drop)
            for _ in range(depth)])

        p = ((size - 8 + 2 * 0) // 4 + 1) ** 2
        t = T // 4

        self.conv1 = layers_lrp.Conv2d(in_channels=hid_channels, out_channels=deep_channels // 2,
                               kernel_size=(p, 3), stride=(p, 1), padding=(0, 1))
        self.conv2 = layers_lrp.Conv2d(in_channels=hid_channels, out_channels=deep_channels // 2,
                               kernel_size=(p, 5), stride=(p, 1), padding=(0, 2))
        self.bn = layers_lrp.BatchNorm3d(num_features=1)
        self.elu = layers_lrp.ELU()
        self.pool = layers_lrp.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        # TODO  layers_lrp of Flatten
        self.fla = nn.Flatten(start_dim=1, end_dim=-1)
        self.l1 = layers_lrp.Linear(in_features=240, out_features=128)
        self.l2 = layers_lrp.Linear(in_features=128, out_features=num_classes)
        self.d1 = layers_lrp.Dropout(p=drop)
        self.d2 = layers_lrp.Dropout(p=drop)

    def forward(self, x):
        x = einops.rearrange(x, 'b t c w h -> b c w h t')
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
