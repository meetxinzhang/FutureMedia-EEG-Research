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
        x = rearrange(x, 'b c m n t -> b c (m n) t')
        return x

    def relprop(self, cam, **kwargs):
        cam = rearrange(cam, 'b c (m n) t -> b c m n t', m=self.m_cache)
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
        self.clone = layers_lrp.Clone()
        self.conv1 = layers_lrp.Conv2d(in_channels=channels, out_channels=ffd_c // 2,
                                       kernel_size=(1, 3), stride=(1, 1), padding='same')
        self.conv2 = layers_lrp.Conv2d(in_channels=channels, out_channels=ffd_c // 2,
                                       kernel_size=(1, 5), stride=(1, 1), padding='same')
        self.cat = layers_lrp.Cat()
        self.bn = layers_lrp.BatchNorm2d(num_features=ffd_c)
        self.elu = layers_lrp.ELU()
        self.drop = layers_lrp.Dropout(drop)
        self.conv3 = layers_lrp.Conv2d(in_channels=ffd_c, out_channels=channels,
                                       kernel_size=(1, 1), stride=(1, 1), padding='same')

    def forward(self, x):
        # [b, c, p, t]
        [x1, x2] = self.clone(x, 2)
        x1 = self.conv1(x1)  # [b, E/2, p, t]
        x2 = self.conv2(x2)  # [b, E/2, p, t]
        x = self.cat((x1, x2), dim=1)  # [b, E, p, t]
        x = self.bn(x)
        x = self.elu(x)
        x = self.drop(x)
        x = self.conv3(x)  # [b, c, p, t]
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        # cam = self.drop.relprop(cam, **kwargs)  # just return the cam
        cam = self.conv3.relprop(cam, **kwargs)
        cam = self.elu.relprop(cam, **kwargs)  # just return the cam
        cam = self.bn.relprop(cam, **kwargs)
        [cam1, cam2] = self.cat.relprop(cam, **kwargs)
        cam1 = self.conv1.relprop(cam1, **kwargs)
        cam2 = self.conv2.relprop(cam2, **kwargs)
        cam = self.clone.relprop([cam1, cam2], **kwargs)
        return cam


class MHA(nn.Module):
    def __init__(self, channels, num_heads=8, drop=0.1):
        super().__init__()
        self.h = num_heads
        self.d = channels // num_heads
        # scale factor
        self.scale = self.d ** -0.5

        self.rel_pos_emb = RelPosEmb1DAISummer(tokens=50, dim_head=124, heads=None)  # print q for size at 90 line

        self.conv_qkv = layers_lrp.Conv2d(in_channels=channels, out_channels=3 * channels, kernel_size=(1, 1),
                                          stride=(1, 1))
        self.matmul1 = layers_lrp.einsum('bhpi,bhiq->bhpq')
        self.matmul2 = layers_lrp.einsum('bhpq,bhqi->bhpi')
        self.softmax = layers_lrp.Softmax(dim=-1)
        self.drop = layers_lrp.Dropout(drop)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def forward(self, x):
        # [b, c, p, t]
        qkv = self.conv_qkv(x)  # [b, c, p, t] -> [b, 3*c, p, t]
        q, k, v = rearrange(qkv, 'b (qkv h d) p t -> qkv b h d p t', qkv=3, h=self.h, d=self.d)
        q = rearrange(q, 'b h d p t -> b h p (d t)')
        k = rearrange(k, 'b h d p t -> b h (d t) p')
        v = rearrange(v, 'b h d p t -> b h p (d t)')
        self.save_v(v)

        # print(q.size(), 'q')  # print q for rel_pos_emb size
        dots = self.matmul1(q, k)  # [b, h, p, p]
        relative_position_bias = self.rel_pos_emb(q)  # [b, h, p, p],  q need [b, h, tokens, dim]
        dots += relative_position_bias

        dots *= self.scale  # [b, h, p, p]
        attn = self.softmax(dots)
        attn = self.drop(attn)

        self.save_attn(attn)  # save attn for rel_prop
        attn.register_hook(self.save_attn_gradients_hook)  # save attn gradients for model.rel_prop: attn_grad*cam

        out = self.matmul2(attn, v)  # [b, h, p, (dt)]
        out = rearrange(out, 'b h p (d t) -> b (h d) p t', h=self.h, d=self.d)
        return out

    def relprop(self, cam, **kwargs):
        cam = rearrange(cam, 'b (h d) p t -> b h p (d t)', h=self.h)
        [cam_attn, cam_v] = self.matmul2.relprop(cam, **kwargs)
        cam_attn /= 2
        cam_v /= 2
        self.save_attn_cam(cam_attn)
        self.save_v_cam(cam_v)

        # cam_attn = self.drop.relprop(cam_attn, **kwargs)
        cam_attn = self.softmax.relprop(cam_attn, **kwargs)
        [cam_q, cam_k] = self.matmul1.relprop(cam_attn, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_q = rearrange(cam_q, 'b h p (d t) -> b h d p t', d=self.d)
        cam_k = rearrange(cam_k, 'b h (d t) p -> b h d p t', d=self.d)
        cam_v = rearrange(cam_v, 'b h p (d t) -> b h d p t', d=self.d)
        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h d p t -> b (qkv h d) p t')
        cam = self.conv_qkv.relprop(cam_qkv, **kwargs)
        return cam

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients_hook(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients


class CTBlock(nn.Module):
    def __init__(self, channels, num_heads, ffd_c, drop=0.3):
        super().__init__()

        self.mha = MHA(channels=channels, num_heads=num_heads, drop=drop)
        self.bn1 = layers_lrp.BatchNorm2d(num_features=channels)
        self.cfe = CFE(channels=channels, ffd_c=ffd_c, drop=drop)
        self.bn2 = layers_lrp.BatchNorm2d(num_features=channels)

        self.add1 = layers_lrp.Add()
        self.add2 = layers_lrp.Add()
        self.clone1 = layers_lrp.Clone()
        self.clone2 = layers_lrp.Add()

    def forward(self, x):
        # [b, c, p=m*m, T]
        x1, x2 = self.clone1(x, 2)
        x = self.add1(self.mha(x1), x2)
        x = self.bn1(x)
        x1, x2 = self.clone2(x, 2)
        x = self.add2(self.cfe(x1), x2)
        x = self.bn2(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.bn2.relprop(cam, **kwargs)
        [cam1, cam2] = self.add2.relprop(cam, **kwargs)
        cam1 = self.cfe.relprop(cam1, **kwargs)
        cam = self.clone2.relprop([cam1, cam2], **kwargs)

        cam = self.bn1.relprop(cam, **kwargs)
        [cam1, cam2] = self.add1.relprop(cam, **kwargs)
        cam1 = self.mha.relprop(cam1, ** kwargs)
        cam = self.clone1.relprop([cam1, cam2], **kwargs)
        return cam


class ConvTransformer(nn.Module):
    def __init__(self, num_classes, in_channels, att_channels=8, num_heads=2,
                 ffd_channels=16, last_channels=64, size=32, T=500, depth=2, drop=0.2):
        super().__init__()
        self.last_c = last_channels
        self.lfe = LocFeaExtractor(in_channels=in_channels, hid_channels=att_channels)

        self.channel_token = nn.Parameter(torch.eye(n=att_channels, m=T // 2)).unsqueeze(0).unsqueeze(
            2)  # [1, c, 1, T/2]
        self.blocks = nn.ModuleList([
            CTBlock(channels=att_channels, num_heads=num_heads, ffd_c=ffd_channels, drop=drop)
            for _ in range(depth)])

        p = ((size - 8 + 2 * 0) // 4 + 1) ** 2

        self.conv1 = layers_lrp.Conv2d(in_channels=att_channels, out_channels=last_channels // 2,
                                       kernel_size=(p, 3), stride=(p, 1), padding=(0, 1))
        self.conv2 = layers_lrp.Conv2d(in_channels=att_channels, out_channels=last_channels // 2,
                                       kernel_size=(p, 5), stride=(p, 1), padding=(0, 2))
        # self.bn = layers_lrp.BatchNorm3d(num_features=1)
        self.bn = layers_lrp.BatchNorm2d(num_features=last_channels)
        self.elu = layers_lrp.ELU()
        self.pool = layers_lrp.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)

        self.l1 = layers_lrp.Linear(in_features=240, out_features=128)
        self.l2 = layers_lrp.Linear(in_features=128, out_features=num_classes)
        self.d1 = layers_lrp.Dropout(p=drop)
        self.d2 = layers_lrp.Dropout(p=drop)

        self.clone = layers_lrp.Clone()
        self.cat = layers_lrp.Cat()

    def forward(self, x):
        x = einops.rearrange(x, 'b t c w h -> b c w h t')
        # [b, 1, M, M, T]
        x = self.lfe(x)  # [b, c, p=m*m, T/2]
        # ct = self.channel_token.expand(x.size()[0], -1, -1, -1).to(x.device)  # [b, c, 1, T/2]
        # x = torch.cat((ct, x), dim=2)  # [b, c, 1+p, T/2]

        for blk in self.blocks:
            x = blk(x)  # [b, c, p, T/2]

        x1, x2 = self.clone(x, 2)
        x1 = self.conv1(x1)  # [b, F/2, 1, T/2]
        x2 = self.conv2(x2)  # [b, F/2, 1, T/2]
        x = self.cat((x1, x2), dim=1)  # [b, F, 1, T/2]
        # x = self.bn(x.unsqueeze(1)).squeeze()  # [b, F, T/2]
        x = self.bn(x).squeeze()  # [b f 1 T/2]
        x = self.elu(x)
        x = self.pool(x)  # [b, F, t]
        x = rearrange(x, 'b f t -> b (f t)')
        x = self.l1(self.d1(x))
        x = self.l2(self.d2(x))  # [b, classes]
        return x

    def relprop(self, cam, **kwargs):
        cam = self.l2.relprop(cam, **kwargs)
        cam = self.d2.relprop(cam, **kwargs)
        cam = self.l1.relprop(cam, ** kwargs)
        cam = self.d2.relprop(cam, ** kwargs)
        cam = rearrange(cam, 'b (f t) -> b f t', f=self.last_c)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.elu.relprop(cam, **kwargs)
        cam = cam.unsqueeze(2)  # [b f t] -> b f 1 t
        cam = self.bn.relprop(cam, ** kwargs)
        [cam1, cam2] = self.cat(cam, **kwargs)
        cam1 = self.conv1.relprop(cam1, **kwargs)
        cam2 = self.conv2.relprop(cam2, ** kwargs)
        cam = self.clone([cam1, cam2], **kwargs)

        for blk in self.blocks:
            cam = blk.relprop(cam, **kwargs)

        cams = []
        for blk in self.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        from modules.nn_lrp import compute_rollout_attention
        rollout = compute_rollout_attention(cams, start_layer=0)
        cam = rollout[:, 0, 1:]

        cam = self.lfe.relprop(cam, ** kwargs)

        return cam

