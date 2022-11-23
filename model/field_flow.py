
# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/13 21:22
 @desc:
"""
import torch
import torch.nn as nn
from utils.weight_init import trunc_normal_
import modules.nn_lrp as nnlrp
import modules.layers_Chefer_H as lylrp
from modules.arcface import ArcFace
import einops


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class FieldFlow(nn.Module):

    def __init__(self, dim=None, num_heads=5, mlp_dilator=4, qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 t=512, n_signals=96, n_classes=40):
        super().__init__()
        self.n_classes = n_classes
        self.bs = None
        self.s = n_signals
        self.t_h = 55
        self.t = t
        self.d = dim or n_classes

        # [b, d=1, t=512, s=96]
        self.conv1 = lylrp.Conv2d(in_channels=1, out_channels=self.d, kernel_size=(15, 1), stride=(1, 1),
                                  dilation=1, bias=True)
        self.act_conv1 = lylrp.ELU()
        self.max_pool1 = lylrp.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1)
        self.norm1 = lylrp.BatchNorm2d(self.d)

        self.conv2 = lylrp.Conv2d(in_channels=self.d, out_channels=self.d, kernel_size=(15, 1), stride=(1, 1),
                                  groups=self.d, dilation=1, bias=True)
        self.act_conv2 = lylrp.ELU()
        self.max_pool2 = lylrp.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1)
        self.norm2 = lylrp.BatchNorm2d(self.d)

        self.conv3 = lylrp.Conv2d(in_channels=self.d, out_channels=self.d, kernel_size=(5, 1), stride=(1, 1),
                                  groups=self.d, dilation=1, bias=True)
        self.act_conv3 = lylrp.ELU()
        self.max_pool3 = lylrp.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1)
        self.norm3 = lylrp.LayerNorm(n_signals, eps=1e-6)

        # [b, d=40, t=128, s=96]
        # self.freqs_residue = nnlrp.Add()

        # Spacial Tokenization
        # rearrange [b, d=40, t=128, s=96] -> [(b, t), s, d]
        self.channel_token = nn.Parameter(torch.zeros(1, 1, self.d))  # [1, 1, d]
        self.ch_embed = nn.Parameter(torch.zeros(1, n_signals + 1, self.d))
        self.add1 = lylrp.Add()

        self.s_blocks = nn.ModuleList([
            nnlrp.Block(
                dim=self.d, num_heads=num_heads, mlp_dilator=mlp_dilator, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(3)])
        # self.ct_select = lylrp.IndexSelect()  # [bt, s+1, d] -> [bt, 1, d]
        #  squeeze [bt, 1, d] -> [bt, d]
        # rearrange [(bt), d] -> [b, t, d]
        # -------embedding  [b, t=128, s, d] -> [b, t, s, 1]
        # -------self.gap = lylrp.AdaptiveAvgPool2d(output_size=(n_signals, 1))

        # Temporal Tokenization
        self.temp_token = nn.Parameter(torch.zeros(1, 1, self.d))  # [1, 1, d]
        self.temp_embed = nn.Parameter(torch.zeros(1, 1+self.t_h, self.d))
        self.add2 = lylrp.Add()
        # self.attn_proj = nnlrp.MultiHeadAttention(dim_in=n_signals, dim_out=d * mlp_dilator, num_heads=8,
        #                             attn_drop=attn_drop_rate, proj_drop=drop_rate)  # [b, t, d*mlp_ratio]
        # self.fc_proj = nnlrp.Linear(in_features=d * mlp_dilator, out_features=d)
        # [b, t, d]

        self.t_blocks = nn.ModuleList([
            nnlrp.Block(
                dim=self.d, num_heads=num_heads, mlp_dilator=mlp_dilator, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(3)])
        # [b, t, d]

        # [b, t, d] -> [b, 1, d]
        # self.tt_select = lylrp.IndexSelect()
        # self.gap_logits = lylrp.AdaptiveAvgPool2d(output_size=(1, d))
        # squeeze [b, t, d] -> [b, d]
        # self.mlp_head = nnlrp.Mlp(in_features=self.d, hidden_features=self.d*mlp_dilator, out_features=self.d)
        # self.arc_margin = ArcFace(dim=self.d, num_classes=self.n_classes, requires_grad=True)

        trunc_normal_(self.channel_token, std=.02)
        trunc_normal_(self.temp_token, std=.02)
        trunc_normal_(self.ch_embed, std=.02)
        trunc_normal_(self.temp_embed, std=.02)
        self.apply(_init_weights)

        self.ct_select = lylrp.IndexSelect()
        self.tt_select = lylrp.IndexSelect()
        self.softmax = lylrp.Softmax(dim=1)
        self.inp_grad = None

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def forward(self, x, label=None):
        # [b, 1, t=512, s=96] if shenzhenface: [b, 1, 500, 128]

        x = self.conv1(x)
        x = self.act_conv1(x)
        x = self.max_pool1(x)  # [bs, d=128, t=256, s=96]
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.act_conv2(x)
        x = self.max_pool2(x)  # [bs, d=40, t=128, s=96]
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.act_conv3(x)
        x = self.max_pool3(x)  # [bs, d=40, t=128, s=96]
        x = self.norm3(x)
        # print(self.t_h, x.shape[2])
        assert self.t_h == x.shape[2]
        self.bs = x.shape[0]

        x.register_hook(self.save_inp_grad)

        # [bs, d=40, t=128, s=96] -> [(b, t), s, d]
        x = einops.rearrange(x, 'b d t s -> (b t) s d', b=self.bs, t=self.t_h, s=self.s)
        channel_tokens = self.channel_token.expand(self.bs*self.t_h, -1, -1)  # [1, 1, d] -> [bt, 1, d], d=c of conv
        x = torch.cat((channel_tokens, x), dim=1)  # [(b, t), 1, d] + [(b, t), s, d]  -> [(b, t), 1+s, d]
        ch_embed = self.ch_embed.expand(self.bs*self.t_h, -1, -1)
        x = self.add1([x, ch_embed])

        for blk in self.s_blocks:
            x = blk(x)  # [(b, t), 1+s, d]

        #  [bt, 1+s, d] -> [bt, 1, d] -> [bt, d]
        x = self.ct_select(inputs=x, dim=1, indices=torch.tensor(0, device=x.device)).squeeze(1)
        x = einops.rearrange(x, '(b t) d -> b t d', b=self.bs, t=self.t_h, d=self.d)

        temp_tokens = self.temp_token.expand(self.bs, -1, -1)  # [1, 1, d] -> [b, 1, d]
        x = torch.cat((temp_tokens, x), dim=1)  # [b, 1, c]+[b, t, d] -> [b, 1+t, d]
        temp_embed = self.temp_embed.expand(self.bs, -1, -1)  # [b, 1+t, d]
        x = self.add2([x, temp_embed])  # [b, 1+t, d]

        for blk2 in self.t_blocks:
            x = blk2(x)  # [b, 1+t, d]

        # [b, 1+t, d] -> [b, 1, d] -> [b, d]
        logits = self.tt_select(inputs=x, dim=1, indices=torch.tensor(0, device=x.device)).squeeze(1)
        # logits = self.mlp_head(logits)
        # logits = self.arc_margin(logits, label)  # [b, d] -> [b, c]
        logits = self.softmax(logits)
        return logits

    def relprop(self, cam=None, method="transformer_attribution", start_layer=0, **kwargs):
        # [1, classes]  b==1
        print("conservation 0", cam.sum())
        cam = cam.unsqueeze(1)  # [b, 1, d]
        # cam = self.gap_logits.relprop(cam, **kwargs)  # [b, _t, d]
        cam = self.tt_select.relprop(cam, **kwargs)  # [b, 1+t, d]

        b = cam.shape[0]

        cams = []
        for blk2 in reversed(self.t_blocks):
            cam = blk2.relprop(cam, **kwargs)  # [b, _t, d], note that _t = 1+t

        for blk2 in reversed(self.t_blocks):
            grad = blk2.attn.get_attn_gradients()  # [b, head, _t, _t]
            cam = blk2.attn.get_attn_cam()         # [b, head, _t, _t]
            cam = cam.reshape(b, -1, cam.shape[-1], cam.shape[-1])  # [b, head, _t, _t] ?????
            grad = grad.reshape(b, -1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam                       # [b, head, _t, _t]
            cam = cam.clamp(min=0).mean(dim=1)     # [b, _t, _t]
            cams.append(cam)          # [b, _t, _t]
        rollout = nnlrp.compute_rollout_attention(cams, start_layer=start_layer, true_bs=None)  # [b, _t, _t]
        cam = rollout[:, 0, 1:]  # [b, t]  temp taken added
        cam = torch.div(cam, cam.sum())

        cam = cam.unsqueeze(-1).expand(-1, -1, self.d)  # [b, t] -> [b, t, 1] -> [b, t, d]  cam=1*40
        cam = torch.div(cam, self.d)  # attention in temporary-dim, so divide it equally among d

        # (cam, _) = self.add2.relprop(cam, **kwargs)  # [b, _t, d], [b, _t, d]
        # cam = cam[:, 1:, :]  # concat [b, t, d]
        # print("conservation 1", cam.sum())

        # [(b, t), d] -> [b, t, d]
        cam = einops.rearrange(cam, 'b t d -> (b t) d', b=b, t=self.t_h, d=self.d)
        cam = cam.unsqueeze(1)   # [bt, 1, d]
        cam = self.ct_select.relprop(cam, **kwargs)  # [bt, 1+s, d]

        for blk in reversed(self.s_blocks):
            cam = blk.relprop(cam, **kwargs)   # [(b, t), 1+s, d] [128, 1+96, 40]

        bt = b*self.t_h
        cams1 = []
        for blk in reversed(self.s_blocks):
            grad = blk.attn.get_attn_gradients()  # [bt, head, 1+s, 1+s]
            cam = blk.attn.get_attn_cam()  # [bt, head, 1+s, 1+s]
            cam = cam.reshape(bt, -1, cam.shape[-1], cam.shape[-1])  # [bt, head, 1+s, 1+s] ?????
            grad = grad.reshape(bt, -1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam  # [bt, head, 1+s, 1+s]
            cam = cam.clamp(min=0).mean(dim=1)  # [bt, 1+s, 1+s]
            cams1.append(cam)  # [bt, 1+s, 1+s]
        rollout = nnlrp.compute_rollout_attention(cams1, start_layer=start_layer, true_bs=b)  # [bt, 1+s, 1+s]
        cam = rollout[:, 0, 1:]  # [bt, s] temp taken added
        cam = torch.div(cam, cam.sum())

        cam = cam.unsqueeze(-1).expand(-1, -1, self.d)  # [bt, s]->[bt, s, 1]->[bt, s, d] cam=1*40
        cam = torch.div(cam, self.d)  # attention on signal-dim, so divide it equally among d

        # (cam, _) = self.add1.relprop(cam, **kwargs)  # [bt, 1+s, d], [bt, 1+s, d]
        # cam = cam[:, 1:, :]  # concat [bt, s, d]

        cam = einops.rearrange(cam, '(b t) s d -> b d t s', b=b, t=self.t_h, s=self.s)

        print("conservation 2", cam.sum())    # =1
        cam = self.norm3.relprop(cam, **kwargs)
        cam = self.max_pool3.relprop(cam, **kwargs)
        cam = self.act_conv3.relprop(cam, **kwargs)
        cam = self.conv3.relprop(cam, **kwargs)
        cam = self.norm2.relprop(cam, **kwargs)
        cam = self.max_pool2.relprop(cam, **kwargs)
        cam = self.act_conv2.relprop(cam, **kwargs)
        cam = self.conv2.relprop(cam, **kwargs)
        cam = self.norm1.relprop(cam, **kwargs)
        cam = self.max_pool1.relprop(cam, **kwargs)
        cam = self.act_conv1.relprop(cam, **kwargs)
        cam = self.conv1.relprop(cam, **kwargs)
        print("conservation e", cam.sum())    # =1
        # print("min", cam.min())
        return cam
