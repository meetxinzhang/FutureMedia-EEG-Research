
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
import einops


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class FieldFlow(nn.Module):

    def __init__(self, num_heads=6, mlp_dilator=4, qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 n_signals=96, n_classes=40):
        super().__init__()
        self.n_classes = n_classes
        self.bs = None
        self.s = n_signals
        self.t_h = None

        # [b, c=1, t=512, s=96]
        self.conv1 = nnlrp.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 1), stride=(1, 1), padding='same',
                                  dilation=1, bias=True)
        self.max_pool1 = nnlrp.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1)
        self.conv2 = nnlrp.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=(5, 1), stride=(1, 1), padding='same',
                                  dilation=1, bias=True)
        self.max_pool2 = nnlrp.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1)
        self.act_conv = lylrp.Sigmoid()
        # [b, c=40, t=128, s=96]
        # self.freqs_residue = nnlrp.Add()

        # Spacial Tokenization
        # rearrange [b, c=40, t=128, s=96] -> [(b, t), s, c]

        self.s_blocks = nn.ModuleList([
            nnlrp.Block(
                dim=n_classes, num_heads=5, mlp_dilator=mlp_dilator, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(2)])

        # rearrange [(b, t), s, c] -> [b, t, s, c]
        # embedding  [b, t=128, s, c] -> [b, t, s, 1]
        self.gap = lylrp.AdaptiveAvgPool2d(output_size=(n_signals, 1))
        # squeeze [b, t, s, 1] -> [b, t, s]

        # Temporal Tokenization
        self.attn_proj = nnlrp.MultiHeadAttention(dim_in=n_signals, dim_out=n_classes * mlp_dilator, num_heads=8,
                                                  attn_drop=attn_drop_rate, proj_drop=drop_rate)  # [b, t, n_classes*mlp_ratio]
        self.fc_proj = nnlrp.Linear(in_features=n_classes * mlp_dilator, out_features=n_classes)
        # [b, t, n_classes]

        self.t_blocks = nn.ModuleList([
            nnlrp.Block(
                dim=n_classes, num_heads=5, mlp_dilator=mlp_dilator, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(3)])
        # [b, t, n_classes]

        # [b, t, classes] -> [b, 1, classes]
        self.gap_logits = lylrp.AdaptiveAvgPool2d(output_size=(1, n_classes))
        # squeeze

        # if mlp_head:
        #     # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
        #     self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        # else:
        #     # with a single Linear layer as head, the param count within rounding of paper
        #     self.head = Linear(embed_dim, num_classes)
        #
        # # FIXME not quite sure what the proper weight init is supposed to be,
        # # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        # trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_weights)
        self.inp_grad = None

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def forward(self, x):
        # [b, 1, t=512, s=96]
        x = self.conv1(x)
        x = self.act_conv(x)
        x = self.max_pool1(x)  # [bs, c=128, t=256, s=96]

        x = self.conv2(x)
        x = self.act_conv(x)
        x = self.max_pool2(x)  # [bs, c=40, t=128, s=96]
        self.t_h = x.shape[2]
        self.bs = x.shape[0]

        x.register_hook(self.save_inp_grad)

        # [bs, c=40, t=128, s=96] -> [(b, t), s, c]
        x = einops.rearrange(x, 'b c t s -> (b t) s c', b=self.bs, t=self.t_h, s=self.s)
        for blk in self.s_blocks:
            x = blk(x)  # [(b, t), s, c]
        # [(b, t), s, c] -> [b, t, s, c]
        x = einops.rearrange(x, '(b t) s c -> b t s c', b=self.bs, t=self.t_h, s=self.s)

        x = self.gap(x).squeeze(-1)  # [b, t, s, c] -> [b, t, s, 1] -> [b, t, s]

        x = self.attn_proj(x)  # [b, t, s] -> [b, t, n_classes * mlp_dilator]
        x = self.fc_proj(x)    # [b, t, n_classes * mlp_dilator] -> [b, t, n_classes]

        # cls_tokens = self.cls_token.expand(B, -1, -1)  # [1, 1, embed_dim] -> [b, 1, embed_dim], embed_dim=c of conv
        # x = torch.cat((cls_tokens, x), dim=1)  # [b, 1, c'], [b, h'w', c']
        # x = self.add([x, self.pos_embed])  # [b, h'w'+1, c'] + [1, h'w'+1, c']

        for blk2 in self.t_blocks:
            x = blk2(x)  # [b, t, n_classes]

        logits = self.gap_logits(x).squeeze()  # [b, t, n_classes] -> [b, 1, n_classes] -> [b, n_classes]
        return logits

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # [1, classes]  b==1
        print("conservation start", cam.sum())
        cam = cam.unsqueeze(1)  # [b, 1, classes]
        cam = self.gap_logits.relprop(cam, **kwargs)  # [b, t, classes]
        b, t, classes = cam.shape

        cams = []
        for blk2 in reversed(self.t_blocks):
            cam = blk2.relprop(cam, **kwargs)  # [b, t, classes]

        for blk2 in reversed(self.t_blocks):
            grad = blk2.attn.get_attn_gradients()  # [b, head, t, t]
            cam = blk2.attn.get_attn_cam()         # [b, head, t, t]
            cam = cam.reshape(b, -1, cam.shape[-1], cam.shape[-1])  # [b, head, t, t] ?????
            grad = grad.reshape(b, -1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam                       # [b, head, t, t]
            cam = cam.clamp(min=0).mean(dim=1)     # [b, t, t]
            cams.append(cam)          # [b, t, t]
        rollout = nnlrp.compute_rollout_attention(cams, start_layer=start_layer)  # [b, t, t]
        cam = rollout[:, 0, :]                     # [b, t]  if class taken added then =rollout[:, 0, :]
        cam = cam.unsqueeze(-1).expand(b, t, classes)  # [b, t] -> [b, t, 1] -> [b, t, classes]  cam=1*40
        cam = torch.div(cam, classes)

        cam = self.fc_proj.relprop(cam, **kwargs)    # [b, t, n_classes] -> [b, t, n_classes * mlp_dilator]
        cam = self.attn_proj.relprop(cam, **kwargs)  # [b, t, n_classes * mlp_dilator] -> [b, t, signals]

        cam = cam.unsqueeze(-1)   # [b, t, s, 1]
        cam = self.gap.relprop(cam, **kwargs)  # [b, t, s ,channels_of_conv2]

        # [(b, t), s, c] -> [b, t, s, c]
        cam = einops.rearrange(cam, 'b t s c -> (b t) s c', b=self.bs, t=self.t_h, s=self.s)
        for blk in reversed(self.s_blocks):
            cam = blk.relprop(cam, **kwargs)   # [(b, t), s, c] [128, 96, 40]

        bt = cam.shape[0]
        cams1 = []
        for blk in reversed(self.s_blocks):
            grad = blk.attn.get_attn_gradients()  # [bt, head, s, s]
            cam = blk.attn.get_attn_cam()  # [bt, head, s, s]
            cam = cam.reshape(bt, -1, cam.shape[-1], cam.shape[-1])  # [bt, head, s, s] ?????
            grad = grad.reshape(bt, -1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam  # [bt, head, s, s]
            cam = cam.clamp(min=0).mean(dim=1)  # [bt, s, s]
            cams1.append(cam)  # [bt, s, s]
        rollout = nnlrp.compute_rollout_attention(cams1, start_layer=start_layer, true_bs=b)  # [bt, s, s]
        cam = rollout[:, 0, :]  # [bt, s]  if class taken added then =rollout[:, 0, :]
        print("conservation 2", cam.shape)
        cam = cam.unsqueeze(-1).expand(bt, self.s, classes)  # [bt, s] -> [bt, s, 1] -> [bt, s, classes]  cam=1*40
        cam = torch.div(cam, classes*t)

        # [(b, t), s, c] -> [bs, c=40, t=128, s=96]
        cam = einops.rearrange(cam, '(b t) s c -> b c t s', b=self.bs, t=self.t_h, s=self.s)
        cam = self.max_pool2.relprop(cam, **kwargs)
        cam = self.act_conv.relprop(cam, **kwargs)
        cam = self.conv2.relprop(cam, **kwargs)
        cam = self.max_pool1.relprop(cam, **kwargs)
        cam = self.act_conv.relprop(cam, **kwargs)
        cam = self.conv1.relprop(cam, **kwargs)
        print("conservation end", cam.sum())
        # print("min", cam.min())
        return cam
