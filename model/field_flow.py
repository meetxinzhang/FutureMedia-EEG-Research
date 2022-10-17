
# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/13 21:22
 @desc:
"""
import torch.nn as nn
from model.weight_init import trunc_normal_
import model.nn_lrp as nnlrp
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
                 batch_size=64, time=512, channels=96, n_classes=40):
        super().__init__()
        self.n_classes = n_classes
        self.bs = batch_size

        # [b, c=1, t=512, c=96]
        self.conv1 = nnlrp.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding='same',
                                  dilation=1, bias=True)
        self.max_pool1 = nnlrp.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1)
        self.conv2 = nnlrp.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=(1, 5), stride=(1, 1), padding='same',
                                  dilation=1, bias=True)
        self.max_pool2 = nnlrp.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1)
        self.act_conv = lylrp.Sigmoid()
        # [b, a=40, c=96, t=128]
        # self.freqs_residue = nnlrp.Add()

        # Spacial Tokenization
        # rearrange [b, a=40, c=96, t=128] -> [(b, t), c, a]

        self.s_blocks = nn.ModuleList([
            nnlrp.Block(
                dim=n_classes, num_heads=num_heads, mlp_dilator=mlp_dilator, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(2)])

        # embedding
        self.gap = lylrp.AdaptiveAvgPool2d(output_size=(batch_size * time, channels, 1))  # [(b, t), c, a] -> [(b, t), c, 1]
        # squeeze [(b, t), c, 1] -> [(b, t), c]

        # Temporal Tokenization
        # rearrange [(b, t), c] -> [b, t, c]

        self.attn_proj = nnlrp.MultiHeadAttention(dim_in=channels, dim_out=n_classes * mlp_dilator, num_heads=num_heads,
                                                  attn_drop=0, proj_drop=0)  # [b, t, n_classes*mlp_ratio]
        self.fc_proj = nnlrp.Linear(in_features=n_classes * mlp_dilator, out_features=n_classes)  # [b, t, n_classes]

        self.t_blocks = nn.ModuleList([
            nnlrp.Block(
                dim=n_classes, num_heads=num_heads, mlp_dilator=mlp_dilator, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(3)])
        # [b, t, n_classes]

        self.gap_logits = lylrp.AdaptiveAvgPool2d(output_size=(batch_size, 1, n_classes))  # [b, t, classes] -> [b, 1, classes]
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
        # x [b, c=96, t=512]
        x = self.conv1(x)
        x = self.act_conv(x)
        x = self.max_pool1(x)  # [bs, a=128, c=96, t=256]

        x = self.conv2(x)
        x = self.act_conv(x)
        x = self.max_pool2(x)  # [bs, a=40, c=96, t=128]
        t = x.shape[-1]

        x = einops.rearrange(x, 'b a c t -> (b t) c a')  # [b, a=40, c=96, t=128] -> [(b, t), c, a]
        x.register_hook(self.save_inp_grad)

        for blk in self.s_blocks:
            x = blk(x)  # [(b, t), c, a]

        x = self.gap(x).squeeze(1)  # [(b, t), c, a] -> [(b, t), c, 1] -> [(b, t), c]
        x = einops.rearrange(x, '(b t) c -> b t c', b=self.bs)  # [(b, t), c] -> [b, t, c]

        x = self.attn_proj(x)  # [b, t, c] -> [b, t, n_classes * mlp_dilator]
        x = self.fc_proj(x)    # [b, t, n_classes * mlp_dilator] -> [b, t, n_classes]

        # cls_tokens = self.cls_token.expand(B, -1, -1)  # [1, 1, embed_dim] -> [b, 1, embed_dim], embed_dim=c of conv
        # x = torch.cat((cls_tokens, x), dim=1)  # [b, 1, c'], [b, h'w', c']
        # x = self.add([x, self.pos_embed])  # [b, h'w'+1, c'] + [1, h'w'+1, c']

        for blk2 in self.t_blocks:
            x = blk2(x)  # [b, t, n_classes]

        logits = self.gap_logits(x).squeeze()  # [b, t, n_classes] -> [b, 1, n_classes] -> [b, n_classes]
        return logits

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = cam.unsqueeze(1)
        cam = self.gap_logits.relprop(cam, **kwargs)

        for blk2 in reversed(self.t_blocks):
            cam = blk2.relprop(cam, **kwargs)
        cams = []
        for blk2 in self.t_blocks:
            grad = blk2.attn.get_attn_gradients()
            cam = blk2.attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = nnlrp.compute_rollout_attention(cams, start_layer=start_layer)
        cam = rollout[:, 0, 1:]

        cam = self.fc_proj.relprop(cam, **kwargs)
        cam = self.attn_proj.relprop(cam, **kwargs)
        cam = einops.rearrange(cam, 'b t c -> (b t) c', b=self.bs)
        cam = cam.unsqueeze(2)
        cam = self.gap.relprop(cam, **kwargs)

        for blk in reversed(self.s_blocks):
            cam = blk.relprop(cam, **kwargs)
        cams = []
        for blk in self.s_blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = nnlrp.compute_rollout_attention(cams, start_layer=start_layer)
        cam = rollout[:, 0, 1:]

        cam = einops.rearrange(cam, '(b t) c a -> b a c t')
        cam = self.max_pool2.relprop(cam, **kwargs)
        cam = self.act_conv.relprop(cam, **kwargs)
        cam = self.conv2.relprop(cam, **kwargs)
        cam = self.max_pool1.relprop(cam, **kwargs)
        cam = self.act_conv.relprop(cam, **kwargs)
        cam = self.conv1.relprop(cam, **kwargs)
        # print("conservation 2", cam.sum())
        # print("min", cam.min())
        return cam
