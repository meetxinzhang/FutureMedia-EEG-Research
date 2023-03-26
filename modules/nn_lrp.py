""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from einops import rearrange
from modules.layers_lrp import *
from utils.my_tools import to_2tuple
from utils.pos_embed import RelPosEmb1DAISummer


def compute_rollout_attention(all_layer_matrices, start_layer=0, true_bs=None):
    """input [layers, b, t, t]
    @true_bs: somtimes we integrate vit dimensions into batch_size, that lead to times addition of eye matrix,
    so we need to do eye/integrated-dim. integrated-dim=batch_size//true_bs, modified by Xin Zhang.
    layer-wise C=(A1*A2*A3*A4...An), and add one for each matrix adding residual consideration
    """
    num_tokens = all_layer_matrices[0].shape[1]  # t
    batch_size = all_layer_matrices[0].shape[0]  # b
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)  # [b, t, t]
    if true_bs is not None:
        eye = torch.div(eye, (batch_size//true_bs)**2)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]  # [l, b, t, t]
    all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)  # [l, b, t, t]
                          for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer + 1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)  # batch-size matrix multipy
    return joint_attention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class MultiHeadAttention(nn.Module):
    def __init__(self, tokens, dim_in, dim_out=None, num_heads=8, qkv_bias=False, rel_pos=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        dim_out = dim_out or dim_in
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5

        # print q for tokens when Runtime Error at pos_emb
        self.rel_pos_emb = RelPosEmb1DAISummer(tokens=tokens, dim_head=head_dim, heads=None) if rel_pos else None

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv_linear = Linear(dim_in, dim_out * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj_linear = Linear(dim_out, dim_out)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def forward(self, x):
        b, t, _, = x.shape  # [b, t, dim]
        qkv = self.qkv_linear(x)  # [b, t, dim_in] -> [b, t, dim_out*3]
        q, k, v = rearrange(qkv, 'b t (qkv h d) -> qkv b h t d', qkv=3, h=self.num_heads)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        if self.rel_pos_emb is not None:
            # print(q.size(), ' !!!!!!!! Please check the tokens number of pos_emb when Runtime Error')
            relative_position_bias = self.rel_pos_emb(q)  # [b, h, p, p],  q need [b, h, tokens, dim]
            dots += relative_position_bias
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)  # save attn for relprop
        attn.register_hook(self.save_attn_gradients_hook)  # save attn gradients for model.relprop: attn_grad*cam

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h t d -> b t (h d)')

        out = self.proj_linear(out)  # [b, t, dim_out=(h*d)] -> [b, t, dim_out]
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        # CAM(Class Activation Mapping)
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj_linear.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # A*V <- out
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2  # cam1 = cam1 / 2
        cam_v /= 2  # cam2 = cam2 / 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)  # [b, head, t, t]

        # Chefer H et al.
        # attn_grad = self.get_attn_gradients()  # [b, head, t, t]
        # cam1 = cam1 * attn_grad
        # t = cam1.shape[-1]    # num of tokens, it's time in my model
        # bs = cam1.shape[0]    # batch_size
        # head = cam1.shape[1]  # num of head
        # eye = torch.eye(t).expand(bs, head, t, t).to(cam1.device)
        # cam1 = cam1 + eye                    # [b, head, t, t]
        # cam1 = cam1.clamp(min=0).mean(dim=0)
        # cam1 = cam1 / cam1.sum(dim=-1, keepdims=True)
        # print('conservation 3', cam1.sum())

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)  # [b, head, t, t]

        # Q*K^T <- A
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)  # [b, head, t, dim/head=h_dim]
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h t d -> b t (qkv h d)', qkv=3, h=self.num_heads)
        # [b, t, dim*3]
        return self.qkv_linear.relprop(cam_qkv, **kwargs)  # [b, t, dim]

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


class Block(nn.Module):

    def __init__(self, tokens, dim, num_heads, mlp_dilator=4., qkv_bias=False, rel_pos=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(tokens=tokens, dim_in=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       rel_pos=rel_pos, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_dilator)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()  # res-connect
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)  # [b, t, dim] dim==classes or signals in my model
        cam2 = self.attn.relprop(cam2, **kwargs)         # [b, t, dim]
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # (224, 224)
        patch_size = to_2tuple(patch_size)  # (16, 16)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # embed_dim: num of conv kernels
        self.proj_conv = Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj_conv(x).flatten(2).transpose(1, 2)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1, 2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                          (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj_conv.relprop(cam, **kwargs)


#
# class VisionTransformer(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """
#
#     def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=1000, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0., attn_drop_rate=0.):
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with vit models
#
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches
#
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                 drop=drop_rate, attn_drop=attn_drop_rate)
#             for i in range(depth)])
#
#         self.norm = LayerNorm(embed_dim)
#         if mlp_head:
#             # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
#             self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
#         else:
#             # with a single Linear layer as head, the param count within rounding of paper
#             self.head = Linear(embed_dim, num_classes)
#
#         # FIXME not quite sure what the proper weight init is supposed to be,
#         # normal / trunc normal w/ std == .02 similar to vit Bert like transformers
#         trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
#         trunc_normal_(self.cls_token, std=.02)
#         self.apply(self._init_weights)
#
#         self.pool = IndexSelect()
#         self.add = Add()
#
#         self.inp_grad = None
#
#     def save_inp_grad(self, grad):
#         self.inp_grad = grad
#
#     def get_inp_grad(self):
#         return self.inp_grad
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @property
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}
#
#     def forward(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)  # [b, c, h, w] -> [b, c', h'w'] -> [b, h'w', c'] like [b, time, d]
#
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # [1, 1, embed_dim] -> [b, 1, embed_dim], embed_dim=c of conv
#         x = torch.cat((cls_tokens, x), dim=1)  # [b, 1, c'], [b, h'w', c']
#         x = self.add([x, self.pos_embed])  # [b, h'w'+1, c'] + [1, h'w'+1, c']
#
#         x.register_hook(self.save_inp_grad)
#
#         for blk in self.blocks:
#             x = blk(x)  # [b, h'w'+1, c']
#
#         x = self.norm(x)
#         x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))  # select the indices of dim 1 -> [b, 1, c']
#         x = x.squeeze(1)  # [b, c']
#         x = self.head(x)  # [b, classes]
#         return x
#
#     def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
#         # print(kwargs)
#         # print("conservation 1", cam.sum())
#         cam = self.head.relprop(cam, **kwargs)
#         cam = cam.unsqueeze(1)
#         cam = self.pool.relprop(cam, **kwargs)
#         cam = self.norm.relprop(cam, **kwargs)
#         for blk in reversed(self.blocks):
#             cam = blk.relprop(cam, **kwargs)
#
#         # print("conservation 2", cam.sum())
#         # print("min", cam.min())
#
#         if method == "full":
#             (cam, _) = self.add.relprop(cam, **kwargs)
#             cam = cam[:, 1:]
#             cam = self.patch_embed.relprop(cam, **kwargs)
#             # sum on channels
#             cam = cam.sum(dim=1)
#             return cam
#
#         elif method == "rollout":
#             # cam rollout
#             attn_cams = []
#             for blk in self.blocks:
#                 attn_heads = blk.attn.get_attn_cam().clamp(min=0)
#                 avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
#                 attn_cams.append(avg_heads)
#             cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
#             cam = cam[:, 0, 1:]
#             return cam
#
#         # our method, method name grad is legacy
#         elif method == "transformer_attribution" or method == "grad":
#             cams = []
#             for blk in self.blocks:
#                 grad = blk.attn.get_attn_gradients()
#                 cam = blk.attn.get_attn_cam()
#                 cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
#                 grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
#                 cam = grad * cam
#                 cam = cam.clamp(min=0).mean(dim=0)
#                 cams.append(cam.unsqueeze(0))
#             rollout = compute_rollout_attention(cams, start_layer=start_layer)
#             cam = rollout[:, 0, 1:]
#             return cam
#
#         elif method == "last_layer":
#             cam = self.blocks[-1].attn.get_attn_cam()
#             cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
#             if is_ablation:
#                 grad = self.blocks[-1].attn.get_attn_gradients()
#                 grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
#                 cam = grad * cam
#             cam = cam.clamp(min=0).mean(dim=0)
#             cam = cam[0, 1:]
#             return cam
#
#         elif method == "last_layer_attn":
#             cam = self.blocks[-1].attn.get_attn()
#             cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
#             cam = cam.clamp(min=0).mean(dim=0)
#             cam = cam[0, 1:]
#             return cam
#
#         elif method == "second_layer":
#             cam = self.blocks[1].attn.get_attn_cam()
#             cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
#             if is_ablation:
#                 grad = self.blocks[1].attn.get_attn_gradients()
#                 grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
#                 cam = grad * cam
#             cam = cam.clamp(min=0).mean(dim=0)
#             cam = cam[0, 1:]
#             return cam