# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/5 15:45
 @desc:
"""
import torch
from torch import nn, einsum
from einops import rearrange


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(q):
    """
    Converts the dimension that is specified from the axis
    from relative distances (with length 2*tokens-1) to absolute distance (length tokens)

    borrowed from lucidrains:
    https://github.com/lucidrains/bottleneck-transformer-pytorch/blob/main/bottleneck_transformer_pytorch/bottleneck_transformer_pytorch.py#L21

    Input: [bs, heads, t, 2*t - 1]
    Output: [bs, heads, t, t]
    """
    b, h, l, _, device, dtype = *q.shape, q.device, q.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim=3, k=h)
    return logits


class RelPosEmb2d(nn.Module):
    def __init__(
            self,
            fmap_size,
            dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x=h, y=w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h


def rel_pos_emb_1d(q, rel_emb, shared_heads):
    """
   Same functionality as RelPosEmb1D
   Args:
       q: a 4d tensor of shape [batch, heads, tokens, dim]
       rel_emb: a 2D or 3D tensor of shape [ 2*tokens-1 , dim] or [ heads, 2*tokens-1 , dim]
   """
    try:
        if shared_heads:
            emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
        else:
            emb = torch.einsum('b h t d, h r d -> b h t r', q, rel_emb)
    except RuntimeError as e:
        print('!!!\n  pos_embed: line 97. Please check the tokens and dim_head in RelPosEmb1DAISummer class object')
        print(e)
        emb = None
    return rel_to_abs(emb)


class RelPosEmb1DAISummer(nn.Module):
    def __init__(self, tokens, dim_head, heads=None):
        """
       Output: [batch head tokens tokens]
       Args:
           tokens: the number of the tokens of the seq
           dim_head: the size of the last dimension of q
           heads: if None representation is shared across heads.
           else the number of heads must be provided
       """
        super().__init__()
        scale = dim_head ** -0.5
        self.shared_heads = heads if heads is not None else True
        if self.shared_heads:
            self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim_head) * scale)  # [2*t-1, d]
        else:
            self.rel_pos_emb = nn.Parameter(torch.randn(heads, 2 * tokens - 1, dim_head) * scale)

    def forward(self, q):
        return rel_pos_emb_1d(q, self.rel_pos_emb, self.shared_heads)
