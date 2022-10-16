import argparse
import torch
import numpy as np
from numpy import *


class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)  # [b, c, h, w] -> [b, classes]
        kwargs = {"alpha": 1}
        if index is None:  # classificatory index
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)  # [1, classes]
        one_hot[0, index] = 1  # [1, classes]
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)  # classificatory mask, if b=1 then [1, classes]
        # TODO auto-validate per batch

        self.model.zero_grad()
        # backward to calculate gradients of all nodes /Deep Taylor Decomposition principle ?
        one_hot.backward(retain_graph=True)

        # the input of model.relprop() is one_hot
        return self.model.relprop(cam=torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)


# class Baselines:
#     def __init__(self, model):
#         self.model = model
#         self.model.eval()
#
#     def generate_cam_attn(self, input, index=None):
#         output = self.model(input.cuda(), register_hook=True)
#         if index is None:
#             index = np.argmax(output.cpu().data.numpy())
#
#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0][index] = 1
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         one_hot = torch.sum(one_hot.cuda() * output)
#
#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)
#         #################### attn
#         grad = self.model.blocks[-1].attn.get_attn_gradients()
#         cam = self.model.blocks[-1].attn.get_attention_map()
#         cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
#         grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
#         grad = grad.mean(dim=[1, 2], keepdim=True)
#         cam = (cam * grad).mean(0).clamp(min=0)
#         cam = (cam - cam.min()) / (cam.max() - cam.min())
#
#         return cam
#         #################### attn
#
#     def generate_rollout(self, input, start_layer=0):
#         self.model(input)
#         blocks = self.model.blocks
#         all_layer_attentions = []
#         for blk in blocks:
#             attn_heads = blk.attn.get_attention_map()
#             avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
#             all_layer_attentions.append(avg_heads)
#         rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
#         return rollout[:, 0, 1:]
