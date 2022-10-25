# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/18 22:40
 @name:
 @desc:
"""
import torch
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def ignite_relprop(model, x, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
    model.eval()
    logits = model(x)  # [b, c, h, w] -> [b, classes]
    kwargs = {"alpha": 1}
    if index is None:  # classificatory index
        index = np.argmax(logits.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, logits.size()[-1]), dtype=np.float32)  # [1, classes]
    one_hot[0, index] = 1  # [1, classes]
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits)  # classificatory mask, if b=1 then [1, classes]
    # TODO auto-validate per batch

    model.zero_grad()
    one_hot.backward(retain_graph=True)  # generate partial-gradients

    # the input of model.relprop() is one_hot
    return model.relprop(cam=torch.tensor(one_hot_vector).to(x.device), method=method, is_ablation=is_ablation,
                         start_layer=start_layer, **kwargs).detach()


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


# normalize = transforms.Normalize(mean=[0.5], std=[0.5])
# transform = transforms.Compose([
#     transforms.Resize(256),  # rescale the shorter-edge to 256, and keep the ratio of length/weight
#     transforms.CenterCrop(224),  # cut at center, when input integer then return square
#     transforms.ToTensor(),
#     normalize,
# ])


def generate_visualization(x, cam, save_name=None):
    # image = Image.fromarray(x)
    # image = transform(image)
    # torch.nn.functional.interpolate up-sampling, to re
    # cam = torch.nn.functional.interpolate(cam, scale_factor=16, mode='bilinear')
    cam = cam.data.cpu().numpy()
    # permute: trans dimension at original image.permute(1, 2, 0)
    x = x.data.cpu().numpy()
    # image + attribution
    img, heatmap, vis = add_cam_on_image(x, cam)

    if save_name is not None:
        path = './log/image/' + save_name

        vis = Image.fromarray(vis)
        vis.save(path + '_cam.jpg')

        img = Image.fromarray(img)
        img.save(path + '.jpg')

        heatmap = Image.fromarray(heatmap)
        heatmap.save(path + '_heatmap.jpg')

        print('saved ' + save_name)
    else:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[0].axis('off')
        axs[1].imshow(vis)
        axs[1].axis('off')
        plt.show()


def add_cam_on_image(x, cam):
    x = (x - x.min()) / (x.max() - x.min())
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img = cv2.cvtColor(np.array(np.uint8(255 * x)), cv2.COLOR_RGB2BGR)

    vis = np.float32(heatmap) / 255 + np.float32(img) / 255
    vis = vis / np.max(vis)
    vis = cv2.cvtColor(np.array(np.uint8(255 * vis)), cv2.COLOR_RGB2BGR)

    del x, cam
    return img, heatmap, vis
