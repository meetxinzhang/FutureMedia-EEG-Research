# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/18 22:40
 @name:
 @desc:
"""
import einops
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imageio
np.seterr(divide='ignore', invalid='ignore')


def ignite_relprop(model, x, device, index=None):
    assert x.shape[0] == 1  # [b=1, **x.shape]
    model.eval()
    x = x.to(device)
    y = model(x)  # [b, c, h, w] -> [b, classes]
    kwargs = {"alpha": 1}
    if index is None:  # classificatory index
        index = torch.argmax(y, dim=-1)
    one_hot = torch.nn.functional.one_hot(index, num_classes=y.shape[-1]).float().to(device)
    mask = one_hot.requires_grad_(True)
    mask = torch.sum(mask * y)  # classificatory mask, if b=1 then [1, classes], put sum() to play loss value
    # TODO auto-validate per batch
    model.zero_grad()
    mask.backward(retain_graph=True)  # generate partial-gradients
    return model.relprop(cam=one_hot.unsqueeze(0), **kwargs).detach()


def get_heatmap(cam, save_name=None, rgb=False):
    # [h w /c]
    cam = cam.data.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    if rgb:
        heatmap = np.uint8(255 * cam)
        heatmap = Image.fromarray(heatmap, mode="RGB")
    else:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = Image.fromarray(heatmap)
    if save_name is not None:
        heatmap.save('./log/image/' + save_name + '_heatmap.jpg')
        print('saved ' + save_name)
    return heatmap


def get_heatmap_gif(cam, save_name=None):
    # [t h w c]
    cam = cam.data.cpu().numpy()
    frames = []
    for c in cam:
        c = (c-c.min())/(c.max()-c.min())
        c = np.uint8(255 * c)
        c = Image.fromarray(c, mode="RGB")
        frames.append(c)
    imageio.mimsave('./log/image/' + save_name + '_heatmap.gif', frames)
    print('saved ' + save_name)


def get_heatmap_gallery(cam, x, save_name=None):
    # [t h w c] cpu
    x = einops.rearrange(x, 't c h w -> t h w c')
    line = torch.ones([24, 1, 20, 3])
    # x = (x - x.min()) / (x.max() - x.min())
    # cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = torch.cat([cam, line,  x], dim=1)

    cam = cam.data.numpy()  # cpu
    h = np.shape(cam)[1]
    interval = np.ones([h, 1, 3])
    figs = []
    for c in cam:
        # c = (c - c.min()) / (c.max() - c.min())
        c1 = c[:20, :, :]
        c2 = c[20:, :, :]
        c1 = (c1 - c1.min()) / (c1.max() - c1.min())
        c2 = (c2 - c2.min()) / (c2.max() - c2.min())
        c = np.concatenate([c1, c2], axis=0)
        c = np.concatenate([interval, c], axis=1)  # [h w c]
        c = np.uint8(255 * c)
        figs.append(c)
    figs = np.array(figs)
    figs = einops.rearrange(figs, 't h w c -> h (t w) c')
    figs = Image.fromarray(figs, mode="RGB")
    figs.save('./log/image/' + save_name + '_gallery.jpg')
    print('saved ' + save_name)


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
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(img)
        axs[0].axis('off')
        axs[1].imshow(heatmap)
        axs[1].axis('off')
        axs[2].imshow(vis)
        axs[2].axis('off')
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
