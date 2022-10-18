# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/18 22:40
 @name: 
 @desc:
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


# normalize = transforms.Normalize(mean=[0.5], std=[0.5])
# transform = transforms.Compose([
#     transforms.Resize(256),  # rescale the shorter-edge to 256, and keep the ratio of length/weight
#     transforms.CenterCrop(224),  # cut at center, when input integer then return square
#     transforms.ToTensor(),
#     normalize,
# ])


def generate_visualization(x, cam, class_index=None):
    # image = Image.fromarray(x)
    # image = transform(image)
    # torch.nn.functional.interpolate up-sampling, to re
    # cam = torch.nn.functional.interpolate(cam, scale_factor=16, mode='bilinear')
    cam = cam.cuda().data.cpu().numpy()
    cam = (cam - cam.min()) / (
            cam.max() - cam.min())
    # permute: trans dimension at original image.permute(1, 2, 0)
    x = x.data.cpu().numpy()
    x = (x - x.min()) / (
            x.max() - x.min())
    # image + attribution
    img, vis = show_cam_on_image(x, cam)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[1].imshow(vis)
    axs[1].axis('off')
    plt.show()
    return vis


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = cv2.applyColorMap(np.uint8(255 * img), cv2.COLORMAP_JET)
    img = np.float32(img) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return img, cam
