# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/9/29 20:26
 @desc:
"""

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

from model.nn_lrp import vit_base_patch16_224 as vit
from model.nn_rel_generator import LRP
from data_pipeline.imagenet_class import CLS2IDX

# some image transform operators
normalize = transforms.Normalize(mean=[0.5], std=[0.5])
transform = transforms.Compose([
    transforms.Resize(256),  # rescale the shorter-edge to 256, and keep the ratio of length/weight
    transforms.CenterCrop(224),  # cut at center, when input integer then return square
    transforms.ToTensor(),
    normalize,
])


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


# initialize ViT pretrained
model = vit(pretrained=False).cuda()
model.eval()
attribution_generator = LRP(model)  # is a tensor flow graph


def generate_visualization(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(),
                                                                 method="transformer_attribution",
                                                                 index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    # torch.nn.functional.interpolate up-sampling, to re
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    # permute: trans dimension at original image
    _image = original_image.permute(1, 2, 0).data.cpu().numpy()
    _image = (_image - _image.min()) / (
            _image.max() - _image.min())
    # image + attribution
    vis = show_cam_on_image(_image, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)


# image = Image.open('data_pipeline/samples/catdog.png')
# dog_cat_image = transform(image)
from data_pipeline.mne_reader import read_auto
from PIL import Image
image = Image.fromarray(read_auto()[0])
eeg_image = transform(image)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(image)
axs[0].axis('off')

output = model(eeg_image.unsqueeze(0).cuda())
print_top_classes(output)

# cat - the predicted class
cat = generate_visualization(eeg_image)

# dog
# generate visualization for class 243: 'bull mastiff'
dog = generate_visualization(eeg_image, class_index=243)


axs[1].imshow(cat)
axs[1].axis('off')
axs[2].imshow(dog)
axs[2].axis('off')
plt.show()

