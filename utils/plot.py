# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/4/6 16:15
 @desc:
"""
import pickle
import einops
import numpy as np
from PIL import Image, ImageSequence
from my_tools import file_scanf2
import imageio
import pandas as pd


# imageio.plugins.freeimage.download()


def paste_gif2(filepath):
    name = filepath.split('\\')[-1][:-4]
    img = Image.open(filepath)
    frames = []
    for i in ImageSequence.Iterator(img):
        i = i.convert('RGB')
        frames.append(np.array(i))
    frames = np.array(frames[1:], dtype=np.uint8)

    # [t h w c]
    h = np.shape(frames)[1]
    interval = np.ones([h, 1, 3])
    figs = frames[0]
    for c in frames:
        figs = np.concatenate([figs, c], axis=1)  # [h w c]

    imageio.mimsave('D:/GitHub/OptimusPrime/log/image/AEP-ConvTsfm-PD-gif/gif2gallery/' + name + '_gallery.jpg',
                    figs)

    # figs = Image.fromarray(figs, mode="RGB")
    # figs.save('D:/GitHub/OptimusPrime/log/image/AEP-ConvTsfm-PD-gif/gif2gallery/' + name + '_gallery.jpg')
    print('saved ' + name + '_gallery.jpg')


def paste_gif(filepath):
    name = filepath.split('\\')[-1][:-4]
    img = Image.open(filepath)

    p = Image.new(mode='RGB', size=(20 * 49, 20))

    for i, im in enumerate(ImageSequence.Iterator(img)):
        p.paste(im, box=(20 * i, 0))

    p.save('D:/GitHub/OptimusPrime/log/image/AEP-ConvTsfm-PD-gif/gif2gallery/' + name + '_gallery.jpg')


def smooth(csv_path, weight=0.85):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step', 'Value'],
                       dtype={'Step': np.int, 'Value': np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
    save.to_csv('smooth_' + csv_path)


def plot_aep(pkl_path):
    with open(pkl_path, 'rb') as f:
        aep = pickle.load(f)  # 50, 3, 20, 20
        _ = pickle.load(f)
    aep = einops.rearrange(aep, 't c h w -> t h w c')

    h = np.shape(aep)[2]
    interval = np.ones([h, 1, 3])
    figs = []
    for f in aep:
        f = (f - f.min()) / (f.max() - f.min())
        f = np.concatenate([interval, f], axis=1)  # [h w c]
        f = np.uint8(255 * f)
        figs.append(f)
    figs = np.array(figs)
    figs = einops.rearrange(figs, 't h w c -> h (t w) c')
    figs = Image.fromarray(figs, mode="RGB")
    figs.save('../log/image/' + 'aep_origin' + '_gallery.jpg')


if __name__ == '__main__':
    # path = 'D:/GitHub/OptimusPrime/log/image/AEP-ConvTsfm-PD-gif/'
    # files = file_scanf2(path, contains='AEP', endswith='.gif')
    # for f in files:
    #     paste_gif2(f)

    plot_aep('/data1/zhangwuxia/Datasets/SZEEG2022/pkl_trial_aep_color_subj1_05s/run_0_test1016-Fz-Fp1-seg-rmartifact_154_462501_1.pkl')
