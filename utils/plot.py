# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/4/6 16:15
 @desc:
"""
import einops
import numpy as np
from PIL import Image, ImageSequence
from my_tools import file_scanf2
import imageio
imageio.plugins.freeimage.download()

path = 'D:/GitHub/OptimusPrime/log/image/AEP-ConvTsfm-PD-gif/'
files = file_scanf2(path, contains='AEP', endswith='.gif')



def seek_gif2(filepath):
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


def seek_gif(filepath):
    name = filepath.split('\\')[-1][:-4]
    img = Image.open(filepath)

    p = Image.new(mode='RGB', size=(20*49, 20))

    for i, im in enumerate(ImageSequence.Iterator(img)):
        p.paste(im, box=(20*i, 0))

    p.save('D:/GitHub/OptimusPrime/log/image/AEP-ConvTsfm-PD-gif/gif2gallery/' + name + '_gallery.jpg')


if __name__ == '__main__':
    for f in files:
        seek_gif2(f)
