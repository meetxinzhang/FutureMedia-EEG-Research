# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/24 9:50
 @name: 
 @desc:
"""
import numpy as np


def windows(length, window_size):
    start = 0
    i = 0
    while start < length:
        yield start, start + window_size, i
        start += int(window_size*0.5)
        i += 1


def adct_seqs(x, w=128):
    # [t=512, s=96]
    length = len(x)
    xs = []
    for start, end, i in windows(length, w):
        if np.shape(x[start: end, :])[0] < w:
            end = length
            start = end - w
            if start < 0:
                break
        xs.append(x[start: end, :])

        pass
    pass


def adct_img(x_batch):
    dct_00 = ((x_batch[:, :, ::2, ::2] + x_batch[:, :, 1::2, ::2]) +
              (x_batch[:, :, ::2, 1::2] + x_batch[:, :, 1::2, 1::2])) * 2
    dct_01 = ((x_batch[:, :, ::2, ::2] + x_batch[:, :, 1::2, ::2]) -
              (x_batch[:, :, ::2, 1::2] + x_batch[:, :, 1::2, 1::2])) * 2
    dct_10 = ((x_batch[:, :, ::2, ::2] - x_batch[:, :, 1::2, ::2]) +
              (x_batch[:, :, ::2, 1::2] - x_batch[:, :, 1::2, 1::2])) * 2
    dct_11 = ((x_batch[:, :, ::2, ::2] - x_batch[:, :, 1::2, ::2]) -
              (x_batch[:, :, ::2, 1::2] - x_batch[:, :, 1::2, 1::2])) * 2
    return dct_00, dct_01, dct_10, dct_11


if __name__ == "__main__":
    x_batch = np.random.random([3, 3, 16, 16])

    dct_00, dct_01, dct_10, dct_11 = adct_img(x_batch)
    print(dct_00, np.shape(dct_00))
