# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/27 11:41
 @name: 
 @desc: frequencies difference
"""
import numpy as np
import torch_dct as dct
import torch
import cv2


def delta(eeg, fold):
    # time-series [t, d]
    length = int(len(eeg) // fold)
    base = eeg[0:length - 1, :]
    a2 = eeg[2 * length:(length * 3) - 1, :]
    assert np.shape(base)[0] == np.shape(a2)[0]
    d = a2 - base
    del eeg, base, a2
    return d


def jiang_four_ave(eeg, fold):
    # [t, d]
    assert len(eeg) % fold == 0
    eeg = np.split(eeg, fold, axis=0)  # [fold length 96]
    eeg = np.average(eeg, axis=0)
    assert np.shape(eeg) == (512, 96)
    return eeg


def jiang_delta_ave(eeg):
    # [t d]
    assert len(eeg) % 4 == 0
    eeg = np.split(eeg, 4, axis=0)  # [fold t 96]
    base = eeg[3]
    d1 = eeg[0] - base
    d2 = eeg[1] - base
    d3 = eeg[2] - base

    re = (d1 + d2 + d3) / 3  # [t 96]

    assert np.shape(re) == (512, 96)
    return re


def dct_1d(eeg):
    # [t d]
    eeg = torch.from_numpy(eeg).T.cuda()  # [d=96 t=512]
    sf = dct.dct(eeg)  # tensor [96 512]
    fea_dim = sf.size()[-1]  # 512
    principle = fea_dim // 4  # 128
    fea_mask = torch.arange(1, fea_dim + 1)  # [512,]
    fea_mask = (fea_mask < principle).type(torch.uint8).to(sf.device)  # >高频  <低频
    # print(fea_mask)
    _sf = sf * fea_mask
    _s = dct.idct(_sf)  # [96 512]
    _s = _s.T  # [512 96]
    assert _s.size() == (512, 96)
    return _s.cpu().numpy()


def dct_2d(eeg):
    # https://zhuanlan.zhihu.com/p/114626779
    #  [t d]
    freqs = cv2.dct(eeg)
    # img_dct_log = 20 * np.log(abs(freqs))
    # 裁剪DCT
    for i in range(freqs.shape[0]):
        for j in range(freqs.shape[1]):
            if i > 24 or j > 128:
                freqs[i, j] = 0  # 裁剪的实质为像素置0
    return cv2.idct(freqs)


def approximated_dct(eeg):
    #  [t d]
    oo = ((eeg[::2, ::2] + eeg[1::2, ::2]) + (eeg[::2, 1::2] + eeg[1::2, 1::2])) * 2
    ol = ((eeg[::2, ::2] + eeg[1::2, ::2]) - (eeg[::2, 1::2] + eeg[1::2, 1::2])) * 2
    lo = ((eeg[::2, ::2] - eeg[1::2, ::2]) + (eeg[::2, 1::2] - eeg[1::2, 1::2])) * 2
    ll = ((eeg[::2, ::2] - eeg[1::2, ::2]) - (eeg[::2, 1::2] - eeg[1::2, 1::2])) * 2
    # [[oo, ol],
    #  [lo, ll]]
    assert np.shape(oo) == np.shape(ll) == (256, 48)
    return np.array([oo, ol, lo, ll])


def down_sample(eeg, ratio):
    return eeg[::ratio, :]


def trial_average(eeg, axis=0):
    #  [1000, 127]
    ave = np.mean(eeg, axis=axis)
    std = np.std(eeg, axis=axis)
    return (eeg - ave) / std


if __name__ == "__main__":
    x = np.random.random([2, 3, 2])
    print(trial_average(x, 1))
