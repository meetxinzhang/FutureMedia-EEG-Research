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


def delta(series, fold):
    # time-series [t, d]
    length = int(len(series) // fold)
    base = series[0:length - 1, :]
    a2 = series[2*length:(length * 3) - 1, :]
    assert np.shape(base)[0] == np.shape(a2)[0]
    d = a2 - base
    del series, base, a2
    return d


def jiang_four_ave(series, fold):
    # [t, d]
    assert len(series) % fold == 0
    series = np.split(series, fold, axis=0)  # [fold length 96]
    series = np.average(series, axis=0)
    assert np.shape(series) == (512, 96)
    return series


def jiang_delta_ave(series):
    # [t d]
    assert len(series) % 4 == 0
    series = np.split(series, 4, axis=0)  # [fold t 96]
    base = series[3]
    d1 = series[0] - base
    d2 = series[1] - base
    d3 = series[2] - base

    re = (d1 + d2 + d3) / 3  # [t 96]

    assert np.shape(re) == (512, 96)
    return re


def dct_1d(series):
    # [t d]
    series = torch.from_numpy(series).T.cuda()  # [d=96 t=512]
    sf = dct.dct(series)  # tensor [96 512]
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


def downsample(x, ratio):
    return x[::ratio, :]


def trial_average(x, axis=0):
    #  [1000, 127]
    ave = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return (x - ave) / std


if __name__ == "__main__":
    x = np.random.random([2, 3, 2])
    print(trial_average(x, 1))
