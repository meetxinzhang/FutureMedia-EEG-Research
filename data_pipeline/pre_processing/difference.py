# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/27 11:41
 @name: 
 @desc: frequencies difference
"""
import numpy as np


def difference(series, fold):
    # time-series [t, d]
    length = int(len(series) // fold)
    base = series[0:length - 1, :]
    a2 = series[2*length:(length * 3) - 1, :]
    assert np.shape(base)[0] == np.shape(a2)[0]
    delta = a2 - base
    del series, base, a2
    return delta


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
