# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/27 11:41
 @name: 
 @desc: frequencies difference
"""
import numpy as np


def df(series, length):
    # time-series [t, d]
    base = series[0:length-1, :]
    a1 = series[length:(length*2)-1, :]
    assert np.shape(base)[0] == np.shape(a1)[0]
    delta = a1-base
    del series, base, a1
    return delta
