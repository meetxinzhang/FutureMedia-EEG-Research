# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/4/21 3:35 PM
@desc:
"""
import pandas as pd
from mne.io import read_raw_bdf
import mne
import numpy as np

# with open('/media/xin/Raid0/dataset/CVPR2021-02785/data/imagenet40-1000-1-00.bdf', 'r', encoding='') as f:
#     print(f.readline())

bdf = read_raw_bdf('/media/xin/Raid0/dataset/CVPR2021-02785/data/imagenet40-1000-1-00.bdf')
print(bdf.info, '\n')
print('channels: ', len(bdf.ch_names), bdf.ch_names, '\n')
print('times: ', bdf.n_times, bdf.times, '\n')
# print(np.shape(bdf[0][0]), '\n')
# print(np.shape(bdf[0][1]), '\n')

df = bdf.to_data_frame(index='time', time_format='ms')

# Status = df['Status']
# stimulus = []
# i = 0
# for index, row in Status.iteritems():
#     if row not in stimulus:
#         stimulus.append(i)
#         i = 0
#     else:
#         i = i + 1

# print(len(stimulus), stimulus)
with pd.option_context('display.max_rows', 500, 'display.max_columns', 10):
    print(df[0:20]['EXG8'])

# for data, times in bdf:
#     print(np.shape(data), data)
#     print(np.shape(times), times)
