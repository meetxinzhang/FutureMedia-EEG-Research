
# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/10 16:37
 @desc:
"""
import numpy as np
import scipy.io

# return a dict
mat = scipy.io.loadmat('E:/Datasets/ShenzhenFace-EEG_Dataset/Session5_Experiment_Split_Data/ChenMinghui'
                       '/Aaron_Eckhart_0067.jpg.0001-0500.eeg.mat')

print(mat.keys(), '\n')
print(mat['__header__'], '\n')
# print(mat['__version__'], '\n')
# print(mat['__globals__'], '\n')
print('SampleRate: ', mat['SampleRate'])
print('SegmentCount: ', mat['SegmentCount'])
print('MarkerCount: ', mat['MarkerCount'], '\n')

print('Time: ', len(mat['t']))  # time

print('ChannelCount: ', mat['ChannelCount'], '\n')
assert len(mat.keys())-9 == 128

print('channel AF3: ', np.shape(mat['AF3']))
print('channel FFC2h: ', np.shape(mat['FFC2h']))

print('Channels: ', mat['Channels'].shape, '\n')

print(mat['Channels'].shape, '\n', mat['Channels'], '\n')


