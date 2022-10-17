
# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/10 16:37
 @desc:
"""
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import pywt

# return a dict
mat = scipy.io.loadmat('D:/Datasets/ShenzhenFace-EEG_Dataset/Session5_Experiment_Split_Data/ChenMinghui'
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

# print(mat['Channels'].shape, '\n', mat['Channels'], '\n')


print(np.shape(mat['AF3']))

# continues wavelet transform
# t = np.linspace(0, 500, 500, endpoint=False)
# widths = np.arange(12, 28)

# cwtmatr = scipy.signal.cwt(data=np.squeeze(mat['AF3']), wavelet=scipy.signal.ricker, widths=widths)
# plt.imshow(cwtmatr, extent=[0, 500, 12, 28], cmap='PRGn', aspect='auto',
#            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
# plt.show()

wavename = 'db5'
cA, cD = pywt.dwt(np.squeeze(mat['AF3']), wavename)
print('aaaaaaaaaaaaa', np.shape(cA))
ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component
x = range(len(np.squeeze(mat['AF3'])))
plt.figure(figsize=(12, 9))
plt.subplot(311)
plt.plot(x, np.squeeze(mat['AF3']))
plt.title('original signal')
plt.subplot(312)
plt.plot(x, ya)
plt.title('approximated component')
plt.subplot(313)
plt.plot(x, yd)
plt.title('detailed component')
plt.tight_layout()
plt.show()
