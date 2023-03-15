# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/6/21 6:56 PM
@desc:

    This file is a testing place, and can be deleted.
"""
# import torch
#
# phi = torch.rand(2, 2, 3)
# label = [0]
# output = phi * 1.0
# print(output)
# batch_size = len(output)
# output[range(batch_size), label] = phi[range(batch_size), label]
#
# print(output)


#
# """
# import mne
# import numpy as np
# import matplotlib.pyplot as plt
#
# bdf = mne.io.read_raw_bdf('G:/Datasets/CVPR2021-02785/data/imagenet40-1000-1-09.bdf',
#                           preload=True, infer_types=True,
#                           exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'])
# bdf = mne.io.read_raw_edf('G:/Datasets/SZFace2/EEG/run_1_test_hzy.edf',
#                           preload=True, infer_types=True)
#
# # # dbf = bdf.filter(l_freq=49, h_freq=51, method='fir', fir_window='hamming')
# #
# # print(bdf.info, '\n')
# print('channels: ', len(bdf.ch_names), '\n', bdf.ch_names, '\n')
# print('times: ', bdf.n_times, bdf.times, '\n')
#
# events1 = mne.find_events(bdf, stim_channel='Status', initial_event=True, output='step')
# # events2 = mne.find_events(bdf, stim_channel='Status', initial_event=True, output='onset')
# # events3 = mne.find_events(bdf, stim_channel='Status', initial_event=True, output='offset')
# # events[:, 2] &= (2**16 - 1)
#
# new_bdf, new_events = bdf.resample(sfreq=1024, events=events1)
#
# # np.set_printoptions(threshold=np.inf)
# print(np.shape(new_events), '\n', new_events)
# # print(np.shape(events2), '\n', events2)
# # print(np.shape(events3), '\n', events3)
# #
# # start_time = []
# # for event in events1:
# #     if event[1] == 65280 and event[2] == 65281:
# #         start_time.append(event[0])
# #
# # print('\nlength: ', len(start_time))
#
# # bdf.pick_types(eeg=False, stim=True).plot()
# """
# 403 events found
# Event IDs: [     0  65281 130816 131069]
# (403, 3)
#  [[      0       0  130816]
#  [   1919  130816  131069]
#  [  43586   65280   65281]
#  ...
#  [4961294   65280   65281]
#  [4973650   65280   65281]
#  [5021696   65280       0]]
#
# length: 400
#
#  see https://mne.tools/dev/generated/mne.find_events.html#mne.find_events for more details
# """
# picks = mne.pick_types(bdf.info, eeg=True, stim=False,
#                        exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'])
# start_time = []
# for event in new_events:
#     if event[1] == 65280 and event[2] == 65281:
#         start_time.append(event[0])
#     if event[1] == 65280 and event[2] == 0:
#         start_time.append(event[0])  # the last sample is contact with 10s blocking
#
# print('\nlength: ', start_time)
#
# EEG_datas = []
# EEG_times = []
# for i in range(len(start_time)-1):
#     start = start_time[i]
#     # if i == 398:
#     end = start + 8192
#     # else:
#     #     end = start_time[i + 1]
#     print(start, end)
#
#     data, times = new_bdf[picks, start:end]
#     EEG_datas.append(data.T)
#     EEG_times.append(times[0])
#     # print(times)
#
# print(len(EEG_times))
#
# # df = bdf.to_data_frame(index='time', time_format='ms')
# # print(df)
#
# # Status = df['Status']
# # stimulus = []
# # i = 0
# # for index, row in Status.iteritems():
# #     if row not in stimulus:
# #         stimulus.append(i)
# #         i = 0
# #     else:
# #         i = i + 1
# # print(len(stimulus), stimulus)
#
# # with pd.option_context('display.max_rows', 500, 'display.max_columns', 10):
# #     print(df.iloc[0:20, 1:96])

# mne.channels.read_montage('GSN-HydroCel-128')
# mne.viz.plot_sensors
# kind = mne.channels.get_builtin_montages()
# print(kind)
# montage = mne.channels.read_custom_montage(fname='D:/GitHub/OptimusPrime/data_pipeline/other/biosemi96.sfp',
#                                            head_size=0.095,
#                                            coord_frame='head')
# montage = mne.channels.make_standard_montage(kind='biosemi128', head_size='auto')
# print(montage.get_positions())

# fig = montage.plot()
# fig.savefig('pd.jpg')



# map = {
#     'EXG1': 'eeg',
#     'EXG2': 'eeg',
#     'EXG3': 'eeg',
#     'EXG4': 'eeg',
#     'EXG5': 'eeg',
#     'EXG6': 'eeg',
#     'EXG7': 'eeg',
#     'EXG8': 'eeg'
# }
# bdf.set_channel_types(mapping=map)

# bdf.set_montage(montage=montage, match_alias=True, on_missing='warn')
# m = bdf.get_montage()
# # fig = m.plot()
# # fig.savefig('pd.jpg')
#
# orderedDict = m.get_positions()['ch_pos']
# # print(orderedDict)
#
# print(list(orderedDict.keys()))
# print(np.array(list(orderedDict.values())))
# # raw.set_montage(mon) ## !!
# ### look at channels
# mne.viz.plot_sensors(raw.info, show_names=True)
# raw.plot_sensors(show_names=True)
# raw.plot_sensors('3d')

# import pywt
# import PIL.Image as Image
# import matplotlib.pyplot as plt
#
#
# def signal2spectrum_pywt_cwt(signal, totalscal=20, wavelet='cmor4.0-20.0'):
#     """
#     scale = 4 是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
#     """
#     fc = pywt.central_frequency(wavelet)  # 计算小波函数的中心频率
#     cparam = 2 * fc * totalscal  # 常数c
#     # 可以用 *f = scale2frequency(wavelet, scale)/sampling_period 来确定物理频率大小。f的单位是赫兹，采样周期的单位为秒。
#     scales = cparam / np.arange(totalscal+1, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
#     cwtmatr, freqs = pywt.cwt(data=signal, scales=scales, wavelet=wavelet, sampling_period=1/1000)
#     return cwtmatr, freqs
#
#
# import pickle
# import numpy as np
# print(pywt.wavelist(family=None, kind='continuous'))
# # filepath = 'G:/Datasets/SZFace2/EEG/pkl_ave/run_1_test_hzy_1_3501_4.pkl'
# filepath = 'G:/Datasets/SZFace2/EEG/pkl_ave/run_1_test_hzy_258_771501_4.pkl'
# t = np.arange(0, 2, 1.0/1000)
# with open(filepath, 'rb') as f:
#     x = pickle.load(f)  # SZU: [t=2000, channels=127], Purdue: [512, 96]
#     y = int(pickle.load(f))
#
#     cwtmatr, freqs = signal2spectrum_pywt_cwt(x[:, 0])
#     print(np.shape(cwtmatr))
#     print(np.shape(freqs))
#     print(np.real(cwtmatr))
#
#     # image = Image.fromarray(np.uint8(cwtmatr))
#     # plt.imshow(image)
#     # plt.imshow(cwtmatr, extent=[1, 2000, 1, 256], cmap='PRGn', aspect='auto',
#     #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
#     plt.contourf(t, freqs, abs(np.real(cwtmatr)))
#     plt.show()
