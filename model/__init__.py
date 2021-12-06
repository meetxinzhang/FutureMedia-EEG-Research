# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/4/21 2:34 PM
@desc:
"""
import mne
import numpy as np

bdf = mne.io.read_raw_bdf('/media/xin/Raid0/dataset/CVPR2021-02785/data/imagenet40-1000-1-09.bdf', preload=True)
# dbf = bdf.filter(l_freq=49, h_freq=51, method='fir', fir_window='hamming')

print(bdf.info, '\n')
print('channels: ', len(bdf.ch_names), bdf.ch_names, '\n')
print('times: ', bdf.n_times, bdf.times, '\n')

events1 = mne.find_events(bdf, stim_channel='Status', initial_event=True, output='step')
# events2 = mne.find_events(bdf, stim_channel='Status', initial_event=True, output='onset')
# events3 = mne.find_events(bdf, stim_channel='Status', initial_event=True, output='offset')
# events[:, 2] &= (2**16 - 1)

new_bdf, new_events = bdf.resample(sfreq=1024, events=events1)

# np.set_printoptions(threshold=np.inf)
print(np.shape(new_events), '\n', new_events)
# print(np.shape(events2), '\n', events2)
# print(np.shape(events3), '\n', events3)
#
# start_time = []
# for event in events1:
#     if event[1] == 65280 and event[2] == 65281:
#         start_time.append(event[0])
#
# print('\nlength: ', len(start_time))

# bdf.pick_types(eeg=False, stim=True).plot()
"""
403 events found
Event IDs: [     0  65281 130816 131069]
(403, 3) 
 [[      0       0  130816]
 [   1919  130816  131069]
 [  43586   65280   65281]
 ...
 [4961294   65280   65281]
 [4973650   65280   65281]
 [5021696   65280       0]]
 
length: 400
 
 see https://mne.tools/dev/generated/mne.find_events.html#mne.find_events for more details
"""
picks = mne.pick_types(new_bdf.info, eeg=True, stim=False,
                       exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'])
start_time = []
for event in new_events:
    if event[1] == 65280 and event[2] == 65281:
        start_time.append(event[0])
    if event[1] == 65280 and event[2] == 0:
        start_time.append(event[0])  # the last sample is contact with 10s blocking

print('\nlength: ', start_time)

EEG_datas = []
EEG_times = []
for i in range(len(start_time)-1):
    start = start_time[i]
    # if i == 398:
    end = start + 8192
    # else:
    #     end = start_time[i + 1]
    print(start, end)

    data, times = new_bdf[picks, start:end]
    EEG_datas.append(data.T)
    EEG_times.append(times[0])
    # print(times)

# print(EEG_times)

# df = bdf.to_data_frame(index='time', time_format='ms')
# print(df)

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

# with pd.option_context('display.max_rows', 500, 'display.max_columns', 10):
#     print(df.iloc[0:20, 1:96])
