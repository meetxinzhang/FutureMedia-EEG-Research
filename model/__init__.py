# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/4/21 2:34 PM
@desc:
"""
import mne

bdf = mne.io.read_raw_bdf('/media/xin/Raid0/dataset/CVPR2021-02785/data/imagenet40-1000-1-00.bdf', preload=True)

print(bdf.info, '\n')
print('channels: ', len(bdf.ch_names), bdf.ch_names, '\n')
print('times: ', bdf.n_times, bdf.times, '\n')

events = mne.find_events(bdf, initial_event=True)
events[:, 2] &= (2**16 - 1)

print(events)

"""
402 events found
Event IDs: [ 65281 131069 131071]
[[      0       0   65535]
 [    977  130816   65533]
 [  43737   65280   65281]
 ...
 [4949078   65280   65281]
 [4961435   65280   65281]
 [4973791   65280   65281]]
 
 see https://mne.tools/dev/generated/mne.find_events.html#mne.find_events for details
"""


# df = bdf.to_data_frame(index='time', time_format=None)
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
