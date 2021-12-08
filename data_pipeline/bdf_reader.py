# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/4/21 3:35 PM
@desc:
"""
import mne
import numpy as np
from utils.exception_message import ExceptionPassing
from skimage.measure import block_reduce
mne.set_log_level(verbose='WARNING')


class BDFReader(object):
    def __init__(self, file_path='/media/xin/Raid0/dataset/CVPR2021-02785/data/imagenet40-1000-1-00.bdf'):
        self.selection = []
        self.file_path = file_path
        self.EEG_datas = self.read_by_events()

    def read_by_paper(self):
        """
        # divide bdf into 400 samples according to the describing in paper:
        Each run started with 10 s of blanking, followed by 400 stimulus presentations, each lasting 2 s,
        with 1 s of blanking between adjacent stimulus presentations, followed by 10 s of blanking at the end
        of the run
        """
        bdf = mne.io.read_raw_bdf(self.file_path, preload=True)
        new_bdf = bdf.resample(sfreq=1024)  # down sampling to 1024Hz
        picks = mne.pick_types(new_bdf.info, eeg=True, stim=False,
                               exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'])
        EEG_datas = []
        # EEG_times = []
        for i in range(400):
            start = 3.0 * i
            end = start + 2.0
            t_idx = new_bdf.time_as_index([10. + start, 10. + end])
            data, times = new_bdf[picks, t_idx[0]:t_idx[1]]
            # EEGs[times[0]] = data.T
            EEG_datas.append(data.T)
            # EEG_times.append(times[0])
        return EEG_datas

    def read_by_events(self):
        # see https://mne.tools/dev/generated/mne.find_events.html#mne.find_events for more details
        bdf = mne.io.read_raw_bdf(self.file_path, preload=True)
        # bdf = bdf.filter(l_freq=49, h_freq=51, method='fir', fir_window='hamming')
        events = mne.find_events(bdf, stim_channel='Status', initial_event=True, output='step')
        # new_bdf, new_events = bdf.resample(sfreq=1024, events=events)  # down sampling to 1024Hz

        picks = mne.pick_types(bdf.info, eeg=True, stim=False,
                               exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'])

        start_time = []
        for event in events:
            if event[1] == 65280 and event[2] == 65281:
                start_time.append(event[0])
            if event[1] == 65280 and event[2] == 0:
                start_time.append(event[0])  # the last sample is contact with 10s blocking
        if not len(start_time) == 401:
            raise Exception('len(start_time) != 400')

        EEG_datas = []
        for i in range(len(start_time) - 1):
            start = start_time[i]
            # each sample lasting 2s, the 0.5s data from starting are selected in citation, 0.5*1000*1.024=512
            end = start + 8192

            data, times = bdf[picks, start:end]
            # data = data[:, 0:8191:20]  # down sampling
            data = block_reduce(data, block_size=(1, 20), func=np.mean, cval=np.mean(data))
            EEG_datas.append(data.T)  # [time, channels]
            # EEG_times.append(times[0])
        return EEG_datas

    def get_item_matrix(self, file_path, sample_idx):
        if file_path == self.file_path:
            return self.EEG_datas[sample_idx]
        else:
            self.file_path = file_path
            self.EEG_datas = self.read_by_events()
            return self.EEG_datas[sample_idx]


