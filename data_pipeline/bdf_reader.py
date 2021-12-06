# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/4/21 3:35 PM
@desc:
"""
import mne


class BDFReader(object):
    def __init__(self, file_path='/media/xin/Raid0/dataset/CVPR2021-02785/data/imagenet40-1000-1-00.bdf'):
        self.selection = []
        self.file_path = file_path
        self.EEG_datas, self.EEG_times = self.read_as_events()

    def read_as_paper(self):
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
        EEG_times = []
        for i in range(400):
            start = 3.0 * i
            end = start + 2.0  #
            t_idx = new_bdf.time_as_index([10. + start, 10. + end])
            data, times = new_bdf[picks, t_idx[0]:t_idx[1]]
            # EEGs[times[0]] = data.T
            EEG_datas.append(data.T)
            EEG_times.append(times[0])
        return EEG_datas, EEG_times

    def read_as_events(self):
        # see https://mne.tools/dev/generated/mne.find_events.html#mne.find_events for more details
        bdf = mne.io.read_raw_bdf(self.file_path, preload=True)
        # bdf = bdf.filter(l_freq=49, h_freq=51, method='fir', fir_window='hamming')
        events = mne.find_events(bdf, stim_channel='Status', initial_event=True, output='step')
        new_bdf, new_events = bdf.resample(sfreq=1024, events=events)  # down sampling to 1024Hz

        picks = mne.pick_types(new_bdf.info, eeg=True, stim=False,
                               exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'])

        start_time = []
        for event in new_events:
            if event[1] == 65280 and event[2] == 65281:
                start_time.append(event[0])
            if event[1] == 65280 and event[2] == 0:
                start_time.append(event[0])  # the last sample is contact with 10s blocking

        EEG_datas = []
        EEG_times = []
        for i in range(len(start_time) - 1):
            start = start_time[i]
            # if i == 398:
            # each sample lasting 2s, the 0.5s data of starting are selected in paper, 0.5*1000*1.024=512
            end = start + 512
            # else:
            #     end = start_time[i + 1]

            data, times = new_bdf[picks, start:end]
            EEG_datas.append(data.T)
            EEG_times.append(times[0])
        return EEG_datas, EEG_times

    def get_item_matrix(self, file_path, sample_idx):
        if file_path == self.file_path:
            return self.EEG_datas[sample_idx]
        else:
            self.file_path = file_path
            self.EEG_datas, self.EEG_times = self.read_as_events()
            return self.EEG_datas[sample_idx]


