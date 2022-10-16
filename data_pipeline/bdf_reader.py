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
    def __init__(self, file_path='E:/Datasets/CVPR2021-02785/data/imagenet40-1000-1-00.bdf', method='stim'):
        self.selection = []
        self.file_path = file_path

        if method == 'auto':
            self.method = self.read_auto
        elif method == 'stim':
            self.method = self.read_by_stim
        elif method == 'manual':
            self.method = self.read_by_manual
        self.set = self.method()

    def get_set(self):
        return self.set

    def get_item(self, file_path, sample_idx):
        if file_path == self.file_path:
            return self.set[sample_idx]
        else:
            self.file_path = file_path
            self.set = self.method()
            return self.set[sample_idx]

    def read_raw(self):
        raw = mne.io.read_raw_bdf(self.file_path, preload=True,
                                  exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'],
                                  stim_channel='Status')
        # raw = raw.filter(l_freq=49, h_freq=51, method='fir', fir_window='hamming')
        events = mne.find_events(raw, stim_channel='Status', initial_event=True, output='step')
        # new_bdf, new_events = raw.resample(sfreq=1024, events=events)  # down sampling to 1024Hz
        print(raw)
        print(raw.info)
        # print(raw.ch_names)
        # raw.plot_psd(fmax=20)
        # raw.plot(duration=5, c_channels=96)
        return raw, events

    def read_auto(self):
        raw, events = self.read_raw()
        event_dict = {'stim': 65281, 'end': 0}
        print(events)

        # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'])
        # fig.subplots_adjust(right=0.7)
        # reject_criteria = dict()

        # mne.viz.plot_raw(raw=raw, events=events)
        # plt.show()

        epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True).drop_channels('Status')
        epochs.equalize_event_counts(['stim'])
        stim_epochs = epochs['stim']
        # stim_epochs.plot_image(picks='A12')

        # for s in stim_epochs:
        #     print(np.shape(s))

        del raw, epochs, events
        return stim_epochs.get_data()

    def read_by_stim(self):
        raw, events = self.read_raw()
        start_time = []
        for event in events:
            if event[1] == 0 and event[2] == 0:
                pass  # 0.2s
            if event[1] == 130816 and event[2] == 131069:
                pass  # 10s
            if event[1] == 65280 and event[2] == 65281:
                start_time.append(event[0])  # 3s and the last sample is contact with 10s blocking
            if event[1] == 65280 and event[2] == 0:
                pass  # end 11s
        if not len(start_time) == 400:
            raise Exception('len(start_time) != 400')

        set = []
        for i in range(len(start_time) - 1):
            start = start_time[i]
            # each sample lasting 2s, the 0.5s data from starting are selected in citation, 0.5*4096=2048
            end = start + 2048

            picks = mne.pick_types(raw.info, eeg=True, stim=False)
            data, times = raw[picks, start:end]
            # data = data[:, 0:8191:20]  # down sampling
            # data = block_reduce(data, block_size=(1, 4), func=np.mean, cval=np.mean(data))
            set.append(data.T)  # [time, channels]
            # EEG_times.append(times[0])
        return set

    def read_by_manual(self):
        """
        # divide bdf into 400 samples according to the describing in paper:
        Each run started with 10 s of blanking, followed by 400 stimulus presentations, each lasting 2 s,
        with 1 s of blanking between adjacent stimulus presentations, followed by 10 s of blanking at the end
        of the run
        """
        raw, events = self.read_raw()
        picks = mne.pick_types(raw.info, eeg=True, stim=False)
        set = []
        for i in range(400):
            start = 3.0 * i
            end = start + 2.0
            t_idx = raw.time_as_index([10. + start, 10. + end])
            data, times = raw[picks, t_idx[0]:t_idx[1]]
            # EEGs[times[0]] = data.T
            set.append(data.T)
            # EEG_times.append(times[0])
        return set


if __name__ == '__main__':
    b = BDFReader(method='stim')
    sample = b.get_set()[0]
    signal = sample[:, 5]

    from data_pipeline.pre_processing.time_frequency import signal2spectrum, signal2spectrum_pywt_cwt
    # signal2spectrum(signal)
    signal2spectrum_pywt_cwt(signal=signal)

