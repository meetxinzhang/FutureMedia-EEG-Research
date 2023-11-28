# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/4/21 3:35 PM
@desc:
"""
import mne
import numpy as np
from utils.my_tools import ExceptionPassing
# from skimage.measure import block_reduce
mne.set_log_level(verbose='WARNING')


def get_electrode_pos(raw, montage='brainproducts-RNP-BA-128'):
    # montage = mne.channels.make_standard_montage(kind=kind, head_size='auto')
    # raw.set_montage('brainproducts-RNP-BA-128', match_alias=True, on_missing='warn')  # SZU
    raw.set_montage(montage, match_alias=True, on_missing='warn')  # PD
    montage = raw.get_montage()
    pos_map = montage.get_positions()['ch_pos']  # ordered dict
    pos = np.array(list(pos_map.values()))
    return pos


class MNEReader(object):
    def __init__(self, filetype='edf', method='stim', resample=None, length=512, exclude=(), stim_channel='auto',
                 montage=None):
        """
        @method: auto, stim, manual, default=stim
        @stim_channel: str. Default=auto, in default case, the stim_list is needed, and method=manual.
        """
        self.filetype = filetype
        self.file_path = None
        self.resample = resample
        self.length = length
        self.exclude = exclude
        self.stim_channel = stim_channel
        self.montage = montage
        if stim_channel == 'auto':
            assert method == 'manual'

        if method == 'auto':
            self.method = self.read_auto
        elif method == 'stim':
            self.method = self.read_by_stim
        elif method == 'manual':
            self.method = self.read_by_manual
        self.set = None
        self.pos = None

    def get_set(self, file_path, stim_list=None):
        self.file_path = file_path
        self.set = self.method(stim_list)
        return self.set

    def get_pos(self):
        assert self.set is not None
        return self.pos

    def get_item(self, file_path, sample_idx, stim_list=None):
        if self.file_path == file_path:
            return self.set[sample_idx]
        else:
            # print('un-hit', '\n', file_path, '\n', self.file_path)
            self.file_path = file_path
            self.set = self.method(stim_list)
            return self.set[sample_idx]

    def read_raw(self):
        if self.filetype == 'bdf':
            raw = mne.io.read_raw_bdf(self.file_path, preload=True, exclude=self.exclude,
                                      stim_channel=self.stim_channel)
        elif self.filetype == 'edf':
            raw = mne.io.read_raw_edf(self.file_path, preload=True, exclude=self.exclude,
                                      stim_channel=self.stim_channel)
        else:
            raise Exception('!!')

        # print(raw)
        # print(raw.info)
        # raw = raw.filter(l_freq=49, h_freq=51, method='fir', fir_window='hamming')
        if self.montage is not None:
            self.pos = get_electrode_pos(raw=raw, montage=self.montage)
        if self.stim_channel == 'auto':
            if self.resample is not None:
                raw = raw.resample(sfreq=self.resample)  # down sampling to 1024Hz
            return raw
        else:
            events = mne.find_events(raw, stim_channel=self.stim_channel, initial_event=True, output='step')
            if self.resample is not None:
                raw, events = raw.resample(sfreq=self.resample, events=events)  # down sampling to 1024Hz
            return raw, events
        # print(events)
        # print(raw.ch_names)
        # raw.plot_psd(fmax=20)
        # raw.plot(duration=5, c_channels=96)
        # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'])
        # fig.subplots_adjust(right=0.7)
        # reject_criteria = dict()
        # mne.viz.plot_raw(raw=raw, events=events)
        # plt.show()
        # stim_epochs.plot_image(picks='A12')

    def read_by_stim(self, *args):
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
        if len(start_time) != 400:
            raise ExceptionPassing('len(start_time) != 400')

        set = []
        for i in range(len(start_time)):
            start = start_time[i] - 512
            # each sample lasting 2s, the 0.5s data from starting are selected in citation, 0.5*sample ratio
            end = start + self.length
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
            data, _ = raw[picks, start:end]
            set.append(data.T)  # [time, channels]
        del raw, events
        return set, start_time  # [b, t, c]

    def read_by_manual(self, stim_list):
        """divide bdf into 400 samples according to the describing in paper:
        Each run started with 10 s of blanking, followed by 400 stimulus presentations, each lasting 2 s,
        with 1 s of blanking between adjacent stimulus presentations, followed by 10 s of blanking at the end
        of the run
        """
        raw = self.read_raw()
        picks = mne.pick_types(raw.info, eeg=True, stim=False)
        set = []
        for i in stim_list:
            end = i + self.length
            # if i in time-unit then: idx = raw.time_as_index([i, end])
            data, times = raw[picks, i:end]
            set.append(data.T)
        # for i in range(400):
        #     start = 3.0 * i
        #     end = start + 2.0
        #     t_idx = raw.time_as_index([10. + start, 10. + end])
        #     data, times = raw[picks, t_idx[0]:t_idx[1]]
        #     set.append(data.T)
        return set

    def read_auto(self, *args):
        raw, events = self.read_raw()
        event_dict = {'stim': 65281, 'end': 0}
        epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True).drop_channels('Status')
        epochs.equalize_event_counts(['stim'])
        stim_epochs = epochs['stim']
        del raw, epochs, events
        return stim_epochs.get_data().transpose(0, 2, 1)  # [b, c, t]


if __name__ == '__main__':
    # b = MNEReader(method='stim', resample=1024)
    # sample = b.get_set()[0]  # [b=1, c=96, t=2048]
    from serialize_szu import ziyan_read
    edf_reader = MNEReader(filetype='edf', method='manual', length=1000, montage='brainproducts-RNP-BA-128')
    stim, y = ziyan_read('label_file')  # [frame_point], [class]
    x = edf_reader.get_set(file_path=label_file.replace('.Markers', '.edf'), stim_list=stim)


# def edf_reader(path='E:/Datasets/set1_2022.10.13/test1016_hzy_one_set-edf.edf'):
#     raw = mne.io.read_raw_edf(path, preload=True)
#     print(raw)
#     print(raw.info)
#     # raw = raw.filter(l_freq=49, h_freq=51, method='fir', fir_window='hamming')
#     # events = mne.find_events(raw, stim_channel='Status', initial_event=True, output='step')
#     # if self.resample is not None:
#     #     raw, events = raw.resample(sfreq=self.resample, events=events)  # down sampling to 1024Hz
#     # print(np.shape(events))
#     # print(events)
#     print(raw.ch_names)
#     # raw.plot_psd(fmax=20)
#     # raw.plot(duration=5, c_channels=96)
#     pass
#
#
# if __name__ == '__main__':
#     pass
#     # mkrs = mne.io.kit.read_mrk('E:/Datasets/set1_2022.10.13/test1016_hzy_one_set_Raw Data Inspection.txt')
#     # raw = mne.io.read_raw_bdf('E:/Datasets/CVPR2021-02785/data/imagenet40-1000-1-00.bdf', preload=True,
#     #                           exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'],
#     #                           stim_channel='Status')
#     # print(raw)
#     # print(raw.info)
