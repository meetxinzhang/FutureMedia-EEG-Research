# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/8 9:47
 @desc:
"""

import mne
import numpy as np


def read_by_event(file_path='E:/Datasets/CVPR2021-02785/data/imagenet40-1000-1-00.bdf'):
    # Visualization of EEG
    import matplotlib.pyplot as plt
    raw = mne.io.read_raw_bdf(file_path, preload=True,
                              exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'],
                              stim_channel='Status')
    # print(raw)
    # print(raw.info)
    # print(raw.ch_names)
    # raw.plot_psd(fmax=20)
    # raw.plot(duration=5, c_channels=96)

    events = mne.find_events(raw, stim_channel='Status', initial_event=True, output='step')
    event_dict = {'stim': 65281, 'block_end': 0}
    # print(events[:, -1])
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
    # picks_channels = mne.pick_types(raw.info, eeg=True, stim=False)
    # samples = stim_epochs.get_data()

    freqs = np.arange(12, 29, 1)  # 频段选取12-28Hz, Beta
    print(freqs)
    n_cycles = 5
    epochs_tfr = mne.time_frequency.tfr_morlet(stim_epochs, freqs=freqs, n_cycles=n_cycles,
                                               average=False, output='complex', return_itc=False)

    del raw, epochs, events
    return stim_epochs


class MNEReader(object):
    def __init__(self, file_path='E:/Datasets/CVPR2021-02785/data/imagenet40-1000-1-00.bdf'):
        self.file_path = file_path
        self.samples = read_by_event(self.file_path)

    def get_item_matrix(self, file_path, idx_item):
        if file_path == self.file_path:
            return self.samples[idx_item]
        else:
            self.file_path = file_path
            self.samples = read_by_event(self.file_path)
            return self.samples[idx_item]
        pass


if __name__ == '__main__':
    file_path = 'E:/Datasets/CVPR2021-02785/data/imagenet40-1000-1-00.bdf'
    epochs = read_by_event(file_path)

    # sample_A1 = mats[0]
    # print('5', np.shape(sample_A1))

    # wave
    # import pywt
    # import matplotlib.pyplot as plt
    # from PIL import Image
    #
    # [cwtmatr, freqs] = pywt.cwt(data=sample_A1, scales=np.arange(1, 97), wavelet='cgau8')
    #
    # image = Image.fromarray(np.uint8(cwtmatr))
    # plt.imshow(image)
    # plt.show()










