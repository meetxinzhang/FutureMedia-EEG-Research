# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/8 9:47
 @desc:
"""

from data_pipeline.bdf_reader import BDFReader
import numpy as np
import mne


def get_sample():
    # Visualization of EEG
    raw = mne.io.read_raw_bdf('E:/Datasets/CVPR2021-02785/data/imagenet40-1000-1-85.bdf', preload=True)
    print(raw.info)
    print(raw.ch_names)
    # raw.plot_psd(fmax=20)
    # raw.plot(duration=5, n_channels=104)

    events = mne.find_events(raw, stim_channel='Status', initial_event=True, output='step')
    # print(events[:, -1])
    event_dict = {'start': 65281, 'block_end': 0}
    # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'])
    # fig.subplots_adjust(right=0.7)
    # reject_criteria = dict()
    epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True)
    # cond_we_care_about = ['start']
    # epochs.equalize_event_counts(cond_we_care_about)
    start_epochs = epochs['start']

    print(start_epochs)
    # start_epochs.plot_image(['B24', 'C17'])

    # frequencies = np.arange(7, 30, 3)
    # power = mne.time_frequency.tfr_morlet(start_epochs[0], n_cycles=2, return_itc=False, freqs=frequencies, decim=3)
    # power.plot()

    picks_channels = mne.pick_types(raw.info, eeg=True, stim=False,
                                    exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'])
    samples = start_epochs.get_data(picks_channels)
    print(np.shape(samples))  # [400, 96, 2868]
    del raw, epochs, events
    return samples[1]
