# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/26 19:07
 @name: 
 @desc:
"""
import mne


def edf_reader(path='E:/Datasets/set1_2022.10.13/test1016_hzy_one_set-edf.edf'):
    raw = mne.io.read_raw_edf(path, preload=True)
    print(raw)
    print(raw.info)
    # raw = raw.filter(l_freq=49, h_freq=51, method='fir', fir_window='hamming')
    # events = mne.find_events(raw, stim_channel='Status', initial_event=True, output='step')
    # if self.resample is not None:
    #     raw, events = raw.resample(sfreq=self.resample, events=events)  # down sampling to 1024Hz
    # print(np.shape(events))
    # print(events)
    print(raw.ch_names)
    # raw.plot_psd(fmax=20)
    # raw.plot(duration=5, c_channels=96)
    pass


if __name__ == '__main__':
    pass
    # mkrs = mne.io.kit.read_mrk('E:/Datasets/set1_2022.10.13/test1016_hzy_one_set_Raw Data Inspection.txt')
    # raw = mne.io.read_raw_bdf('E:/Datasets/CVPR2021-02785/data/imagenet40-1000-1-00.bdf', preload=True,
    #                           exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'],
    #                           stim_channel='Status')
    # print(raw)
    # print(raw.info)
