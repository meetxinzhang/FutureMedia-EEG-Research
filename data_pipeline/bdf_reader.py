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
        self.EEG_datas, self.EEG_times = self.read()

    def read(self):
        bdf = mne.io.read_raw_bdf(self.file_path, preload=True)

        picks = mne.pick_types(bdf.info, eeg=True, stim=False,
                               exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'])
        EEG_datas = []
        EEG_times = []
        for i in range(400):
            start = 3.0 * i
            end = start + 2.0
            t_idx = bdf.time_as_index([10. + start, 10. + end])
            data, times = bdf[picks, t_idx[0]:t_idx[1]]
            # EEGs[times[0]] = data.T
            EEG_datas.append(data.T)
            EEG_times.append(times[0])
        return EEG_datas, EEG_times

    def get_item_matrix(self, file_path, sample_idx):
        if file_path == self.file_path:
            return self.EEG_datas[sample_idx]
        else:
            self.file_path = file_path
            self.EEG_datas, self.EEG_times = self.read()
            return self.EEG_datas[sample_idx]


