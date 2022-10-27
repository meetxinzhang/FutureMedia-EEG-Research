# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/26 19:07
 @name: 
 @desc:
"""
import torch
from data_pipeline.mne_reader import MNEReader
from data_pipeline.label_reader import LabelReader
import glob
import platform
from torch.utils.data.dataloader import default_collate


def collate_(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)


class SZUDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

        self.bdf_filenames = self.file_filter(self.path, endswith='.edf')
        # self.label_filenames = self.file_filter(self.labels_path, endswith='.txt')

        self.edf_reader = MNEReader(method='manual', resample=None, length=500, exclude=None, stim_channel=None)
        self.label_reader = LabelReader(file_path='../../Datasets/CVPR2021-02785/design/run-00.txt')

    def __len__(self):
        return len(self.bdf_filenames) * 400  # each .bdf file embody 400 samples.

    def __getitem__(self, idx):
        file_idx = int(idx / 400)
        sample_idx = idx % 400

        bdf_path = self.bdf_filenames[file_idx]
        try:
            x = self.bdf_reader.get_item(bdf_path, sample_idx)  # [t=512, channels=96]
            number = bdf_path.split('-')[-1].split('.')[0]
            label = self.label_reader.get_item_one_hot(self.labels_path+'/'+'run-'+number+'.txt', sample_idx)
        except Exception as e:
            print(e)
            print(bdf_path)
            return None, None

        return torch.tensor(x, dtype=torch.float).unsqueeze(0), torch.tensor(label, dtype=torch.long)

    def file_filter(self, path, endswith):
        files = glob.glob(path + '/*')
        if platform.system().lower() == 'windows':
            files = [f.replace('\\', '/') for f in files]
        disallowed_file_endings = (".gitignore", ".DS_Store")
        _input_files = files[:int(len(files) * self.sample_rate)]
        return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(endswith),
                           _input_files))







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
