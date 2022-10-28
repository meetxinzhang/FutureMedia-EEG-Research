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
from data_pipeline.labels_purdue import LabelReader
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
        self.edf_filenames = self.file_filter(self.path, endswith='.edf')
        self.edf_reader = MNEReader(method='manual', resample=None, length=500, exclude=None, stim_channel=None)

    def __len__(self):
        return len(self.edf_filenames) * 400  # each .bdf file embody 400 samples.

    def __getitem__(self, idx):
        file_idx = int(idx / 400)
        sample_idx = idx % 400

        bdf_path = self.edf_filenames[file_idx]
        try:
            x = self.edf_reader.get_item(bdf_path, sample_idx)  # [t=512, channels=96]
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
        _input_files = files[:int(len(files) * 1)]
        return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(endswith),
                           _input_files))
