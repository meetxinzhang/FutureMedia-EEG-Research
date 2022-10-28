# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/5/21 8:12 PM
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


class PurdueDataset(torch.utils.data.Dataset):
    def __init__(self, CVPR2021_02785_path):
        self.BDFs_path = CVPR2021_02785_path + '/data'
        self.labels_path = CVPR2021_02785_path + '/design'
        # self.image_path = CVPR2021_02785_path + '/stimuli'

        self.bdf_filenames = self.file_filter(self.BDFs_path, endswith='.bdf')
        # self.label_filenames = self.file_filter(self.labels_path, endswith='.txt')

        self.bdf_reader = MNEReader(resample=1024, length=512,
                                    exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'])
        self.label_reader = LabelReader()

    def __len__(self):
        return len(self.bdf_filenames) * 400  # each .bdf file embody 400 samples.

    def __getitem__(self, idx):
        file_idx = int(idx / 400)
        sample_idx = idx % 400

        bdf_path = self.bdf_filenames[file_idx]
        try:
            x = self.bdf_reader.get_item(bdf_path, sample_idx)  # [t=512, channels=96]
            number = bdf_path.split('-')[-1].split('.')[0]  # ../imagenet40-1000-1-02.bdf
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
