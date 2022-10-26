# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/5/21 8:12 PM
@desc:
"""
import torch
from data_pipeline.mne_reader import BDFReader
from data_pipeline.label_reader import LabelReader
import glob
import platform
from torch.utils.data.dataloader import default_collate


def collate_(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)


class PurdueDataset(torch.utils.data.Dataset):
    def __init__(self, CVPR2021_02785_path, sample_rate=1024):
        self.sample_rate = sample_rate
        self.BDFs_path = CVPR2021_02785_path + '/data'
        self.labels_path = CVPR2021_02785_path + '/design'
        # self.image_path = CVPR2021_02785_path + '/stimuli'

        self.bdf_filenames = self.file_filter(self.BDFs_path, endswith='.bdf')
        # self.label_filenames = self.file_filter(self.labels_path, endswith='.txt')

        self.bdf_reader = BDFReader(file_path='../../Datasets/CVPR2021-02785/data/imagenet40-1000-1-00.bdf',
                                    resample=self.sample_rate)
        self.label_reader = LabelReader(file_path='../../Datasets/CVPR2021-02785/design/run-00.txt')

    def __len__(self):
        return len(self.bdf_filenames) * 400  # each .bdf file embody 400 samples.

    def __getitem__(self, idx):
        file_idx = int(idx / 400)
        sample_idx = idx % 400

        # if file_idx < 10:
        #     file_idx_name = '0'+str(file_idx)  # 00 01 02 ... 09
        # else:
        #     file_idx_name = file_idx  # 10 11 12 ... 99

        # print(file_idx, sample_idx)
        bdf_path = self.bdf_filenames[file_idx]
        # label_path = self.label_filenames[file_idx]
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
