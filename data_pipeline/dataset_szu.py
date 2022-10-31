# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/26 19:07
 @name: 
 @desc:
"""
import torch
import pickle
from utils.my_tools import file_scanf
from torch.utils.data.dataloader import default_collate
from data_pipeline.pre_processing.difference import downsample


def collate_(batch):  # [b, 2], [x, y]
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)


class SZUDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.filepaths = file_scanf(path, endswith='.pkl')

    def __len__(self):  # called by torch.utils.data.DataLoader
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        with open(filepath, 'rb') as f:
            x = pickle.load(f)       # SZU: [t=2000, channels=127], Purdue: [512, 96]
            y = int(pickle.load(f))

            # x = downsample(x, ratio=4)  # SZU, [500, 127]
            # x = x[:1000, :]               # SZU, [1000, 127]
            # y = y-1                  # Ziyan He created EEG form

            assert 0 <= y <= 39
        return torch.tensor(x, dtype=torch.float).unsqueeze(0), torch.tensor(y, dtype=torch.long)
