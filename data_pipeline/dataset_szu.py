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
from pre_process.difference import jiang_delta_ave
import numpy as np


def collate_(batch):  # [b, 2], [x, y]
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)


class SZUDataset(torch.utils.data.Dataset):
    def __init__(self, path, condition='run_', endswith='.pkl'):
        self.filepaths = file_scanf(path, contains=condition, endswith=endswith)

    def __len__(self):  # called by torch.utils.data.DataLoader
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        with open(filepath, 'rb') as f:
            x = pickle.load(f)       # SZU: [t=2000, channels=127], Purdue: [512, 96]
            y = int(pickle.load(f))

            assert 0 <= y <= 39
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, path_list):
        self.path_list = path_list

    def __len__(self):  # called by torch.utils.data.DataLoader
        return len(self.path_list)

    def __getitem__(self, idx):
        filepath = self.path_list[idx]
        with open(filepath, 'rb') as f:
            x = pickle.load(f)       # SZU: [t=2000, channels=127], Purdue: [512, 96]
            y = int(pickle.load(f))

            # jiang ave
            # x = jiang_delta_ave(x)  # [2048 96] -> [512 96]
            # x = jiang_four_ave(x, fold=4)  # [2048 96] -> [512 96]

            # 1D-DCT
            # x = dct_1d(x)  # same with x [512 96]
            # 2D-DCT
            # x = dct_2d(x)  # same with x
            # approximated dct
            # x = approximated_dct(x)  # 1/2 of x shape [4, 256 48]

            # stft SZU:[127, 40, 101], Purdue:[96, 33, 63]
            # x = np.array(x)  # [96, 33, 63]
            # x = np.transpose(x, (1, 0,  2))  # [33, 96, 63]

            # cwt # [127, 85, 2000]
            # x = x[:, :, :1000]  # [127, 85, 1000]
            # x = x[:, :, ::2]  # [127, 85, 500]

            # x = downsample(x, ratio=4)  # SZU, [500, 127]
            x = x[:512, :]               # [512 96]
            # x = difference(x, fold=4)     # SZU, [500, 127]
            # y = y-1                  # Ziyan He created EEG form

            x = np.expand_dims(x, axis=0)  # Purdue [512 96] -> [1 512 96] added channel for EEGNet
            # assert np.shape(x) == (63, 3, 32, 32)  # aep
            assert 0 <= y <= 39
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)
        # return torch.tensor(x, dtype=torch.float).permute(1, 2, 0).unsqueeze(0), torch.tensor(y, dtype=torch.long)
