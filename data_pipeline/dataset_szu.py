# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/26 19:07
 @name: 
 @desc:
"""
import einops
import torch
import pickle
from utils.my_tools import file_scanf
from torch.utils.data.dataloader import default_collate
from pre_process.difference import *
import numpy as np
import os


def collate_(batch):  # [b, 2], [x, y]
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, path_list):
        self.path_list = path_list
        self.wrap = Wrapping(n=3)

    def __len__(self):  # called by torch.utils.data.DataLoader
        return len(self.path_list)

    def __getitem__(self, idx):
        filepath = self.path_list[idx]
        if os.path.getsize(filepath) <= 0:
            print('EOFError: Ran out of input')
            print(filepath)
            return
        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t c]  aep: [2048, 20, 20]
            y = int(pickle.load(f))
            # y = y - 1  # Ziyan He created EEG form

            # x = wrapping(x)
            x = trial_average(x, axis=0)
            x = x[:512, :]
            # x = trial_average(x, axis=0)

            # x = trial_average(x, axis=0)

            x = np.expand_dims(x, axis=0)  # 1 512 96
            x = einops.rearrange(x, 'f t c ->f c t')

            # x = x[:512, :]
            # x = dct_1d_numpy(x)  # same with x [512 96]
            # x = np.expand_dims(x, axis=0)  # 1 512 96
            # x = einops.rearrange(x, 'f t c ->f c t')

            # x = x[:, :, :248]  # 127 40 250
            # x = np.expand_dims(x, axis=0)  # [t c] -> [1, t, c]  # eegnet
            # x = einops.rearrange(x, 'c f t -> f c t')  # eegnet
            # x = einops.rearrange(x, 't c -> c t')  # resnet1d

            # eeg_conv_tsfm: b c w h t
            # x = x[:512, :, :]
            # x = x[::4, :, :]
            # x = four_ave(x, fold=4)  # [1024 96] -> [512 96]
            # x = np.expand_dims(x, axis=0)  # [t, w, h] -> [1, t, w, h]
            # x = einops.rearrange(x, 'c t w h -> c w h t')

            # x = delta_1(x)
            # x = x[::4, :, :]

            assert 0 <= y <= 39
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)


class AdaptedListDataset(torch.utils.data.Dataset):
    def __init__(self, path_list, exp, model):
        self.path_list = path_list
        self.exp = exp
        self.model = model

    def __len__(self):  # called by torch.utils.data.DataLoader
        return len(self.path_list)

    def __getitem__(self, idx):
        filepath = self.path_list[idx]
        if os.path.getsize(filepath) <= 0:
            print('EOFError: Ran out of input')
            print(filepath)

            return
        with open(filepath, 'rb') as f:
            x = pickle.load(f)       # 512 96
            y = int(pickle.load(f))

            x = trial_average(x, axis=0)

            # exps = ['nm', 'dct1d', 'dct2d', 'adc`1`   as  1
            # t', 'ave', 't_dff', 'dff_1', 'dff_b']
            if self.exp == 'nm':
                x = x[:512, :]
            elif self.exp == 'dct1d':
                x = x[:512, :]
                x = dct_1d_numpy(x)  # same with x [512 96]
            elif self.exp == 'dct2d':
                x = x[:512, :]
                x = dct2d(x)
            elif self.exp == 'adct':
                x = x[:512, :]
                x = approximated_dct(x)
            elif self.exp == 'ave':
                x = four_ave(x, fold=4)  # [1024 96] -> [512 96]
            elif self.exp == 't_dff':
                x = x[:514, :]
                x = frame_delta(x)
                x = x[:512, :]
            elif self.exp == 'dff_1':
                x = delta_1(x)
            elif self.exp == 'dff_b':
                x = delta_b(x)

            # models = ['cnn1d', 'cnn2d', 'resnet2d', 'lstm', 'mlp', 'resnet1d', 'eegchannelnet']
            if self.model =='cnn2d':
                x = np.expand_dims(x, axis=0)
                x = einops.rearrange(x, 'f t c -> f c t')
            if self.model == 'lstm':
                pass
                # t c
            if self.model =='mlp':
                pass
                # t c
            if self.model =='resnet1d':
                x = einops.rearrange(x, 't c -> c t')
            if self.model =='cnn1d':
                x = einops.rearrange(x, 't c -> c t')
            if self.model =='syncnet':
                x = einops.rearrange(x, 't c -> c t')
            if self.model == 'eegchannelnet':
                x = np.expand_dims(x, axis=0)
                x = einops.rearrange(x, 'f t c ->f c t')
            if self.model == 'eegnet':
                x = np.expand_dims(x, axis=0)  # 1 512 96
                x = einops.rearrange(x, 'f t c ->f c t')
            if self.model == 'resnet2d':
                x = np.expand_dims(x, axis=0)
                x = einops.rearrange(x, 'f t c ->f c t')
            if self.model == 'eegTsfm':
                x = np.expand_dims(x, axis=0)
                x = einops.rearrange(x, 'f t c ->f c t')

            # down sample
            # x = x[::2, :]  # [512, 96]
            # x = x[:512, :]  # [512 96]

            # x = delta_ave(x)  # [2048 96] -> [512 96]
            # x = four_ave(x, fold=2)  # [2048 96] -> [512 96]
            # x = time_delta_ave(x)  # [2048->512 96]

            # 1D-DCT
            # x = dct_1d(x)  # same with x [512 96]
            # 2D-DCT
            # x = block_img_dct(x)
            # approximated dct
            # x = approximated_dct(x)  # 1/2 of x shape [4, 256 48]

            # stft SZU:[127, 40, 101], Purdue:[96, 33, 63]
            # x = np.array(x)  # [96, 33, 63]
            # x = np.transpose(x, (1, 0,  2))  # [33, 96, 63]

            # cwt  [c=96 f=30 t=1024]
            # x = x[:, :, :512]

            # AEP
            # x = x[:512, :, :, :]
            # x = einops.rearrange(x, 't c w h -> c w h t')

            # CNN 2D
            # x = np.expand_dims(x, axis=0)  # added channel for EEGNet
            # x = einops.rearrange(x, 'f t c -> f c t')  # EEGChannelNet, EEGNet

            # x = difference(x, fold=4)     # SZU, [500, 127]
            # y = y-1  # Ziyan He created EEG form

            # x = np.expand_dims(x, axis=0)  # added channel for EEGNet
            # x = einops.rearrange(x, 'f t c -> f c t')  # EEGChannelNet, EEGNet  raw data
            # x = einops.rearrange(x, 'c f t -> f c t')  # EEGChannelNet, EEGNet
            assert 0 <= y <= 39
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)
        # return torch.tensor(x, dtype=torch.float).permute(1, 2, 0).unsqueeze(0), torch.tensor(y, dtype=torch.long)



class AdaptedListDataset3d(torch.utils.data.Dataset):
    def __init__(self, path_list, exp, model):
        self.path_list = path_list
        self.exp = exp
        self.model = model

    def __len__(self):  # called by torch.utils.data.DataLoader
        return len(self.path_list)

    def __getitem__(self, idx):
        filepath = self.path_list[idx]
        if os.path.getsize(filepath) <= 0:
            print('EOFError: Ran out of input')
            print(filepath)

            return
        with open(filepath, 'rb') as f:
            x = pickle.load(f)       # 512 96
            y = int(pickle.load(f))
            y = y - 1  # Ziyan He created EEG form
            # x = trial_average(x, axis=0)
            # print(x, 'qqq')  # 2048, 20, 20

            # exps = ['nm', 'dct1d', 'dct2d', 'adct', 'ave', 't_dff', 'dff_1', 'dff_b']
            if self.exp == 'nm':
                x = x[:500, :, :]
            elif self.exp == 'dct1d':
                x = x[:512, :, :]
                x = dct_1d_numpy(x, axis=0)  # same with x [512 96]
            elif self.exp == 'dct2d':
                x = x[:512, :, :]
                new_x = []
                for xi in x:
                    new_x.append(dct2d(xi, block=4))
                x = new_x
            elif self.exp == 'adct':
                x = x[:512, :, :]
                new_x = []
                for xi in x:
                    new_x.append(approximated_dct(xi))
                x = new_x
            elif self.exp == 'ave':
                x = four_ave(x, fold=4)  # [1024 96] -> [512 96]
            elif self.exp == 't_dff':
                x = x[:514, :, :]
                x = frame_delta_video(x)
                x = x[:512, :, :]
            elif self.exp == 'dff_1':
                x = delta_1(x)
            elif self.exp == 'dff_b':
                x = delta_b(x)

            x = np.array(x)
            x = x[::4, :, :]

            # models = ['VideoTsfm', 'ConvTsfm']
            if self.model =='VideoTsfm':
                x = np.expand_dims(x, axis=0)  # [t, w, h] -> [1, t, w, h]
                x = einops.rearrange(x, 'c t w h -> c t h w')
            if self.model == 'ConvTsfm':
                x = np.expand_dims(x, axis=0)  # [t, w, h] -> [1, t, w, h]
                x = einops.rearrange(x, 'c t w h -> c w h t')

            assert 0 <= y <= 39
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)
