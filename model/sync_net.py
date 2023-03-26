# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/26 13:42
 @desc:
"""

import torch
import torch.nn as nn


class SyncNet(nn.Module):
    def __init__(self, num_layers_in_fc_layers=1024):
        super(SyncNet, self).__init__()

        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32

        self.cnn_aud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),

            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc_aud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

    def forward(self, x):
        # [bs c 24 m]
        mid = self.cnn_aud(x)  # N x ch x 24 x M
        mid = mid.view((mid.size()[0], -1))  # N x (ch x 24)
        out = self.fc_aud(mid)

        return out
