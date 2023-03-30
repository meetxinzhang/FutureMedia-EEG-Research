# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/30 11:04
 @desc:
"""
import einops
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, classes, input_size=96, depth=3):
        super().__init__()
        self.backbone = nn.GRU(input_size=input_size, hidden_size=input_size, num_layers=depth, bias=True,
                               batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(in_features=input_size, out_features=classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


class MLP2layers(nn.Module):
    def __init__(self, in_features, hidden_size, classes):
        super().__init__()

        self.channel_fc = nn.Linear(in_features=in_features, out_features=hidden_size)  # [b t c->h]
        self.time_fc = nn.Linear(in_features=hidden_size * 128, out_features=512)

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.classifier = nn.Linear(in_features=512, out_features=classes)

    def forward(self, x):
        x = einops.rearrange(x, 'b c t -> b t c')
        x = self.channel_fc(x)  # [b t h]

        x = x.unfold(dimension=1, size=128, step=64)  # [b t w h]
        x = einops.rearrange(x, 'b t w h -> b t (w h)')
        x = self.time_fc(x)  # [b t 512]

        x = self.pool(torch.transpose(x, 1, 2)).squeeze(-1)

        return self.classifier(x)


class Bottleneck(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, down_sample=False):
        super().__init__()
        self.stride = 1
        if down_sample:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, hid_channels, 1, self.stride),
            torch.nn.BatchNorm1d(hid_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hid_channels, hid_channels, 3, padding=1),
            torch.nn.BatchNorm1d(hid_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hid_channels, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
        )

        if in_channels != out_channels:
            self.res_layer = torch.nn.Conv1d(in_channels, out_channels, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class ResNet1D(torch.nn.Module):
    def __init__(self, in_channels=96, classes=40):
        super(ResNet1D, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 128, kernel_size=15, stride=2, padding=7),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),

            Bottleneck(128, 128, 256, False),
            Bottleneck(256, 128, 256, False),
            Bottleneck(256, 128, 256, False),
            #
            Bottleneck(256, 128, 512, True),
            Bottleneck(512, 128, 512, False),
            Bottleneck(512, 128, 512, False),
            Bottleneck(512, 128, 512, False),
            #
            Bottleneck(512, 256, 1024, True),
            Bottleneck(1024, 256, 1024, False),
            Bottleneck(1024, 256, 1024, False),
            Bottleneck(1024, 256, 1024, False),
            Bottleneck(1024, 256, 1024, False),
            Bottleneck(1024, 256, 1024, False),
            #
            Bottleneck(1024, 512, 2048, True),
            Bottleneck(2048, 512, 2048, False),
            Bottleneck(2048, 512, 2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2048, classes)
        )

    def forward(self, x):
        # [b t c] -> [b c t]
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.view(-1, 2048)
        x = self.classifier(x)
        return x


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
