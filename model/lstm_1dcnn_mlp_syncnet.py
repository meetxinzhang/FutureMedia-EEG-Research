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


# def _init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, std=.002)
#         if isinstance(m, nn.Linear) and m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.constant_(m.bias, 0)
#         nn.init.constant_(m.weight, 1.0)
#     elif isinstance(m, nn.Conv1d):
#         nn.init.xavier_uniform_(m.weight)
#     elif isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform_(m.weight)
#     elif isinstance(m, nn.GRU):
#         nn.init.xavier_uniform_(m.weight)


class LSTM(nn.Module):
    def __init__(self, classes, input_size=96, depth=3):
        super().__init__()
        f = 12*input_size
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=f, kernel_size=15, stride=8, groups=input_size),
            nn.LeakyReLU(),
        )
        self.backbone = nn.GRU(input_size=f, hidden_size=f, num_layers=depth, bias=True,
                               batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=depth*f, out_features=f),
            nn.LeakyReLU(),
            nn.Linear(in_features=f, out_features=classes),
            nn.Softmax(dim=-1)

        )

    def forward(self, x):
        # [b t c]
        x = self.feature_extractor(x.transpose(1, 2))
        _, hn = self.backbone(x.transpose(1, 2))
        x = einops.rearrange(hn, 'l b f -> b (l f)')
        return self.classifier(x)


class SlidMLP(nn.Module):
    def __init__(self, in_features, classes, w=128, drop=0.2):
        super().__init__()
        self.w = w
        self.backbone = nn.Sequential(
            nn.Linear(in_features=int(in_features*w), out_features=int(in_features*w)),
            nn.LeakyReLU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=int(in_features*w), out_features=int(in_features*w//4)),
            nn.LeakyReLU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=int(in_features*w//4), out_features=int(in_features*4)),
            nn.ReLU(),
            nn.Linear(in_features=int(in_features*4), out_features=in_features),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=672, out_features=in_features*4),
            nn.ReLU(),
            nn.Linear(in_features=in_features*4, out_features=classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # [b t c]
        x = x.unfold(dimension=1, size=self.w, step=self.w//2)  # [b l w c]
        x = einops.rearrange(x, 'b t w c -> b t (w c)')
        x = self.backbone(x)  # [b t i]
        x = einops.rearrange(x, 'b t i -> b (t i)')
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
            torch.nn.Linear(2048, classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # [b t c] -> [b c t]
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.view(-1, 2048)
        x = self.classifier(x)
        return x


class SyncNet(nn.Module):
    def __init__(self, in_channels, num_layers_in_fc_layers=1024):
        super(SyncNet, self).__init__()

        # self.__nFeatures__ = 24
        # self.__nChs__ = 32
        # self.__midChs__ = 32

        self.cnn_aud = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 64, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fc_aud = nn.Sequential(
            nn.Linear(145152, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # [bs c 24 m]  [b c=96 f=30 t=1024]
        mid = self.cnn_aud(x)  # N x ch x 24 x M
        mid = mid.view((mid.size()[0], -1))  # N x (ch x 24)
        out = self.fc_aud(mid)

        return out
