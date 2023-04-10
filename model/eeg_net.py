# encoding: utf-8
"""
Forked from https://github.com/HANYIIK/EEGNet-pytorch/blob/master/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(m):
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        # nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class ComplexEEGNet(nn.Module):
    def __init__(self, classes_num, in_channels=30, electrodes=127, drop_out=0.25):
        super(ComplexEEGNet, self).__init__()

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((8, 7, 0, 0)),  # left, right, top, bottom of 2D img
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 3, kernel_size=(1, 16),
                      groups=in_channels, bias=False),  # output shape (63, C, T)
            nn.BatchNorm2d(in_channels * 3),  # output shape (63, C, T)
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 3)),  # (63, C, T/2)

            nn.ZeroPad2d((8, 7, 0, 0)),  # left, right, top, bottom of 2D img
            nn.Conv2d(in_channels=in_channels * 3, out_channels=in_channels * 9, kernel_size=(1, 16),  # filter size
                      groups=in_channels, bias=False),  # output shape (64, C, T)
            nn.BatchNorm2d(in_channels * 9),  # output shape (64, C, T)
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 3)),  # (63, C, T/2)
            nn.Dropout(drop_out)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 9, out_channels=32, kernel_size=(1, 1), bias=False),  # (16, 1, T//16)
            nn.BatchNorm2d(32),  # output shape (16, 1, T//16)
            nn.ELU(),

            nn.Conv2d(in_channels=32, out_channels=512, kernel_size=(electrodes, 1),  # filter size
                      groups=32, bias=False),  # output shape (128, 1, T//2)
            nn.BatchNorm2d(512),  # output shape (128, 1, T//2)
            nn.ELU(),

            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(1, 1), bias=False),  # output shape (16, 1, T//16)
            nn.BatchNorm2d(32),  # output shape (16, 1, T//16)
            nn.ELU(),
        )

        self.res_3 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 7), bias=False),
            nn.MaxPool2d((1, 4)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 15), bias=False),  # output shape (64, 1, T//4)
            nn.BatchNorm2d(64),  # output shape (64, 1, T//8)
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), bias=False),  # output shape (16, 1, T//16)
            nn.BatchNorm2d(64),  # output shape (16, 1, T//16)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),  # output shape (64, 1, T//16)

            nn.ZeroPad2d((4, 3, 0, 0)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 7), bias=False),  # output shape (64, 1, T//4)
            nn.BatchNorm2d(128),  # output shape (64, 1, T//8)
            nn.ELU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), bias=False),  # output shape (16, 1, T//16)
            nn.BatchNorm2d(128),  # output shape (16, 1, T//16)
            nn.ELU(),
            nn.AvgPool2d((1, 2))  # output shape (64, 1, T//16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1792, 256),
            nn.ELU(),
            nn.Linear(256, classes_num),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # f c t
        x = self.block_1(x)
        x = self.block_2(x)
        x1 = self.block_3(x)
        x2 = self.res_3(x.clone())
        x = x1 + x2

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EEGNet(nn.Module):
    def __init__(self, classes_num, in_channels=1, electrodes=127, drop_out=0.25):
        super(EEGNet, self).__init__()

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((1, 2, 0, 0)),  # left, right, top, bottom of 2D img
            nn.Conv2d(
                in_channels=in_channels,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 64),  # filter size
                # kernel_size=(1, 3),  # filter size 1111111111111 short T
                bias=False
            ),  # output shape (b, 8, C, T)
            # nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters  (16, C, T)
                kernel_size=(electrodes, 1),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            # nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//2) 1111111111111 short T
            nn.Dropout(drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                # kernel_size=(1, 3),  # 1111111111111 short T
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            # nn.AvgPool2d((1, 2)),  # 1111111111111 short T
            nn.Dropout(drop_out)
        )

        self.out = nn.Linear(192, classes_num)

    def forward(self, x):
        # f c t needed
        # x = x.transpose(2, 3)  # [b 1 c t] needed
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        return F.softmax(x, dim=1)
