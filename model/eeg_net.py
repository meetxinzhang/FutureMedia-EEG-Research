# encoding: utf-8
"""
Forked from https://github.com/HANYIIK/EEGNet-pytorch/blob/master/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, classes_num, channels=127, drop_out=0.25):
        super(EEGNet, self).__init__()

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((16, 15, 0, 0)),  # left, right, top, bottom of 2D img
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=63,  # num_filters
                kernel_size=(1, 32),  # filter size
                bias=False
            ),  # output shape (63, C, T)
            nn.BatchNorm2d(63),  # output shape (63, C, T)
            nn.ELU()
        )

        self.block_11 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((8, 7, 0, 0)),  # left, right, top, bottom of 2D img
            nn.Conv2d(
                in_channels=64,  # input shape (64, C, T)
                out_channels=64,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=64,
                bias=False
            ),  # output shape (64, C, T)
            nn.BatchNorm2d(64),  # output shape (64, C, T)
            nn.ELU()
        )
        self.pool1 = nn.AvgPool2d((1, 2))  # output shape (64, C, T//2)
        self.drop1 = nn.Dropout(drop_out)  # output shape (64, C, T//2)

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # input shape (64, C, T//2)
                out_channels=128,  # num_filters  (128, C, T//2)
                kernel_size=(channels, 1),  # filter size
                groups=64,
                bias=False
            ),  # output shape (128, 1, T//2)
            nn.BatchNorm2d(128),  # output shape (128, 1, T//2)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),  # output shape (128, 1, T//4)
            nn.Dropout(drop_out)  # output shape (128, 1, T//4)
        )

        self.block_22 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=128,  # input shape (128, 1, T//4)
                out_channels=128,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=128,
                bias=False
            )  # output shape (128, 1, T//4)
        )

        self.pool22 = nn.AvgPool2d((1, 2))  # output shape (128, 1, T//8)
        self.drop22 = nn.Dropout(drop_out)  # output shape (128, 1, T//8)

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,  # input shape (128, 1, T//8)
                out_channels=64,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (64, 1, T//4)
            nn.BatchNorm2d(64),  # output shape (64, 1, T//8)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),  # output shape (64, 1, T//16)
            nn.Dropout(drop_out),

            nn.Conv2d(
                in_channels=64,  # input shape (64, 1, T//16)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//16)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//16)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//32)
            nn.Dropout(drop_out)
        )

        self.out = nn.Linear((16 * 15), classes_num)

    def forward(self, x):
        x = x.transpose(2, 3)

        # x = self.block_1(x)
        x = torch.cat([x, self.block_1(x)], dim=1)
        x = x + self.block_11(x)
        x = self.drop1(self.pool1(x))

        x = self.block_2(x)
        x = x + self.block_22(x)
        x = self.drop22(self.pool22(x))

        x = self.block_3(x)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        return F.softmax(x, dim=1)

