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
    def __init__(self, classes_num, channels=127, drop_out=0.25):
        super(ComplexEEGNet, self).__init__()

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
        self.apply(_init_weights)

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


class EEGNet(nn.Module):
    def __init__(self, classes_num, in_channels=1, electrodes=127, drop_out=0.25):
        super(EEGNet, self).__init__()

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),  # left, right, top, bottom of 2D img
            nn.Conv2d(
                in_channels=in_channels,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=False
            ),  # output shape (b, 8, C, T)
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
            nn.Dropout(drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
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
            nn.Dropout(drop_out)
        )

        self.out = nn.Linear(256, classes_num)

    def forward(self, x):
        # x = x.transpose(2, 3)  # [b 1 c t] needed
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        return F.softmax(x, dim=1)

