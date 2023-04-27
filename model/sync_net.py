# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/4/26 22:21
 @desc:
"""
import torch
import torch.nn as nn
import numpy as np


class SyncNet(nn.Module):
    def __init__(self, channel, time, classes, dropout=0.1):
        super(SyncNet, self).__init__()
        K = min(10, channel)
        Nt = min(40, time)
        pool_size = Nt
        b = np.random.uniform(low=-0.05, high=0.05, size=(1, channel, K))
        omega = np.random.uniform(low=0, high=1, size=(1, 1, K))
        zeros = np.zeros(shape=(1, 1, K))
        phi_ini = np.random.normal(
            loc=0, scale=0.05, size=(1, channel - 1, K))
        phi = np.concatenate([zeros, phi_ini], axis=1)
        beta = np.random.uniform(low=0, high=0.05, size=(1, 1, K))
        t = np.reshape(range(-Nt // 2, Nt // 2), [Nt, 1, 1])
        tc = np.single(t)
        W_osc = b * np.cos(tc * omega + phi)
        W_decay = np.exp(-np.power(tc, 2) * beta)
        W = W_osc * W_decay
        W = np.transpose(W, (2, 1, 0))
        bias = np.zeros(shape=[K])
        self.net = nn.Sequential(nn.ConstantPad1d((Nt // 2, Nt // 2 - 1), 0),
                                 nn.Conv1d(in_channels=channel,
                                           out_channels=K,
                                           kernel_size=1,
                                           stride=1,
                                           bias=True),
                                 nn.MaxPool1d(kernel_size=pool_size,
                                              stride=pool_size),
                                 nn.ReLU()
                                 )
        self.net[1].weight.data = torch.FloatTensor(W)
        self.net[1].bias.data = torch.FloatTensor(bias)
        self.fc = nn.Linear((time // pool_size) * K, classes)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    # def cuda(self, gpuIndex):
    #     self.net = self.net.cuda(gpuIndex)
    #     self.fc = self.fc.cuda(gpuIndex)
    #     return self
