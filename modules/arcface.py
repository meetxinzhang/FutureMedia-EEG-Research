# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/3 15:58
 @name: 
 @desc:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):"""
    def __init__(self, dim, num_classes, margin=0.3, scale=30, easy_margin=False):
        super().__init__()
        self.d = dim
        self.c = num_classes
        self.m = torch.tensor(margin)
        self.s = scale
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(self.c, self.d))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(self.m)
        self.sin_m = torch.sin(self.m)
        self.th = torch.cos(torch.pi - self.m)
        self.mm = torch.sin(torch.pi - self.m) * self.m

    def forward(self, x, y):
        # [b, d]
        cosine = F.linear(F.normalize(x, dim=1), F.normalize(self.weight))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ+m) = cos θ cos m − sin θ sin m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Since cos(θ+m) is lower than cos(θ) when θ in [0, π − m], the constraint is more stringent for
            # classification.
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size()).to(x.device)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
        # output = cosine * 1.0  # make backward works
        # batch_size = len(output)
        # output[range(batch_size), label] = phi[range(batch_size), label]
        # return output * self.s
