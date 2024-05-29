import torch.nn as nn
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding=128, in_channels=1, electrodes=127, drop_out=0.25):
        super(Encoder, self).__init__()

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),  # left, right, top, bottom of 2D img
            nn.Conv2d(
                in_channels=in_channels,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                # kernel_size=(1, 64),  # filter size
                kernel_size=(1, 16),  # filter size 1111111111111 short T
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
            # nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//2) 1111111111111 short T
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
            # nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.MaxPool2d((1, 2)),  # 1111111111111 short T
            nn.Dropout(drop_out)
        )

        self.block_4 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 7),  # filter size
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
            nn.MaxPool2d((1, 4)),  # output shape (16, 1, T//32)
            # nn.AvgPool2d((1, 4)),  # 1111111111111 short T
            nn.Dropout(drop_out)
        )

        self.out = nn.Sequential(
            # nn.Linear(464, 128),
            nn.Linear(480, 128),
            nn.Linear(128, embedding)
        )

    def forward(self, x):
        # b f c t needed
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class Clock(nn.Module):
    def __init__(self, in_channels=1, electrodes=127, drop_out=0.25):
        super(Clock, self).__init__()


class Association(nn.Module):
    def __init__(self, in_channels=1, electrodes=127, drop_out=0.25):
        super(Association, self).__init__()


class ArcMargin(nn.Module):
    """Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
     Forked from https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py"""
    def __init__(self, dim, num_classes, margin=0.3, scale=64, easy_margin=True, requires_grad=True):
        super().__init__()
        self.d = dim
        self.c = num_classes
        self.m = torch.tensor(margin)
        self.s = scale
        self.easy_margin = easy_margin

        self.cos_m = torch.cos(self.m)
        self.sin_m = torch.sin(self.m)
        self.th = torch.cos(torch.pi - self.m)
        self.mm = torch.sin(torch.pi - self.m) * self.m

    def forward(self, x, y, weight):
        # [b, d]
        cosine = F.linear(F.normalize(x), F.normalize(weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ+m) = cos θ cos m − sin θ sin m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Since cos(θ+m) is lower than cos(θ) when θ in [0, π − m], the constraint is more stringent for classification.
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(x.device)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # torch.where(out_i = {x_i if condition_i else y_i)
        output *= self.s
        return output


class ThinkNet(nn.Module):
    def __init__(self, classes=40, memories=40, embedding=128, margin=0.1, scale=64):
        super(ThinkNet, self).__init__()
        self.encoder = Encoder(embedding=embedding, in_channels=1, electrodes=96, drop_out=0.25)

        self.memory = nn.Parameter(torch.randn((memories, embedding)), requires_grad=True)
        nn.init.xavier_uniform_(self.memory)
        self.arc_margin = ArcMargin(dim=embedding, num_classes=classes, margin=margin, scale=scale)


    def forward(self, x, y):
        # [b f c t]  [b,]
        x = self.encoder(x)  # [b d]
        if self.training:
            output = self.arc_margin(x, y, self.memory)
        else:
            output = F.linear(F.normalize(x), F.normalize(self.memory))
        return F.softmax(output, dim=-1)




