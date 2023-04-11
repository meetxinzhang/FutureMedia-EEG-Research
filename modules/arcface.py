# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/3 15:58
 @name: 
 @desc:
"""
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):"""
    def __init__(self, dim, num_classes, margin=0.3, scale=64, easy_margin=False, requires_grad=True):
        super().__init__()
        self.d = dim
        self.c = num_classes
        self.m = torch.tensor(margin)
        self.s = scale
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(self.c, self.d), requires_grad=requires_grad)
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(self.m)
        self.sin_m = torch.sin(self.m)
        self.th = torch.cos(torch.pi - self.m)
        self.mm = torch.sin(torch.pi - self.m) * self.m

    def forward(self, x, y):
        # [b, d]
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ+m) = cos θ cos m − sin θ sin m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Since cos(θ+m) is lower than cos(θ) when θ in [0, π − m], the constraint is more stringent for
            # classification.
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # output = cosine * 1.0  # make backward works
        one_hot = torch.zeros(cosine.size()).to(x.device)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # torch.where(out_i = {x_i if condition_i else y_i)
        output *= self.s
        return output
        # output = cosine * 1.0  # make backward works
        # batch_size = len(output)
        # output[range(batch_size), label] = phi[range(batch_size), label]
        # return output * self.s


class ArcEEG(nn.Module):
    """Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):"""
    def __init__(self, dim, num_classes, margin=0.3, scale=64, easy_margin=False, requires_grad=True):
        super().__init__()
        self.d = dim
        self.c = num_classes
        self.m = torch.tensor(margin)
        self.s = scale
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(self.c, 512), requires_grad=requires_grad)
        nn.init.xavier_uniform_(self.weight)

        self.embedding = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_features=dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512)
        )

        self.cos_m = torch.cos(self.m)
        self.sin_m = torch.sin(self.m)
        self.th = torch.cos(torch.pi - self.m)
        self.mm = torch.sin(torch.pi - self.m) * self.m

    def forward(self, x, y):
        # [b f t]  [b,]
        x = x.unfold(dimension=-1, size=8, step=4)  # [b c t s]
        x = einops.rearrange(x, "b f t s -> b t (f s)")
        x = self.embedding(x)  # [b t 512]
        b, t, _ = x.size()

        if self.training:
            memory_view = torch.index_select(self.weight, 0, y).unsqueeze(-1)  # [b fs] -> [b fs 1]
            support = torch.matmul(F.normalize(x), F.normalize(memory_view)).squeeze(-1)  # [b t fs]*[b fs 1]=[b t 1]
            targeted_idx = torch.argmax(support, dim=-1, keepdim=False)  # [b t] -> [b, ]

            X = []
            for sample, idx in zip(x.clone(), targeted_idx):
                X.append(torch.select(sample, dim=0, index=idx).unsqueeze(0))  # [1 fs]
            x = torch.cat(X, dim=0)  # [b fs]
            # mask = torch.zeros((b, t)).to(x.device)  # [b t]
            # mask.scatter_(1, targeted_idx.view(-1, 1).long(), 1)
            # mask = mask.ge(1).unsqueeze(-1)  # [b t 1]
            # x = torch.mul(x, mask)  # [b t fs] * [b t 1] = [b t fs]

            # ArcFace
            cosine = F.linear(F.normalize(x), F.normalize(self.weight))  # [b fs]*[c fs]=[b c]
            # sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            # phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ+m) = cos θ cos m − sin θ sin m
            #
            # if self.easy_margin:
            #     phi = torch.where(cosine > 0, phi, cosine)
            # else:
            #     # Since cos(θ+m) is lower than cos(θ) when θ in [0, π − m], the constraint is more stringent for
            #     # classification.
            #     phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            #
            # # output = cosine * 1.0  # make backward works
            # one_hot = torch.zeros(cosine.size()).to(x.device)  # [b class]
            # one_hot.scatter_(1, y.view(-1, 1).long().to(x.device), 1)
            # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # torch.where(out_i = {x_i if condition_i else y_i)
            # output *= self.s
            output = cosine*self.s
        else:
            multi_supports = F.linear(F.normalize(x), F.normalize(self.weight)).squeeze(-1)  # [b t fs]*[c fs]=[b t c]
            # TODO more strategies
            # output = torch.sum(multi_supports, dim=1, keepdim=False)  # [b cls]
            output, _ = torch.max(multi_supports, dim=1, keepdim=False)  # [b cls])
        return output
