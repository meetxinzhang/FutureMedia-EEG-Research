# encoding: utf-8
"""
@author: Xin Zhang
@contact: meetdevin.zh@outlook.com
@file: my_tools.py
@time: 3/6/19
@desc: 自定义异常类
"""

import glob
import platform
import torch
import collections
from itertools import repeat


def file_scanf(path, contains, endswith, sub_ratio=1):
    files = glob.glob(path + '/*')
    if platform.system().lower() == 'windows':
        files = [f.replace('\\', '/') for f in files]
    disallowed_file_endings = (".gitignore", ".DS_Store")
    _input_files = files[:int(len(files) * sub_ratio)]
    return list(filter(lambda x: contains in x and x.endswith(endswith), _input_files))


def file_scanf2(path, contains, endswith, sub_ratio=1):
    files = glob.glob(path + '/*')
    input_files = []
    for f in files[:int(len(files) * sub_ratio)]:
        if not any([c in f for c in contains]):
            continue
        if not f.endswith(endswith):
            continue
        if platform.system().lower() == 'windows':
            f.replace('\\', '/')
        input_files.append(f)
    return input_files


class IterForever:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.it = iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            del self.it
            self.it = iter(self.dataloader)
            return next(self.it)


class ExceptionPassing(Exception):
    """
    继承自基类 Exception
    """
    def __init__(self, *message, expression=None):
        super(ExceptionPassing, self).__init__(message)
        self.expression = expression
        self.message = str.join('', [str(a) for a in message])


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LabelSmoothing(torch.nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, torch.autograd.Variable(true_dist, requires_grad=False))