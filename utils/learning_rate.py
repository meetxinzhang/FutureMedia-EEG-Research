# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/7/21 22:20 PM
@desc:
         see http://nlp.seas.harvard.edu/2018/04/03/attention.html for more details.
"""
import torch


class NoamOpt:
    """optimizer wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        return rate

    def rate(self, step=None):
        """Implement `learning_rate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


