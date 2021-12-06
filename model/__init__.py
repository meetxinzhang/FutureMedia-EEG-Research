# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/4/21 2:34 PM
@desc:
"""
import torch

gpu = torch.cuda.is_available()

if gpu:
    n = torch.cuda.device_count()
    for i in range(n):
        print(torch.cuda.get_device_name(i))

