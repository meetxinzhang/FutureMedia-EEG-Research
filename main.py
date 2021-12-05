# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/5/21 9:02 PM
@desc:
"""
from data_pipeline.dataset import BDFDataset
import torch

dataset = BDFDataset(CVPR2021_02785_path='/media/xin/Raid0/dataset/CVPR2021-02785')
loader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=1)


for x, label in loader:
    print(x.size())
    print(label)
