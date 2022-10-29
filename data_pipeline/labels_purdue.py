# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/5/21 7:56 PM
@desc:
"""
import numpy as np

classes = {"n02106662": 0,
           "n02124075": 1,
           "n02281787": 2,
           "n02389026": 3,
           "n02492035": 4,
           "n02504458": 5,
           "n02510455": 6,
           "n02607072": 7,
           "n02690373": 8,
           "n02906734": 9,
           "n02951358": 10,
           "n02992529": 11,
           "n03063599": 12,
           "n03100240": 13,
           "n03180011": 14,
           "n03272010": 15,
           "n03272562": 16,
           "n03297495": 17,
           "n03376595": 18,
           "n03445777": 19,
           "n03452741": 20,
           "n03584829": 21,
           "n03590841": 22,
           "n03709823": 23,
           "n03773504": 24,
           "n03775071": 25,
           "n03792782": 26,
           "n03792972": 27,
           "n03877472": 28,
           "n03888257": 29,
           "n03982430": 30,
           "n04044716": 31,
           "n04069434": 32,
           "n04086273": 33,
           "n04120489": 34,
           "n04555897": 35,
           "n07753592": 36,
           "n07873807": 37,
           "n11939491": 38,
           "n13054560": 39}


def get_one_hot(idx):
    one_hot = np.zeros([40], dtype=np.int)
    print(one_hot, idx)
    one_hot[idx] = 1
    return one_hot


class LabelReader(object):
    def __init__(self, one_hot=False):
        self.file_path = None  # '../../Datasets/CVPR2021-02785/design/run-00.txt'
        self.one_hot = one_hot
        self.lines = None

    def read(self):
        with open(self.file_path) as f:
            lines = f.readlines()
        return [line.split('_')[0] for line in lines]

    def get_set(self, file_path):
        if self.file_path == file_path:
            return [classes[e] for e in self.lines]
        else:
            self.file_path = file_path
            self.lines = self.read()
            return [classes[e] for e in self.lines]

    def get_item(self, file_path, sample_idx):
        if self.file_path == file_path:
            idx = classes[self.lines[sample_idx]]

        else:
            self.file_path = file_path
            self.lines = self.read()
            idx = classes[self.lines[sample_idx]]
        if self.one_hot:
            return get_one_hot(idx)
        else:
            return idx
