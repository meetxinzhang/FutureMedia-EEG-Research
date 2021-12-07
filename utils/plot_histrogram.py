# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/7/21 2:27 PM
@desc:
"""

import matplotlib.pyplot as plt
import numpy as np


with open('/media/xin/Raid0/ACS/gmx/interaction/ding/6ZER/1-10-hyhoh/apply_windows.log', 'r') as f:
    lines = f.readlines()

    times = []
    R = []
    L = []
    for line in lines:
        if line.startswith(' '):
            if line.startswith('  L'):
                L.append(line.split(' ')[2:])
            if line.startswith('  R'):
                R.append(line.split(' ')[2:])
        else:
            times.append(line)






x = np.array(["Runoob-1", "Runoob-2", "Runoob-3", "C-RUNOOB"])
y = np.array([12, 22, 6, 18])

plt.barh(x,y)
plt.show()

