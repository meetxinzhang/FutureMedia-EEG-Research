# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/28 21:49
 @name: 
 @desc:
"""


def ziyan_read(file_path):
    with open(file_path) as f:
        stim = []
        for line in f.readline():
            if line.strip().startswith('Stimulus'):
                line = line.strip().split(',')
                classes = int(line[1][-2:])
                time = int(line[2].strip())
                stim.append([time, classes])
    return stim

