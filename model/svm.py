# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/10 17:22
 @desc:
"""

from utils.my_tools import file_scanf2
from sklearn.svm import LinearSVC
import pickle
import numpy as np
import matplotlib.pyplot as plt


def sk_svm():
    svm = LinearSVC(fit_intercept=True, C=0.1, dual=True, max_iter=5000)
    return svm


if __name__ == '__main__':
    path = '../../../Datasets/CVPR2021-02785/pkl_pca_lda_from_spec'
    filenames = file_scanf2(path, contains=['1000-1'], endswith='.pkl')
    np.random.shuffle(filenames)
    total = len(filenames)

    svm = sk_svm()

    train_x = []  # [b 3780]
    train_y = []
    for i, file in enumerate(filenames[0:int(total*0.7)]):
        with open(file, 'rb') as f:
            x = pickle.load(f)  # [39,]
            y = int(pickle.load(f))  # [1,]
            train_x.append(x)
            train_y.append(y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print(np.shape(train_x), np.shape(train_y))
    svm.fit(train_x, train_y)

    print("\nSVM model: Y = w0 + w1*x1 + w2*x2")  # 分类超平面模型
    print('截距: w0={}'.format(svm.intercept_))  # w0: 截距, YouCans
    print('系数: w1={}'.format(svm.coef_))  # w1,w2: 系数, XUPT

    test_x = []  # [b 3780]
    test_y = []
    for i, file in enumerate(filenames[int(total * 0.7):]):
        with open(file, 'rb') as f:
            x = pickle.load(f)  # [39,]
            y = int(pickle.load(f))  # [1,]
            test_x.append(x)
            test_y.append(y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    print('\n分类准确度：{:.4f}'.format(svm.score(test_x, test_y)))  # 对训练集的分类准确度
