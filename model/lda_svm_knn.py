# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/16 21:43
 @desc:
"""
import numpy as np
import einops
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from utils.my_tools import file_scanf2
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed


def thread_read_write(x, y, pkl_filename):
    """Writes and dumps the processed pkl file for each stimulus(or called subject).
    [time=2999, channels=127], y
    """
    with open(pkl_filename + '.pkl', 'wb') as file:
        pickle.dump(x, file)
        pickle.dump(y, file)


def go_through(xs, ys, names, pkl_path):
    Parallel(n_jobs=1)(
        delayed(thread_read_write)(xs[i], ys[i],
                                   pkl_path + names[i].split('/')[-1].replace('.pkl', '_'))
        for i in tqdm(range(len(ys)), desc=' writing ', colour='WHITE', ncols=80))


def pca_dataset(file_names):
    pca = PCA(n_components=60, copy=False, svd_solver='auto')

    dataset = []  # [b 3780]
    labels = []
    for file in tqdm(file_names, desc=' reading '):
        with open(file, 'rb') as f:
            x = pickle.load(f)  # [96, 33, 63]
            y = int(pickle.load(f))
            x = einops.rearrange(x, 'c f t -> t (c f)')
            x = pca.fit_transform(x)  # [63 60]
            x = einops.rearrange(x, 't c -> (t c)')  # [3780,]
            dataset.append(x)
            labels.append(y)
    return dataset, labels


if __name__ == '__main__':
    path = '../../../Datasets/CVPR2021-02785/pkl_spec_from_2048'
    file_names = file_scanf2(path, contains=['1000-1-00', '1000-1-01', '1000-1-02', '1000-1-03', '1000-1-04',
                                             '1000-1-05', '1000-1-06', '1000-1-07', '1000-1-08', '1000-1-09'],
                             endswith='.pkl')
    np.random.shuffle(file_names)
    total = len(file_names)
    train_x, train_y = pca_dataset(file_names=file_names[0:int(total * 0.6)])
    test_x, test_y = pca_dataset(file_names=file_names[int(total * 0.6):])

    lda = LinearDiscriminantAnalysis()
    lda.fit(train_x, train_y)

    print('\n分类准确度：{:.4f}'.format(lda.score(test_x, test_y)))

    test_x = np.dot(test_x, lda.scalings_)
    print(np.shape(test_x[0:1000]))
    # go_through(dataset_lda, labels, filenames, pkl_path=path + '/../pkl_pca_lda_from_spec/')

    svm = LinearSVC(fit_intercept=True, C=0.1, dual=True, max_iter=5000)
    svm.fit(test_x[0:1000], test_y[0:1000])

    print("\nSVM model: Y = w0 + w1*x1 + w2*x2")  # 分类超平面模型
    print('截距: w0={}'.format(svm.intercept_))  # w0: 截距, YouCans
    print('系数: w1={}'.format(svm.coef_))  # w1,w2: 系数, XUPT
    print('\n分类准确度：{:.4f}'.format(svm.score(test_x[1000:], test_y[1000:])))  # 对训练集的分类准确度
