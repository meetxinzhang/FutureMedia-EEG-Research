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
from utils.my_tools import file_scanf2
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import torch


# def lda_torch(x1, x2, device="cpu"):
#     """Forked from https://github.com/juliusbierk/torchlda/blob/master/torchlda.py
#     """
#     with torch.no_grad():
#         x1 = torch.tensor(x1, device=device, dtype=torch.float)
#         x2 = torch.tensor(x2, device=device, dtype=torch.float)
#
#         m1 = torch.mean(x1, dim=0)
#         m2 = torch.mean(x2, dim=0)
#         m = (len(x1) * m1 + len(x2) * m2) / (len(x1) + len(x2))
#
#         d1 = x1 - m1[None, :]
#         scatter1 = d1.t() @ d1
#         d2 = x2 - m2[None, :]
#         scatter2 = d2.t() @ d2
#         within_class_scatter = scatter1 + scatter2
#
#         d1 = m1 - m[None, :]
#         scatter1 = len(x1) * (d1.t() @ d1)
#         d2 = m2 - m[None, :]
#         scatter2 = len(x2) * (d2.t() @ d2)
#         between_class_scatter = scatter1 + scatter2
#
#         p = torch.pinverse(within_class_scatter) @ between_class_scatter
#         eigenvalues, eigenvectors = torch.eig(p, eigenvectors=True)
#         idx = torch.argsort(eigenvalues[:, 0], descending=True)
#         eigenvalues = eigenvalues[idx, 0]
#         eigenvectors = eigenvectors[idx, :]
#
#         return eigenvectors[0, :].cpu().numpy()


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


if __name__ == '__main__':
    path = '../../../Datasets/CVPR2021-02785/pkl_spec_from_2048'
    filenames = file_scanf2(path, contains=['1000-1-00', '1000-1-01', '1000-1-02', '1000-1-03', '1000-1-04',
                                            '1000-1-05', '1000-1-06', '1000-1-07', '1000-1-08', '1000-1-09'],
                            endswith='.pkl')

    lda = LinearDiscriminantAnalysis()
    pca = PCA(n_components=60, copy=False, svd_solver='auto')

    dataset = []  # [b 3780]
    labels = []

    for file in tqdm(filenames, desc=' process '):
        with open(file, 'rb') as f:
            x = pickle.load(f)  # [96, 33, 63]
            y = int(pickle.load(f))
            x = einops.rearrange(x, 'c f t -> t (c f)')
            x = pca.fit_transform(x)  # [63 60]
            x = einops.rearrange(x, 't c -> (t c)')  # [3780,]
            dataset.append(x)
            labels.append(y)

    # dataset_lda = lda.fit_transform(dataset, labels)  # [b=2000 3780]
    lda.fit(dataset, labels)
    dataset_lda = np.dot(dataset, lda.scalings_[:, 0:1024])
    del dataset

    go_through(dataset_lda, labels, filenames, pkl_path=path + '/../pkl_pca_lda_from_spec/')
