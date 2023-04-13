# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/28 21:47
 @name: 
 @desc:
"""
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from data_pipeline.mne_reader import MNEReader
from utils.my_tools import file_scanf2
import numpy as np
import einops
from pre_process.difference import trial_average
from pre_process.aep import azim_proj, gen_images
from pre_process.time_frequency import three_bands, cwt_scipy, cwt_pywt


def ziyan_read(file_path):
    # read labels and stimulus from .Markers file which created by Ziyan He
    with open(file_path) as f:
        stim = []
        y = []
        for line in f.readlines():
            if line.strip().startswith('Stimulus'):
                line = line.strip().split(',')
                classes = int(line[1][-2:])  # 'S 17'
                time = int(line[2].strip())  # ' 39958'
                stim.append(time)
                y.append(classes)
    return stim, y


def thread_write(x, y, pos, pkl_filename):
    """Writes and dumps the processed pkl file for each stimulus(or called subject).
    [time, channels=127], y
    """
    # x = x[1000:, :]
    assert np.shape(x) == (1000, 127)

    # AEP
    # x = three_bands(x)  # [t=24, 3*127]
    # locs_2d = np.array([azim_proj(e) for e in pos])
    # x = gen_images(locs=locs_2d, features=x, len_grid=20, normalize=True).squeeze()  # [time, colors=1, W, H]
    # assert np.shape(x) == (24, 3, 20, 20)

    # time-spectrum
    # x = downsample(x, ratio=4)
    # specs = []  # [127, f=40, t=101]
    # for i in range(0, 127):
    #     # spectrum = signal2spectrum_stft(x[:, i])  # [f=40, t=101]
    #     spectrum = signal2spectrum_pywt_cwt(x[:, i])  # [40, 2000]
    #     specs.append(spectrum)

    # CWT
    # x = cwt_scipy(x)  # [c f=30 t=1000]
    # x = cwt_pywt(x)  # [c f=33 t=1000]
    # assert np.shape(x) == (127, 30, 1000)

    with open(pkl_filename + '.pkl', 'wb') as file:
        pickle.dump(x, file)
        pickle.dump(y, file)


def thread_read(label_file, pkl_path):
    edf_reader = MNEReader(filetype='edf', method='manual', length=1000, montage='brainproducts-RNP-BA-128')

    stim, y = ziyan_read(label_file)  # [frame_point], [class]
    x = edf_reader.get_set(file_path=label_file.replace('.Markers', '.edf'), stim_list=stim)
    pos = edf_reader.get_pos()
    assert len(x) == len(y)
    # x = x[:-1]  # For SZ2023, remove the last one of (2499, 127)
    # y = y[:-1]

    x = einops.rearrange(x, 'b t c -> (b t) c')
    x = trial_average(x, axis=0)
    x = einops.rearrange(x, '(b t) c -> b t c', t=1000)

    name = label_file.split('/')[-1].replace('.Markers', '')
    Parallel(n_jobs=6)(
        delayed(thread_write)(x[i], y[i], pos, pkl_path + '/' + name + '_' + str(i) + '_' + str(stim[i]) + '_' + str(y[i]))
        for i in tqdm(range(len(y)), desc=' write ', colour='RED', position=0, leave=False, ncols=80)
    )


if __name__ == "__main__":
    # path = 'G:/Datasets/SZFace2/EEG/10-17'
    path = '/data1/zhangwuxia/Datasets/SZEEG2022/Raw'
    label_filenames = file_scanf2(path, contains=['subject1', 'hzy', 'test1016'], endswith='.Markers')

    # go_through(label_filenames, pkl_path=path+'/pkl_cwt_torch/')
    Parallel(n_jobs=6)(
        delayed(thread_read)(
            f, pkl_path='/data1/zhangwuxia/Datasets/SZEEG2022/pkl_trial_subj1_1s_1000'
        )
        for f in tqdm(label_filenames, desc=' read ', colour='WHITE', position=1, leave=True, ncols=80)
    )








