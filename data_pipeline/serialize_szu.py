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
from utils.my_tools import file_scanf
import numpy as np
from pre_process.difference import downsample
# from pre_process.aep import azim_proj, gen_images
from pre_process.time_frequency import signal2spectrum_pywt_cwt
import torch
from pre_process.cwt_torch import CWT

parallel_jobs = 8


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


def thread_read_write(x, y, pos, pkl_filename):
    """Writes and dumps the processed pkl file for each stimulus(or called subject).
    [time, channels=127], y
    """
    assert np.shape(x) == (2000, 127)

    # AEP
    # locs_2d = np.array([azim_proj(e) for e in pos])
    # imgs = gen_images(locs=locs_2d, features=x, n_gridpoints=32, normalize=True).squeeze()  # [time, colors=1, W, H]

    # time-spectrum
    # x = downsample(x, ratio=4)
    # specs = []  # [127, f=40, t=101]
    # for i in range(0, 127):
    #     # spectrum = signal2spectrum_stft(x[:, i])  # [f=40, t=101]
    #     spectrum = signal2spectrum_pywt_cwt(x[:, i])  # [40, 2000]
    #     specs.append(spectrum)

    # cwt-torch
    pycwt = CWT(dj=0.0625, dt=1/1000, fmin=1, fmax=40, output_format="Magnitude")
    x = torch.tensor(x, dtype=torch.float32).permute(1, 0).unsqueeze(0)  # [1, c=127, t=2000]
    specs = pycwt(x)  # [1, 127, f=85, 2000]
    specs = specs.squeeze().numpy()  # [127, 85, 2000]
    # end

    with open(pkl_filename + '.pkl', 'wb') as file:
        pickle.dump(specs, file)
        pickle.dump(y, file)


def go_through(label_filenames, pkl_path):
    edf_reader = MNEReader(filetype='edf', method='manual', length=2000, pos_kind='brainproducts-RNP-BA-128')

    for f in tqdm(label_filenames, desc=' Total', position=0, leave=True, colour='YELLOW', ncols=80):
        stim, y = ziyan_read(f)  # [frame_point], [class]
        x = edf_reader.get_set(file_path=f.replace('.Markers', '.edf'), stim_list=stim)
        pos = edf_reader.get_pos()
        assert len(x) == len(y)
        assert np.shape(x[0]) == (2000, 127)

        # x = np.reshape(x, (len(x)*2000, 127))
        # x = trial_average(x, axis=0)
        # x = np.reshape(x, (-1, 2000, 127))

        name = f.split('/')[-1].replace('.Markers', '')
        Parallel(n_jobs=parallel_jobs)(
            delayed(thread_read_write)(x[i], y[i], pos, pkl_path + name+'_' + str(i) + '_' + str(stim[i])+'_'+str(y[i]))
            for i in tqdm(range(len(y)), desc=' write '+name, position=1, leave=False, colour='WHITE', ncols=80))


if __name__ == "__main__":
    # path = 'G:/Datasets/SZFace2/EEG/10-17'
    path = '../../../Datasets/sz_eeg'
    label_filenames = file_scanf(path, contains='run', endswith='.Markers')
    go_through(label_filenames, pkl_path=path+'/pkl_cwt_torch/')






