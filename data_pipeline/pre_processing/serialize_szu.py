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

parallel_jobs = 6


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


def thread_read_write(x, y, pkl_filename):
    """Writes and dumps the processed pkl file for each stimulus(or called subject).
    [time=2999, channels=127], y
    """
    with open(pkl_filename + '.pkl', 'wb') as file:
        pickle.dump(x, file)
        pickle.dump(y, file)


def go_through(label_filenames, pkl_path):
    edf_reader = MNEReader(ftype='edf', method='manual', resample=None, length=3000)

    for f in label_filenames:
        stim, y = ziyan_read(f)  # [frame_point], [class]
        x = edf_reader.get_set(file_path=f.replace('.Markers', '.edf'), stim_list=stim)
        assert len(x) == len(y)

        name = f.split('/')[-1].replace('.Markers', '')
        print('do i/o: '+name+'.edf to pkl use parallel:')
        Parallel(n_jobs=parallel_jobs)(
            delayed(thread_read_write)(x[i], y[i], pkl_path+name+'_'+str(stim[i])+'_'+str(y[i]))
            for i in tqdm(range(len(y))))


if __name__ == "__main__":
    path = 'E:/Datasets/eegtest/run/'
    label_filenames = file_scanf(path, endswith='-seg-rmartifact.Markers')
    go_through(label_filenames, pkl_path=path+'/pkl/')






