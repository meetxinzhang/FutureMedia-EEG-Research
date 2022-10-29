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
from data_pipeline.labels_purdue import LabelReader
from utils.my_tools import file_scanf

parallel_jobs = 6


def thread_read_write(x, y, pkl_filename):
    """Writes and dumps the processed pkl file for each stimulus(or called subject).
    [time=2999, channels=127], y
    """
    with open(pkl_filename + '.pkl', 'wb') as file:
        pickle.dump(x, file)
        pickle.dump(y, file)


def go_through(bdf_filenames, label_dir, pkl_path):
    bdf_reader = MNEReader(ftype='bdf', resample=1024, length=512, stim_channel='Status',
                           exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'])
    label_reader = LabelReader(one_hot=False)

    for f in tqdm(bdf_filenames):
        xs, times = bdf_reader.get_set(file_path=f)
        number = f.split('-')[-1].split('.')[0]  # ../imagenet40-1000-1-02.bdf
        ys = label_reader.get_set(file_path=label_dir + '/' + 'run-' + number + '.txt')
        assert len(times) == len(ys)

        name = f.split('/')[-1].replace('.bdf', '')
        print('do i/o: ' + name + '.edf to pkl use parallel:')
        Parallel(n_jobs=parallel_jobs)(
            delayed(thread_read_write)(xs[i], ys[i], pkl_path + name + '_' + str(times[i]) + '_' + str(ys[i]))
            for i in tqdm(range(len(ys))))


if __name__ == "__main__":
    path = 'E:/Datasets/CVPR2021-02785'
    bdf_dir = path + '/data'
    label_dir = path + '/design'
    # self.image_path = path + '/stimuli'

    bdf_filenames = file_scanf(bdf_dir, endswith='.bdf')
    go_through(bdf_filenames, label_dir, pkl_path=path + '/pkl/')
