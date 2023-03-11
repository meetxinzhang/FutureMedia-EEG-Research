# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 12/5/21 7:56 PM
 @name: 
 @desc:
"""
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from data_pipeline.mne_reader import MNEReader
from utils.my_tools import file_scanf
from pre_process.difference import trial_average

parallel_jobs = 6


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
    # print(one_hot, idx)
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


def thread_read_write(x, y, pkl_filename):
    """Writes and dumps the processed pkl file for each stimulus(or called subject).
    [time=2999, channels=127], y
    """
    with open(pkl_filename + '.pkl', 'wb') as file:
        pickle.dump(x, file)
        pickle.dump(y, file)


def go_through(bdf_filenames, label_dir, pkl_path):
    bdf_reader = MNEReader(filetype='bdf', resample=1024, length=512, stim_channel='Status',
                           exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'])
    label_reader = LabelReader(one_hot=False)

    for f in tqdm(bdf_filenames, desc=' Total', position=0, leave=True, colour='YELLOW', ncols=80):
        xs, times = bdf_reader.get_set(file_path=f)
        number = f.split('-')[-1].split('.')[0]  # ../imagenet40-1000-1-02.bdf
        ys = label_reader.get_set(file_path=label_dir + '/' + 'run-' + number + '.txt')
        assert len(times) == len(ys)
        assert np.shape(xs[0]) == (512, 96)  # [length, channels]

        x = np.reshape(x, (len(x)*512, 96))
        x = trial_average(x, axis=0)
        x = np.reshape(x, (-1, 512, 96))

        name = f.split('/')[-1].replace('.bdf', '')
        Parallel(n_jobs=parallel_jobs)(
            delayed(thread_read_write)(xs[i], ys[i], pkl_path+name+'_'+str(i)+'_'+str(times[i])+'_'+str(ys[i]))
            for i in tqdm(range(len(ys)), desc=' write '+name, position=1, leave=False, colour='WHITE', ncols=80))


if __name__ == "__main__":
    path = '../../../Datasets/CVPR2021-02785'
    bdf_dir = path + '/data'
    label_dir = path + '/design'
    # self.image_path = path + '/stimuli'

    bdf_filenames = file_scanf(bdf_dir, contains='1000-1', endswith='.bdf')
    go_through(bdf_filenames, label_dir, pkl_path=path + '/pkl/')

