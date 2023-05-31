# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/24 17:56
 @name: 
 @desc:
"""
from utils.my_tools import file_scanf2, mkdirs
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import torch
from agent_train import XinTrainer
from torch.utils.tensorboard import SummaryWriter
# import os
# import numpy as np
from data_pipeline.dataset_szu import ListDataset
from model.eeg_net import EEGNet
# from model.field_flow_1p2 import FieldFlow1p2
# from model.conv_tsfm_lrp import ConvTransformer
# from model.field_flow_2p1 import FieldFlow2
# from model.video_tsfm import VideoTransformer

# random.seed = 2022
torch.manual_seed(2022)
torch.cuda.manual_seed(2022)

# def k_fold_share(path, k):
#     database = []
#     for i in range(0, 100):
#         # files_list = file_scanf(path, contains='run_'+str(i)+'_', endswith='.pkl')  # SZ
#         i = '0' + str(i) if i < 10 else str(i)
#         files_list = file_scanf(path, contains='1000-1-' + i + '_', endswith='.pkl')  # PD
#         np.random.shuffle(files_list)
#         database.append(files_list)
#
#     p = 0
#     while p < k:
#         train_set = []
#         test_set = []
#         for inset in database:
#             k_len = len(inset) // k
#             test_set += inset[p * k_len:(p + 1) * k_len]
#             train_set += inset[:p * k_len] + inset[(p + 1) * k_len:]
#             assert len(test_set) > 0
#         yield p, train_set, test_set
#         p += 1


device = torch.device(f"cuda:{7}")
batch_size = 4
accumulation_steps = 1  # to accumulate gradient when you want to set larger batch_size but out of memory.
n_epoch = 100
k = 5
lr = 0.001

id_exp = 'szeeg-0524-tu-2-2'
# path = '/data1/zhangwuxia/Datasets/PD/pkl_trial_aep_color_05s_1024'
path = '/data1/zhangwuxia/Datasets/SZEEG0524/pkl_cwt_tu_2-500'
time_exp = '2023-05-29--21-38'
mkdirs(['./log/image/' + id_exp + '/' + time_exp, './log/checkpoint/' + id_exp, './log/' + id_exp])

k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=2023)
filepaths = file_scanf2(path=path, contains=['0524'], endswith='.pkl')
# filepaths = file_scanf2(path=path, contains=['-1-00_', '-1-00_', '-1-01_', '-1-02_', '-1-03_', '-1-04_'],
#                         endswith='.pkl')
labels = [int(f.split('_')[-1].replace('.pkl', '')) for f in filepaths]
dataset = ListDataset(filepaths)
print(len(filepaths), ' total')

if __name__ == '__main__':
    for fold, (train_idx, valid_idx) in enumerate(k_fold.split(X=dataset, y=labels)):
        train_set = Subset(dataset, train_idx)
        valid_set = Subset(dataset, valid_idx)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, prefetch_factor=1)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=1, prefetch_factor=1)

        # for (fold, train_files, valid_files) in k_fold_share(path, k):
        #     print(len(train_files), len(valid_files))
        #     train_loader = DataLoader(ListDataset(train_files), batch_size=batch_size, num_workers=1, shuffle=False)
        #     valid_loader = DataLoader(ListDataset(valid_files), batch_size=batch_size, num_workers=1, shuffle=False)

        # ff = ComplexEEGNet(classes_num=40, in_channels=30, electrodes=127, drop_out=0.1).to(device)
        # ff = ConvTransformer(num_classes=40, in_channels=3, att_channels=64, num_heads=8,
        #                      ffd_channels=64, last_channels=16, time=23, depth=2, drop=0.2).to(device)
        # ff = VideoTransformer()
        # ff = FieldFlow1p2(channels=30, electrodes=127, time=512, early_drop=0.2, late_drop=0.05).to(device)
        ff = EEGNet(classes_num=40, in_channels=40, electrodes=127, drop_out=0.2).to(device)
        optim_paras = [p for p in ff.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(optim_paras, lr=lr, weight_decay=0.001, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)  # 设定优优化器更新的时刻表

        print(f'FOLD {fold}', len(train_idx), len(valid_idx))
        summary = SummaryWriter(log_dir='./log/' + id_exp + '/' + time_exp + '---' + str(fold) + '_fold/')

        xin = XinTrainer(n_epoch=n_epoch, model=ff, optimizer=optimizer, batch_size=batch_size, gpu_rank=0,
                         id_exp=id_exp + '/' + time_exp, device=device, train_loader=train_loader,
                         val_loader=valid_loader,
                         summary=summary, lr_scheduler=lr_scheduler)
        for epoch in range(1, n_epoch + 1):
            xin.train_period(epoch=epoch, accumulation=accumulation_steps, fold=fold, print_step=1)
            if epoch % 10 == 0:
                xin.val_period(epoch=epoch)
        summary.close()
        summary.flush()
        # torch.save(ff.state_dict(), './log/checkpoint/' + id_exp + '/' + time_exp + '---' + str(fold) + '.pkl')
    print('done')
