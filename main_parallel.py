# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/22 9:23
 @desc:
"""

import torch
import torch.utils.data as tud
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as torch_lr
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
import os
from agent_train import XinTrainer
from data_pipeline.dataset_szu import ListDataset
from data_pipeline.data_loader_x import DataLoaderX
# from model.field_flow_2p1 import FieldFlow2
from model.eeg_net import EEGNet, ComplexEEGNet
# from model.lstm_1dcnn_mlp_syncnet import SyncNet
# from model.eeg_channel_net import EEGChannelNet
# from model.resnet_arcface import resnet18 as resnet2d
from utils.my_tools import file_scanf2, mkdirs

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '7890'
# os.environ['NCCL_LL_THRESHOLD'] = '0'
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IB_DISABLE'] = '1'
torch.manual_seed(1994)
torch.cuda.manual_seed(1994)

id_exp = 'EEGNet-SZ-trial-subj1-cwt-05s-512-8bs'
data_path = '/data1/zhangwuxia/Datasets/SZEEG2022/pkl_trial_cwt_subj1_1s_1000'
# data_path = '/data1/zhangwuxia/Datasets/PD/pkl_trial_cwt_1s_1024'
time_exp = '2023-04-17--15-40'
init_state = './log/checkpoint/rank0_init_' + id_exp + '.pkl'

device_list = [0, 1, 2, 3, 4, 5]
main_gpu_rank = 0
train_loaders = 8
valid_loaders = 8

batch_size = 8
accumulation_steps = 1  # to accumulate gradient when you want to set larger batch_size but out of memory.
n_epoch = 100
k = 5
learn_rate = 0.01


def main_func(gpu_rank, device_id, fold_rank, train_dataset: ListDataset, valid_dataset: ListDataset):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=len(device_list), rank=gpu_rank)
    the_device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(the_device)

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = tud.distributed.DistributedSampler(train_dataset, rank=gpu_rank, shuffle=True)
    valid_sampler = tud.distributed.DistributedSampler(valid_dataset, rank=gpu_rank, shuffle=True)
    # 将样本索引每batch_size个元素组成一个list
    train_b_s = tud.BatchSampler(train_sampler, batch_size, drop_last=True)
    valid_b_s = tud.BatchSampler(valid_sampler, batch_size, drop_last=True)

    # train_loader = tud.DataLoader(train_dataset, batch_sampler=train_b_s, pin_memory=True, num_workers=train_loaders)
    # valid_loader = tud.DataLoader(valid_dataset, batch_sampler=valid_b_s, pin_memory=True, num_workers=valid_loaders)
    train_loader = DataLoaderX(local_rank=device_id, dataset=train_dataset, batch_sampler=train_b_s, pin_memory=True,
                               num_workers=train_loaders)
    valid_loader = DataLoaderX(local_rank=device_id, dataset=valid_dataset, batch_sampler=valid_b_s, pin_memory=True,
                               num_workers=valid_loaders)

    # ff = EEGChannelNet(in_channels=30, input_height=96, input_width=512, num_classes=40,
    #                  num_spatial_layers=3, spatial_stride=(2, 1), num_residual_blocks=3, down_kernel=3, down_stride=2)
    # ff = LSTM(classes=40, input_size=96, depth=3)
    ff = EEGNet(classes_num=40, in_channels=30, electrodes=127, drop_out=0.1).to(the_device)
    # ff = ComplexEEGNet(classes_num=40, in_channels=30, electrodes=127, drop_out=0.1).to(the_device)
    # ff = ConvTransformer(num_classes=40, in_channels=3, hid_channels=8, num_heads=2,
    #                      ffd_channels=16, deep_channels=16, size=32, T=63, depth=1, drop=0.2).cuda()
    # ff = FieldFlow2(channels=127, early_drop=0.2, late_drop=0.1).to(the_device)
    # ff = ResNet1D(in_channels=96, classes=40).to(the_device)
    # ff = MLP2layers(in_features=96, hidden_size=128, classes=40).to(the_device)
    # ff = SyncNet(in_channels=30, num_layers_in_fc_layers=40)
    # ff = resnet2d(pretrained=False, n_classes=40, input_channels=30).to(the_device)
    ff = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ff).to(the_device)
    ff = torch.nn.parallel.DistributedDataParallel(ff, broadcast_buffers=False, device_ids=[device_id],
                                                   output_device=device_id)

    summary = None
    if gpu_rank == main_gpu_rank:
        torch.save(ff.state_dict(), init_state)
        summary = SummaryWriter(log_dir='./log/' + id_exp + '/' + time_exp + '---' + str(fold_rank) + '_fold/')
    dist.barrier()  # waite the main process
    ff.load_state_dict(torch.load(init_state, map_location=the_device))  # 指定 map_location 参数，否则第一块GPU占用更多资源
    print(str(gpu_rank) + ' rank is initialized')

    optim_paras = [p for p in ff.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(optim_paras, lr=learn_rate, momentum=0.9, weight_decay=0.001, nesterov=True)
    optimizer = torch.optim.SGD(optim_paras, lr=learn_rate, weight_decay=0.001, momentum=0.9)
    # lr_scheduler = torch_lr.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, verbose=True,
    #                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.001,
    #                                           eps=1e-08)
    lr_scheduler = torch_lr.StepLR(optimizer, step_size=10, gamma=0.4, last_epoch=-1)

    xin = XinTrainer(n_epoch=n_epoch, model=ff, train_loader=train_loader, val_loader=valid_loader,
                     optimizer=optimizer, batch_size=batch_size, lr_scheduler=lr_scheduler,
                     gpu_rank=gpu_rank, device=the_device, id_exp=id_exp, summary=summary)
    for epoch in range(1, n_epoch + 1):
        train_sampler.set_epoch(epoch)  # to update epoch related random seed
        xin.train_period_parallel(epoch=epoch, accumulation=accumulation_steps)
        # if epoch % 1 == 0:
        valid_sampler.set_epoch(epoch)
        xin.validate_epoch_parallel(epoch=epoch)

    if gpu_rank == main_gpu_rank:
        summary.flush()
        summary.close()
        if os.path.exists(init_state):
            os.remove(init_state)
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    mkdirs(['./log/image/' + id_exp + '/' + time_exp, './log/checkpoint/' + id_exp, './log/' + id_exp])
    # filepaths = file_scanf2(path=data_path, contains=['-1-00_', '-1-00_', '-1-01_', '-1-02_', '-1-03_', '-1-04_'],
    #                         endswith='.pkl')
    filepaths = file_scanf2(path=data_path, contains=['run'], endswith='.pkl')
    labels = [int(f.split('_')[-1].replace('.pkl', '')) for f in filepaths]

    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1994)
    dataset = ListDataset(filepaths)
    print(len(filepaths), ' total')

    for fold, (train_idx, valid_idx) in enumerate(k_fold.split(X=dataset, y=labels)):
        train_set = tud.Subset(dataset, train_idx)
        valid_set = tud.Subset(dataset, valid_idx)

        print(f'FOLD {fold}')
        print(len(train_idx), len(valid_idx), '--------------------------------')

        process = []
        for rank, device in enumerate(device_list):
            p = mp.Process(target=main_func, args=(rank, device, fold, train_set, valid_set))
            p.start()
            process.append(p)
        for p in process:
            p.join()
        # mp.spawn(main_func, nprocs=len(device_list), args=(fold, train_set, valid_set))

        print('done')
