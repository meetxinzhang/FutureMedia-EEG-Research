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
# from model.field_flow_2p1 import FieldFlow2
# from model.eeg_net import EEGNet
# from model.lstm_1dcnn_mlp_syncnet import ResNet1D
# from model.eeg_channel_net import EEGChannelNet
from model.resnet_arcface import resnet18 as resnet2d
from utils.my_tools import IterForever, file_scanf2, mkdirs

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '7890'
os.environ['NCCL_LL_THRESHOLD'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
# random.seed = 2022
# torch.manual_seed(2022)
# torch.cuda.manual_seed(2022)

id_exp = 'ResNet18_2D-trial-cwt-1024-p50e01l64b'
data_path = '/data1/zhangwuxia/Datasets/pkl_trial_cwt_1s_1024'
# data_path = '/data0/zhangwuxia/zx/Datasets/pkl_trial_cwt_1024'
# data_path = '../../Datasets/sz_eeg/pkl_cwt_torch'
time_exp = '2023-04-03--15-08'
init_state = './log/checkpoint/rank0_init_' + id_exp + '.pkl'

device_list = [0, 1]
main_gpu_rank = 0
train_loaders = 4
valid_loaders = 2

batch_size = 32
accumulation_steps = 2  # to accumulate gradient when you want to set larger batch_size but out of memory.
n_epoch = 50
k = 5
learn_rate = 0.1


def main_func(gpu_rank, device_id, fold_rank, train_dataset: ListDataset, valid_dataset: ListDataset):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=len(device_list), rank=gpu_rank)
    the_device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(the_device)
    dist.barrier()

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = tud.distributed.DistributedSampler(train_dataset, rank=gpu_rank, shuffle=True)
    valid_sampler = tud.distributed.DistributedSampler(valid_dataset, rank=gpu_rank, shuffle=True)
    # 将样本索引每batch_size个元素组成一个list
    train_b_s = tud.BatchSampler(train_sampler, batch_size, drop_last=True)
    valid_b_s = tud.BatchSampler(valid_sampler, batch_size, drop_last=True)

    train_loader = tud.DataLoader(train_dataset, batch_sampler=train_b_s, pin_memory=True, num_workers=train_loaders)
    valid_loader = tud.DataLoader(valid_dataset, batch_sampler=valid_b_s, pin_memory=True, num_workers=valid_loaders)
    val_iterable = IterForever(valid_loader)

    # ff = EEGChannelNet(input_height=96, input_width=1024, in_channels=30, num_classes=40)
    # ff = LSTM(classes=40, input_size=96, depth=3)
    # ff = EEGNet(classes_num=40, in_channels=30, electrodes=96, drop_out=0.1).to(the_device)
    # ff = ConvTransformer(num_classes=40, in_channels=3, hid_channels=8, num_heads=2,
    #                      ffd_channels=16, deep_channels=16, size=32, T=63, depth=1, drop=0.2).cuda()
    # ff = FieldFlow2(channels=96, early_drop=0.2, late_drop=0.1).to(the_device)
    # ff = ResNet1D(in_channels=96, classes=40).to(the_device)
    # ff = MLP2layers(in_features=96, hidden_size=128, classes=40).to(the_device)
    # ff = SyncNet(in_channels=96, num_layers_in_fc_layers=40)
    ff = resnet2d(pretrained=False, n_classes=40, input_channels=30).to(the_device)
    ff = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ff).to(the_device)
    ff = torch.nn.parallel.DistributedDataParallel(ff)

    summary = None
    if gpu_rank == main_gpu_rank:
        torch.save(ff.state_dict(), init_state)
        summary = SummaryWriter(log_dir='./log/' + id_exp + '/' + time_exp + '---' + str(fold_rank) + '_fold/')
    dist.barrier()  # waite the main process
    # 这里注意，一定要指定 map_location 参数，否则会导致第一块GPU占用更多资源
    ff.load_state_dict(torch.load(init_state, map_location=the_device))
    print(str(gpu_rank) + ' rank is initialized')

    optim_paras = [p for p in ff.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(optim_paras, lr=learn_rate, momentum=0.9, weight_decay=0.001, nesterov=True)
    optimizer = torch.optim.Adam(optim_paras, lr=learn_rate, weight_decay=0.001)
    # lr_scheduler = torch_lr.ReduceLROnPlateau(optimizer, mode='min', factor=0.001, patience=10, verbose=True,
    #                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.001,
    #                                           eps=1e-08)
    lr_scheduler = torch_lr.StepLR(optimizer, step_size=10, gamma=0.6, last_epoch=-1)

    xin = XinTrainer(n_epoch=n_epoch, model=ff, train_loader=train_loader, val_iterable=val_iterable,
                     optimizer=optimizer, batch_size=batch_size, lr_shecduler=lr_scheduler,
                     gpu_rank=gpu_rank, device=the_device, id_exp=id_exp, summary=summary)
    for epoch in range(1, n_epoch + 1):
        train_sampler.set_epoch(epoch)  # to update epoch related random seed
        xin.train_period_parallel(epoch=epoch, accumulation=accumulation_steps)

    if gpu_rank == main_gpu_rank:
        summary.close()
        if os.path.exists(init_state):
            os.remove(init_state)
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    mkdirs(['./log/image/' + id_exp + '/' + time_exp, './log/checkpoint/' + id_exp, './log/' + id_exp])
    filepaths = file_scanf2(path=data_path, contains=['imagenet'], endswith='.pkl')
    labels = [int(f.split('_')[-1].replace('.pkl', '')) for f in filepaths]

    k_fold = StratifiedKFold(n_splits=k, shuffle=True)
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
        # mp.spawn(main_func, nprocs=2, args=(0, fold, train_set, valid_set))

        print('done')
