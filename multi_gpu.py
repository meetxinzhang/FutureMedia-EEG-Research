# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/22 9:23
 @desc:
"""
from utils.my_tools import file_scanf
import torch.utils.data as tud
from sklearn.model_selection import KFold
import torch
from torch.multiprocessing import Process
import torch.distributed as dist
from train_test import XinTrainer
from torch.utils.tensorboard import SummaryWriter
import time
import os
from data_pipeline.dataset_szu import ListDataset
from main_k_fold import kfold_loader
# from model.eeg_net import EEGNet
# from model.eeg_net import ComplexEEGNet
# from model.conv_transformer import ConvTransformer
from model.field_flow_2p1 import FieldFlow2
from utils.my_tools import IterForever

# random.seed = 2022
# torch.manual_seed(2022)
# torch.cuda.manual_seed(2022)


torch.cuda.set_device(7)
batch_size = 16
accumulation_steps = 4  # to accumulate gradient when you want to set larger batch_size but out of memory.
n_epoch = 50
k = 5
learn_rate = 0.01

id_exp = 'EEGNet-trial-ff2-on-cwt-50e01l64b'
path = '../../Datasets/CVPR2021-02785/pkl_trial_cwt_from_1024'
# path = '../../Datasets/sz_eeg/pkl_cwt_torch'
time_exp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


# k_fold = KFold(n_splits=k, shuffle=True)
# filepaths = file_scanf(path=path, contains='i', endswith='.pkl')
# dataset = ListDataset(filepaths)


def main_func(gpu_rank, main_gpu, fold_rank, learn_rate, train_dataset: ListDataset, valid_dataset: ListDataset):
    torch.cuda.set_device(gpu_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=4, rank=gpu_rank)
    dist.barrier()
    device = torch.device('cuda')

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = tud.distributed.DistributedSampler(train_dataset)
    valid_sampler = tud.distributed.DistributedSampler(valid_dataset)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    train_loader = tud.DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=True, num_workers=nw,
                                  shuffle=True, batch_size=batch_size)
    valid_loader = tud.DataLoader(valid_dataset, batch_sampler=valid_sampler, pin_memory=True, num_workers=1,
                                  shuffle=True, batch_size=batch_size)
    val_iterable = IterForever(valid_loader)

    # ff = EEGNet(classes_num=40, in_channels=1, electrodes=96, drop_out=0.1).cuda()
    # ff = ConvTransformer(num_classes=40, in_channels=3, hid_channels=8, num_heads=2,
    #                      ffd_channels=16, deep_channels=16, size=32, T=63, depth=1, drop=0.2).cuda()
    ff = FieldFlow2(channels=96, early_drop=0.2, late_drop=0.1).cuda()
    init_state = 'log/checkpoint/rank0_init_' + time_exp + id_exp + '.pkl'

    if gpu_rank == main_gpu:
        torch.save(ff.state_dict(), init_state)
        print(f'FOLD {fold_rank}')
        train_num = len(train_dataset)
        print(train_num, len(valid_dataset), '--------------------------------')
        summary = SummaryWriter(log_dir='./log/' + time_exp + id_exp + '/' + str(fold_rank) + '_fold/')
    else:
        summary = None
        dist.barrier()
        # 这里注意，一定要指定 map_location 参数，否则会导致第一块GPU占用更多资源
        ff.load_state_dict(torch.load(init_state, map_location=device))

    ff = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ff).cuda()
    ff = torch.nn.parallel.DistributedDataParallel(ff, device_ids=[gpu_rank])

    optim_paras = [p for p in ff.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(optim_paras, lr=learn_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)  # 设定优优化器更新的时刻表

    xin = XinTrainer(n_epoch=n_epoch, model=ff, optimizer=optimizer, batch_size=batch_size, main_gpu=0, device=device)
    for epoch in range(1, n_epoch + 1):
        train_sampler.set_epoch(epoch)
        xin.train_period_parallel(epoch=epoch, accumulation=accumulation_steps, summary=summary, gpu_rank=rank,
                                  train_loader=train_loader, val_iterable=val_iterable)
        lr_scheduler.step()  # 更新学习率
    summary.flush()
    summary.close()

    if gpu_rank == 0:
        if os.path.exists(init_state):
            os.remove(init_state)
    dist.destroy_process_group()


if __name__ == '__main__':
    # for fold, (train_ids, valid_ids) in enumerate(k_fold.split(dataset)):
    #     train_sampler = SubsetRandomSampler(train_ids)
    #     valid_sampler = SubsetRandomSampler(valid_ids)
    #     train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4,
    #                               prefetch_factor=1)
    #     valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1,
    #                               prefetch_factor=1)
    for (fold, train_files, valid_files) in kfold_loader(path, k):
        train_set = ListDataset(train_files)
        valid_set = ListDataset(valid_files)

        process = []
        for rank in range(4):
            p = Process(target=main_func, args=())
            p.start()
            process.append(p)
        for p in process:
            p.join()

        print('done')
