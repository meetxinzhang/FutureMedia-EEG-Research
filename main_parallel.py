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
from torch.utils.tensorboard import SummaryWriter
import time
import os
from train_test import XinTrainer
from data_pipeline.dataset_szu import ListDataset
from main_k_fold import k_fold_share
from model.field_flow_2p1 import FieldFlow2
from utils.my_tools import IterForever
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
# random.seed = 2022
# torch.manual_seed(2022)
# torch.cuda.manual_seed(2022)


# torch.cuda.set_device(7)
batch_size = 16
accumulation_steps = 4  # to accumulate gradient when you want to set larger batch_size but out of memory.
n_epoch = 50
k = 5
learn_rate = 0.1

id_exp = 'EEGNet-trial-ff2-on-cwt-50e01l64b'
data_path = '../../Datasets/CVPR2021-02785/pkl_trial_cwt_from_1024'
# data_path = '../../Datasets/sz_eeg/pkl_cwt_torch'
time_exp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
init_state = './log/checkpoint/rank0_init_' + id_exp + '.pkl'
devices_id = [6, 7]
main_gpu_rank = 0


def main_func(gpu_rank, device_id, fold_rank, train_dataset: ListDataset, valid_dataset: ListDataset):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=len(devices_id), rank=gpu_rank)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    dist.barrier()

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = tud.distributed.DistributedSampler(train_dataset, rank=gpu_rank, shuffle=True)
    valid_sampler = tud.distributed.DistributedSampler(valid_dataset, rank=gpu_rank, shuffle=True)
    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = tud.BatchSampler(train_sampler, batch_size, drop_last=True)
    valid_batch_sampler = tud.BatchSampler(valid_sampler, batch_size, drop_last=True)

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_loader = tud.DataLoader(train_dataset, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=1)
    valid_loader = tud.DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, pin_memory=True, num_workers=1)
    val_iterable = IterForever(valid_loader)

    # ff = EEGNet(classes_num=40, in_channels=1, electrodes=96, drop_out=0.1).cuda()
    # ff = ConvTransformer(num_classes=40, in_channels=3, hid_channels=8, num_heads=2,
    #                      ffd_channels=16, deep_channels=16, size=32, T=63, depth=1, drop=0.2).cuda()
    ff = FieldFlow2(channels=96, early_drop=0.2, late_drop=0.1).to(device)
    # ff = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ff).to(device)
    ff = torch.nn.parallel.DistributedDataParallel(ff)

    summary = None
    if gpu_rank == main_gpu_rank:
        torch.save(ff.state_dict(), init_state)
        summary = SummaryWriter(log_dir='./log/' + time_exp + id_exp + '/' + str(fold_rank) + '_fold/')

    dist.barrier()  # waite the main process
    # 这里注意，一定要指定 map_location 参数，否则会导致第一块GPU占用更多资源
    ff.load_state_dict(torch.load(init_state, map_location=device))
    print(str(gpu_rank) + ' rank is initialized')

    optim_paras = [p for p in ff.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(optim_paras, lr=learn_rate, momentum=0.9, weight_decay=0.001, nesterov=True)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)  # 设定优优化器更新的时刻表

    xin = XinTrainer(n_epoch=n_epoch, model=ff, optimizer=optimizer, batch_size=batch_size,
                     train_loader=train_loader, val_iterable=val_iterable,
                     summary=summary, gpu_rank=gpu_rank, device=device)
    for epoch in range(1, n_epoch + 1):
        train_sampler.set_epoch(epoch)  # to update epoch related random seed
        xin.train_period_parallel(epoch=epoch, accumulation=accumulation_steps)
        # lr_scheduler.step()  # 更新学习率

    if gpu_rank == main_gpu_rank:
        summary.flush()
        summary.close()
        if os.path.exists(init_state):
            os.remove(init_state)
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # for fold, (train_ids, valid_ids) in enumerate(k_fold.split(dataset)):
    #     train_sampler = SubsetRandomSampler(train_ids)
    #     valid_sampler = SubsetRandomSampler(valid_ids)
    #     train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4,
    #                               prefetch_factor=1)
    #     valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1,
    #                               prefetch_factor=1)
    for (fold, train_files, valid_files) in k_fold_share(data_path, k):
        train_set = ListDataset(train_files)
        valid_set = ListDataset(valid_files)

        print(f'FOLD {fold}')
        print(len(train_files), len(valid_files), '--------------------------------')

        process = []
        for rank, device in enumerate(devices_id):
            p = mp.Process(target=main_func, args=(rank, device, fold, train_set, valid_set))
            p.start()
            process.append(p)
        for p in process:
            p.join()
        # mp.spawn(main_func, nprocs=2, args=(0, fold, train_set, valid_set))

        print('done')
