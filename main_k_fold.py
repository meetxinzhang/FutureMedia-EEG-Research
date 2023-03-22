# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/24 17:56
 @name: 
 @desc:
"""
from utils.my_tools import file_scanf
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import torch
from train_test import train_accumulate, test
from torch.utils.tensorboard import SummaryWriter
import time
from data_pipeline.dataset_szu import ListDataset
# from model.eeg_net import EEGNet
# from model.eeg_net import ComplexEEGNet
# from model.conv_transformer import ConvTransformer
from model.field_flow_2p1 import FieldFlow2
from utils.my_tools import IterForever
# random.seed = 2022
# torch.manual_seed(2022)
# torch.cuda.manual_seed(2022)


def kfold_loader(path, k):
    database = []
    for i in range(0, 100):
        # files_list = file_scanf(path, contains='run_'+str(i)+'_', endswith='.pkl')  # SZ
        i = '0' + str(i) if i < 10 else str(i)
        files_list = file_scanf(path, contains='1000-1-' + i + '_', endswith='.pkl')  # PD
        database.append(files_list)

    p = 0
    while p < k:
        train_set = []
        test_set = []
        for inset in database:
            k_len = len(inset)//k
            test_set += inset[p*k_len:(p+1)*k_len]
            train_set += inset[:p*k_len] + inset[(p+1)*k_len:]
            assert len(test_set) > 0
        yield p, train_set, test_set
        p += 1


torch.cuda.set_device(7)
batch_size = 8
accumulation_steps = 8  # to accumulate gradient when you want to set larger batch_size but out of memory.
n_epoch = 50
k = 5
lr = 0.01

id_exp = 'EEGNet-trial-ff2-on-cwt-50e01l64b'
path = '../../Datasets/CVPR2021-02785/pkl_trial_cwt_from_1024'
# path = '../../Datasets/sz_eeg/pkl_cwt_torch'
time_exp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# k_fold = KFold(n_splits=k, shuffle=True)
# filepaths = file_scanf(path=path, contains='i', endswith='.pkl')
# dataset = ListDataset(filepaths)

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')

    # for fold, (train_ids, valid_ids) in enumerate(k_fold.split(dataset)):
    #     train_sampler = SubsetRandomSampler(train_ids)
    #     valid_sampler = SubsetRandomSampler(valid_ids)
    #     train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4,
    #                               prefetch_factor=1)
    #     valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1,
    #                               prefetch_factor=1)
    for (fold, train_files, valid_files) in kfold_loader(path, k):
        train_loader = DataLoader(ListDataset(train_files), batch_size=batch_size, num_workers=2, shuffle=True)
        valid_loader = DataLoader(ListDataset(valid_files), batch_size=batch_size, num_workers=1, shuffle=True)
        val_iterable = IterForever(valid_loader)

        # ff = EEGNet(classes_num=40, in_channels=1, electrodes=96, drop_out=0.1).cuda()
        # ff = ConvTransformer(num_classes=40, in_channels=3, hid_channels=8, num_heads=2,
        #                      ffd_channels=16, deep_channels=16, size=32, T=63, depth=1, drop=0.2).cuda()
        ff = FieldFlow2(channels=96, early_drop=0.2, late_drop=0.1).cuda()
        optimizer = torch.optim.Adam(ff.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)  # 设定优优化器更新的时刻表

        print(f'FOLD {fold}')
        train_num = len(train_files)
        print(train_num, len(valid_files), '--------------------------------')
        summary = SummaryWriter(log_dir='./log/' + time_exp + id_exp + '/' + str(fold) + '_fold/')

        global_step = 0
        for epoch in range(1, n_epoch + 1):
            for step, (x, label) in enumerate(train_loader):  # [b, 1, 500, 127], [b]
                if x is None and label is None:
                    continue

                loss, acc = train_accumulate(ff, x, label, optimizer, batch_size=batch_size,
                                             step=step, accumulation=accumulation_steps, cal_acc=True)
                # loss, acc = train(ff, x, label, optimizer, batch_size=batch_size, cal_acc=True)
                lr = optimizer.param_groups[0]['lr']
                summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
                summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=global_step)

                global_step += 1
                if step % 10 == 0:
                    x_val, label_val = val_iterable.next()
                    loss_val, acc_val = test(model=ff, x=x_val, label=label_val, batch_size=batch_size)
                    print('epoch:{}/{} step:{}/{} global_step:{} lr:{:.4f}'
                          ' loss={:.5f} acc={:.5f} val_loss={:.5f} val_acc={:.5f}'.
                          format(epoch, n_epoch, step, int(train_num / batch_size), global_step, lr,
                                 loss, acc, loss_val, acc_val))
                    summary.add_scalar(tag='ValLoss', scalar_value=loss_val, global_step=global_step)
                    summary.add_scalar(tag='ValAcc', scalar_value=acc_val, global_step=global_step)
                # if step % 10 == 0:
                #     cam = ignite_relprop(model=ff, x=x[0].unsqueeze(0), index=label[0])  # [1, 1, 512, 96]
                #     generate_visualization(x[0].squeeze(), cam.squeeze(),
                #                            save_name='S' + str(global_step) + '_C' + str(label[0].cpu().numpy()))
            lr_scheduler.step()  # 更新学习率
        summary.flush()
        summary.close()
    print('done')
