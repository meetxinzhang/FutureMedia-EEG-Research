# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/24 17:56
 @name: 
 @desc:
"""
import random
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
from model.field_flow_2 import FieldFlow2
from utils.my_tools import IterForever
# random.seed = 2022
# torch.manual_seed(2022)
# torch.cuda.manual_seed(2022)


def kfold_loader(path, k):
    database = []
    for i in range(0, 9):
        files_list = file_scanf(path, contains='run_'+str(i)+'_', endswith='.pkl')
        # random.shuffle(files_list)  # shuffle the set by random
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
batch_size = 16
accumulation_steps = 4  # to accumulate gradient when you want to set larger batch_size but out of memory.
n_epoch = 50
k = 7
lr = 0.01

id_exp = 'cwt_ff2_200e01l64b'
# path = '../../Datasets/pkl_ave'
path = '../../Datasets/sz_eeg/pkl_cwt_torch'
time_exp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# k_fold = KFold(n_splits=k, shuffle=True)
# filepaths = file_scanf(path=path, contains='i', endswith='.pkl')
# dataset = ListDataset(filepaths)

if __name__ == '__main__':
    # for fold, (train_ids, valid_ids) in enumerate(k_fold.split(dataset)):
    #     train_sampler = SubsetRandomSampler(train_ids)
    #     valid_sampler = SubsetRandomSampler(valid_ids)
    #     train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=3,
    #                               prefetch_factor=2)
    #     valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1,
    #                               prefetch_factor=1)
    for (fold, train_files, test_files) in kfold_loader(path, k):
        train_loader = DataLoader(ListDataset(train_files), batch_size=batch_size, num_workers=2, shuffle=True)
        valid_loader = DataLoader(ListDataset(test_files), batch_size=batch_size, num_workers=1, shuffle=True)
        val_iterable = IterForever(valid_loader)
        train_num = len(train_files)

        # ff = ComplexEEGNet(classes_num=40, channels=127, drop_out=0.2).cuda()
        # ff = EEGNet(classes_num=40, channels=127, drop_out=0.2).cuda()
        # ff = ConvTransformer(num_classes=40, channels=8, num_heads=2, E=16, F=32,  # aep
        #                      size=32, T=1024, depth=1, drop=0.1).cuda()
        # ff = ConvTransformer(num_classes=40, channels=12, num_heads=3, E=16, F=32,  # aep
        #                      size=32, T=500, depth=2, drop=0.3).cuda()
        ff = FieldFlow2(channels=127, early_drop=0.3, late_drop=0.1).cuda()
        optimizer = torch.optim.SGD(ff.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)  # 设定优优化器更新的时刻表

        print(f'FOLD {fold}')
        print(train_num, len(test_files), '--------------------------------')
        summary = SummaryWriter(log_dir='./log/' + time_exp + id_exp + '/' + str(fold) + '_fold/')

        global_step = 0
        for epoch in range(1, n_epoch + 1):
            for step, (x, label) in enumerate(train_loader):  # [b, 1, 500, 127], [b]
                if x is None and label is None:
                    continue

                loss, acc = train_accumulate(ff, x, label, optimizer, batch_size=batch_size,
                                             step=step, accumulation=accumulation_steps, cal_acc=True)
                summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
                summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=global_step)

                # step += 1
                global_step += 1
                if step % 10 == 0:
                    x_val, label_val = val_iterable.next()
                    loss_val, acc_val = test(model=ff, x=x_val, label=label_val, batch_size=batch_size)
                    print('epoch:{}/{} step:{}/{} global_step:{} '
                          'loss={:.5f} acc={:.3f} val_loss={:.5f} val_acc={:.3f}'.format(epoch, n_epoch, step,
                                                                                         int(train_num / batch_size),
                                                                                         global_step, loss, acc,
                                                                                         loss_val, acc_val))
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
