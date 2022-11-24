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
from train_test import train, test
from torch.utils.tensorboard import SummaryWriter
import time
from data_pipeline.dataset_szu import ListDataset, collate_
from model.eeg_net import EEGNet
from utils.my_tools import IterForever

# def files_spilt(path):
#     k = 5
#     endswith = '.pkl'
#     train_sets = []
#     test_sets = []
#     for i in range(1, 10):
#         s = file_scanf(path, contains='run_'+str(i), endswith=endswith)
#         l = len(s)
#         assert l % k == 0
#         train = s[int(l/k):]
#         test = s[:int(l/k)]
#         train_sets.append(e for e in train)
#         test_sets.append(e for e in test)
#
#     test1016 = file_scanf(path, contains='run1016', endswith=endswith)
#     l = len(test1016)
#     assert l % k == 0
#     train = test1016[int(l/k):]
#     test = test1016[:int(l/k)]
#     train_sets.append(e for e in train)
#     test_sets.append(e for e in test)
#
#     # k-fold cross-validation
#     return train_sets, test_sets

torch.cuda.set_device(6)
batch_size = 32
n_epoch = 200
k = 6
kfold = KFold(n_splits=k, shuffle=True)

id_exp = '_2000e03l-6fold'
time_exp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# filepaths = file_scanf(path='E:/Datasets/SZFace2/EEG/pkl_ave', contains='_', endswith='.pkl')
filepaths = file_scanf(path='../../Datasets/pkl_ave', contains='_', endswith='.pkl')
dataset = ListDataset(filepaths)
num = len(filepaths)
print(num)
assert num % k == 0
train_num = num * 0.8

if __name__ == '__main__':
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        summary = SummaryWriter(log_dir='./log/' + time_exp + id_exp + '/' + str(fold) + '_fold/')

        train_sampler = SubsetRandomSampler(train_ids)
        valid_sampler = SubsetRandomSampler(valid_ids)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=3,
                                  prefetch_factor=2)
        valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1,
                                  prefetch_factor=1)
        val_iterable = IterForever(valid_loader)

        ff = EEGNet(classes_num=40, drop_out=0.25).cuda()
        optimizer = torch.optim.Adam(ff.parameters(), lr=0.0003, betas=(0.9, 0.98), eps=1e-9)

        global_step = 0
        for epoch in range(1, n_epoch + 1):
            for step, (x, label) in enumerate(train_loader):  # [b, 1, 500, 127], [b]
                if x is None and label is None:
                    continue

                loss, acc = train(ff, x, label, optimizer, batch_size=batch_size, cal_acc=True)
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
        summary.flush()
        summary.close()
    print('done')
