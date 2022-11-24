# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/5/21 9:02 PM
@desc:
"""
import os
import torch
from train_test import train, test
from torch.utils.tensorboard import SummaryWriter
import time
from data_pipeline.dataset_szu import SZUDataset, collate_
from model.eeg_net import EEGNet
from utils.my_tools import IterForever

# from model.field_flow import FieldFlow
# from model.lrp_manager import ignite_relprop, generate_visualization
# from utils.weight_init import get_state_dict

# torch.cuda.set_device(6)
batch_size = 32
n_epoch = 2000

id_experiment = '_2000e03l-82test'
t_experiment = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# ../../Datasets/pkl_ave
# E:/Datasets/CVPR2021-02785/pkl
# E:/Datasets/SZFace2/EEG/pkl_ave
train_set = SZUDataset(path='../../Datasets/pkl_ave', contains='_', endswith='.pkl')
total_train = train_set.__len__()
print(total_train, ' link start !!!!!!')
loader = torch.utils.data.DataLoader(train_set, collate_fn=collate_, batch_size=batch_size, num_workers=4,
                                     prefetch_factor=2, shuffle=True, drop_last=True)

val_set = SZUDataset(path='../../Datasets/pkl_ave_val', contains='_', endswith='.pkl')
total_val = val_set.__len__()
print(total_val, ' num of validation')
val_loader = torch.utils.data.DataLoader(val_set, collate_fn=collate_, batch_size=batch_size, num_workers=1,
                                         prefetch_factor=1, shuffle=True, drop_last=True)
val_iterable = IterForever(val_loader)


# ff = FieldFlow(dim=96, num_heads=6, mlp_dilator=2, qkv_bias=False, drop_rate=0.2, attn_drop_rate=0.2,
#                t=500, n_signals=127, n_classes=40).cuda()
# ff.load_state_dict(get_state_dict('log/checkpoint/2022-11-04-15-59-42_1000e03l-pre.pkl',
#                                   map_location='cuda:0', exclude=['arc_margin.weight']))
ff = EEGNet(classes_num=40, drop_out=0.25).cuda()

optimizer = torch.optim.AdamW(ff.parameters(), lr=0.0003, betas=(0.9, 0.98), eps=1e-9)
# optimizer = NoamOpt(model_size=40, factor=1, warmup=8000,
#                     optimizer=torch.optim.Adam(ff.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


summary = SummaryWriter(log_dir='./log/' + t_experiment + id_experiment + '/')
if __name__ == '__main__':
    # step = 0
    global_step = 0
    for epoch in range(1, n_epoch + 1):
        for step, (x, label) in enumerate(loader):
            #  [b, 1, 500, 127], [b]
            if x is None and label is None:
                # step += 1
                # global_step += 1
                continue

            loss, acc = train(ff, x, label, optimizer, acc=True)
            acc = acc/batch_size
            # lr = optimizer.param_groups[0]['lr']
            summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
            summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=global_step)

            # step += 1
            global_step += 1
            if step % 10 == 0:
                x_test, label_test = val_iterable.next()
                loss_test, acc_b_test = test(model=ff, x=x_test, label=label_test)
                acc_test = acc_b_test / batch_size
                print('epoch:{}/{} step:{}/{} global_step:{} loss={:.5f} acc={:.3f} test_acc={}'.format(
                    epoch, n_epoch, step, int(total_train/batch_size), global_step, loss, acc, acc_test))
                summary.add_scalar(tag='TestLoss', scalar_value=loss_test, global_step=global_step)
                summary.add_scalar(tag='TestAcc', scalar_value=acc_test, global_step=global_step)
            # if step % 10 == 0:
            #     cam = ignite_relprop(model=ff, x=x[0].unsqueeze(0), index=label[0])  # [1, 1, 512, 96]
            #     generate_visualization(x[0].squeeze(), cam.squeeze(),
            #                            save_name='S' + str(global_step) + '_C' + str(label[0].cpu().numpy()))

        # step = 0
    # torch.save(ff.state_dict(), 'log/checkpoint/' + t_experiment + id_experiment + '.pkl')
    summary.flush()
    summary.close()
    print('done')
