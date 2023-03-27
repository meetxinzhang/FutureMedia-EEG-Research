# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/5/21 9:02 PM
@desc:
"""
import torch
from torch.utils.data import DataLoader
import time
from torch.utils.tensorboard import SummaryWriter
from utils.my_tools import file_scanf
from data_pipeline.dataset_szu import ListDataset
from model.eeg_net import EEGNet
from train_test import train, test
from utils.my_tools import IterForever

# from model.lrp_manager import ignite_relprop, generate_visualization
# from utils.weight_init import get_state_dict

torch.cuda.set_device(7)
batch_size = 64
n_epoch = 50
lr = 0.01

id_experiment = 'aep-lrp-ConvTsfm-50e01l64b'
t_experiment = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
train_files = file_scanf(path='../../Datasets/CVPR2021-02785/pkl_delta_ave_512', contains='i', endswith='.pkl')
test_files = file_scanf(path='../../Datasets/CVPR2021-02785/pkl_512', contains='i', endswith='.pkl')

# ff = FieldFlow(dim=96, num_heads=6, mlp_dilator=2, qkv_bias=False, drop_rate=0.2, attn_drop_rate=0.2,
#                t=500, n_signals=127, n_classes=40).cuda()
# ff.load_state_dict(get_state_dict('log/checkpoint/2022-11-04-15-59-42_1000e03l-pre.pkl',
#                                   map_location='cuda:0', exclude=['arc_margin.weight']))
# ff = ComplexEEGNet(classes_num=40, drop_out=0.25).cuda()

if __name__ == '__main__':
    train_set = ListDataset(path_list=train_files)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    val_set = ListDataset(path_list=test_files)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=1, shuffle=True, drop_last=True)
    val_iterable = IterForever(val_loader)

    # ff = FieldFlow2(channels=127).cuda()
    ff = EEGNet(classes_num=40, electrodes=96, drop_out=0.2).cuda()
    optimizer = torch.optim.Adam(ff.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)

    total_train = train_set.__len__()
    print(total_train, ' training')
    print(val_set.__len__(), ' validation')
    summary = SummaryWriter(log_dir='./log/' + t_experiment + id_experiment + '/')

    global_step = 0
    for epoch in range(1, n_epoch + 1):
        for step, (x, label) in enumerate(loader):
            #  [b 127 85 500], [b]
            if x is None and label is None:
                continue

            loss, acc = train(ff, x, label, optimizer, batch_size=batch_size, cal_acc=True)
            lr = optimizer.param_groups[0]['lr']
            summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
            summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=global_step)

            global_step += 1
            if step % 1 == 0:
                x_test, label_test = val_iterable.next()
                loss_test, acc_test = test(model=ff, x=x_test, label=label_test, batch_size=batch_size)
                print('epoch:{}/{} step:{}/{} global_step:{} lr=P{:.4f} '
                      'loss={:.5f} acc={:.3f} test_loss={:.5f} test_acc={:.3f}'.
                      format(epoch, n_epoch, step, int(total_train / batch_size), global_step, lr,
                             loss, acc, loss_test, acc_test))
                summary.add_scalar(tag='TestLoss', scalar_value=loss_test, global_step=global_step)
                summary.add_scalar(tag='TestAcc', scalar_value=acc_test, global_step=global_step)
            # if step % 10 == 0:
            #     cam = ignite_relprop(model=ff, x=x[0].unsqueeze(0), index=label[0])  # [1, 1, 512, 96]
            #     generate_visualization(x[0].squeeze(), cam.squeeze(),
            #                            save_name='S' + str(global_step) + '_C' + str(label[0].cpu().numpy()))

        lr_scheduler.step()
    # torch.save(ff.state_dict(), 'log/checkpoint/' + t_experiment + id_experiment + '.pkl')
    summary.flush()
    summary.close()
    print('done')
