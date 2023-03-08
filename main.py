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
from utils.my_tools import file_scanf
from data_pipeline.dataset_szu import SZUDataset
from model.field_flow_2 import FieldFlow2
from train_test import train, test
from utils.my_tools import IterForever

# from model.lrp_manager import ignite_relprop, generate_visualization
# from utils.weight_init import get_state_dict

torch.cuda.set_device(6)
batch_size = 32
n_epoch = 200
lr = 0.003

id_experiment = 'ff2_200e003l32b'
t_experiment = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# ff = FieldFlow(dim=96, num_heads=6, mlp_dilator=2, qkv_bias=False, drop_rate=0.2, attn_drop_rate=0.2,
#                t=500, n_signals=127, n_classes=40).cuda()
# ff.load_state_dict(get_state_dict('log/checkpoint/2022-11-04-15-59-42_1000e03l-pre.pkl',
#                                   map_location='cuda:0', exclude=['arc_margin.weight']))
# ff = ComplexEEGNet(classes_num=40, drop_out=0.25).cuda()

# summary = SummaryWriter(log_dir='./log/' + t_experiment + id_experiment + '/')


if __name__ == '__main__':
    train_set = SZUDataset(path='/data0/zhangxin/Datasets/sz_eeg/pkl_cwt_torch', condition='run_', endswith='.pkl')
    total_train = train_set.__len__()
    print(total_train, ' training')
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    val_set = SZUDataset(path='/data0/zhangxin/Datasets/sz_eeg/pkl_cwt_torch', condition='test', endswith='.pkl')
    total_val = val_set.__len__()
    print(total_val, ' validation')
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=1, shuffle=True, drop_last=True)
    val_iterable = IterForever(val_loader)

    ff = FieldFlow2(channels=127).cuda()
    optimizer = torch.optim.AdamW(ff.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)

    step = 0
    global_step = 0
    for epoch in range(1, n_epoch + 1):
        for step, (x, label) in enumerate(loader):
            #  [b 127 85 500], [b]
            if x is None and label is None:
                step += 1
                global_step += 1
                continue
            x = x.cuda()
            # ff.train()
            # y = ff(x)  # [bs, 40]

            loss, acc = train(ff, x, label, optimizer, batch_size=batch_size, cal_acc=True)
            lr = optimizer.param_groups[0]['lr']
            # summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
            # summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=global_step)

            step += 1
            global_step += 1
            if step % 1 == 0:
                x_test, label_test = val_iterable.next()
                loss_test, acc_test = test(model=ff, x=x_test, label=label_test, batch_size=batch_size)
                print('epoch:{}/{} step:{}/{} global_step:{} lr=P{:.4f} '
                      'loss={:.5f} acc={:.3f} test_loss={:.5f} test_acc={:.3f}'.
                      format(epoch, n_epoch, step, int(total_train / batch_size), global_step, lr,
                             loss, acc, loss_test, acc_test))
                # summary.add_scalar(tag='TestLoss', scalar_value=loss_test, global_step=global_step)
                # summary.add_scalar(tag='TestAcc', scalar_value=acc_test, global_step=global_step)
            # if step % 10 == 0:
            #     cam = ignite_relprop(model=ff, x=x[0].unsqueeze(0), index=label[0])  # [1, 1, 512, 96]
            #     generate_visualization(x[0].squeeze(), cam.squeeze(),
            #                            save_name='S' + str(global_step) + '_C' + str(label[0].cpu().numpy()))

        step = 0
        lr_scheduler.step()
    # torch.save(ff.state_dict(), 'log/checkpoint/' + t_experiment + id_experiment + '.pkl')
    # summary.flush()
    # summary.close()
    print('done')
