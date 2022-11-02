# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/5/21 9:02 PM
@desc:
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from data_pipeline.dataset_szu import SZUDataset, collate_
from model.field_flow import FieldFlow
from model.lrp_manager import ignite_relprop, generate_visualization

gpu = torch.cuda.is_available()
# torch.cuda.set_device(6)
batch_size = 32
n_epoch = 1000
total_x = 323  # 400 * 100

id_experiment = '_1000e03l-test-df'
t_experiment = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
os.mkdir('./log/'+t_experiment+id_experiment+'/')
summary = SummaryWriter(log_dir='./log/'+t_experiment+id_experiment+'/')

# ../../Datasets/run00
# ../../Datasets/run16/pkl
# E:/Datasets/CVPR2021-02785/pkl
# E:/Datasets/eegtest/run16/pkl
dataset = SZUDataset(path='../../Datasets/run16/pkl')
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_, num_workers=4, shuffle=True)

ff = FieldFlow(dim=40, num_heads=5, mlp_dilator=2, qkv_bias=False, drop_rate=0.2, attn_drop_rate=0.2,
               t=500, n_signals=127, n_classes=40)


# ff.load_state_dict(torch.load('log/checkpoint/2022-10-28-17-26-04.pkl'))
if gpu:
    ff.cuda()

optimizer = torch.optim.Adam(ff.parameters(), lr=0.0003, betas=(0.9, 0.98), eps=1e-9)
# optimizer = NoamOpt(model_size=40, factor=1, warmup=8000,
#                     optimizer=torch.optim.Adam(ff.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# ----- Testing code start ----- Use following to test code without load data -----
# x = torch.ones(3, 512, 96).unsqueeze(1).cuda()  # [batch_size, 1, time_step, channels]
# y = torch.tensor([1, 0, 1], dtype=torch.long).cuda()
# # from data_pipeline.vit.mat_reader import read_mat
# # x = read_mat()  # [128, t=500]
# # x = torch.Tensor(x).transpose(0, 1).unsqueeze(0).unsqueeze(0).cuda()
# # optimizer.zero_grad()
# # logits = ff(fake_x_for_testing)  # [bs, 40]
# # loss = F.cross_entropy(logits, fake_label_for_testing)
# # loss.backward()
# # optimizer.step()
# cam = ignite_relprop(model=ff, x=x[0].unsqueeze(0), index=y[0])  # [1, 1, 500, 128]
# generate_visualization(x[0].squeeze(), cam.squeeze())
# ----- Testing code end-----------------------------------------------------------

if __name__ == '__main__':
    step = 0
    global_step = 0
    for epoch in range(n_epoch + 1):
        for x, label in loader:
            #  [b, 1, 512, 96], [b]
            if x is None and label is None:
                step += 1
                global_step += 1
                continue
            if gpu:
                x = x.cuda()
                label = label.cuda()

            ff.train()
            # if step % 2 == 0:
            optimizer.zero_grad()  # clean grad per 2 step, to double the batch_size

            y = ff(x)  # [bs, 40]
            loss = F.cross_entropy(y, label)
            loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']

            step += 1
            global_step += 1
            if step % 1 == 0:
                corrects = (torch.argmax(y, dim=1).data == label.data)
                accuracy = corrects.cpu().int().sum().numpy() / batch_size
                print('epoch:{}/{} step:{}/{} global_step:{} '
                      'loss={:.5f} acc={:.3f} lr={}'.format(epoch, n_epoch, step, int(total_x / batch_size), global_step,
                                                            loss, accuracy, lr))
                summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
                summary.add_scalar(tag='TrainAcc', scalar_value=accuracy, global_step=global_step)

            # if step % 10 == 0:
            #     cam = ignite_relprop(model=ff, x=x[0].unsqueeze(0), index=label[0])  # [1, 1, 512, 96]
            #     generate_visualization(x[0].squeeze(), cam.squeeze(),
            #                            save_name='S' + str(global_step) + '_C' + str(label[0].cpu().numpy()))

        step = 0
    # torch.save(ff.state_dict(), 'log/checkpoint/' + t_experiment + id_experiment + '.pkl')
    summary.flush()
    summary.close()
    print('done')
