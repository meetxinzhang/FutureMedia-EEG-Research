# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/5/21 9:02 PM
@desc:
"""
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.learning_rate import get_std_optimizer
from data_pipeline.dataset import BDFDataset, collate_
from model.field_flow import FieldFlow
from model.lrp_ignition import ignite_relprop
from utils.lrp_visualiztion import generate_visualization

summary = SummaryWriter(log_dir='./log/')
gpu = torch.cuda.is_available()
batch_size = 16

dataset = BDFDataset(CVPR2021_02785_path='E:/Datasets/CVPR2021-02785')
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_, num_workers=6)

ff = FieldFlow(num_heads=5, mlp_dilator=2, qkv_bias=False, drop_rate=0.2, attn_drop_rate=0.2,
               n_signals=96, n_classes=40).cuda()

# if gpu:
#     ff.cuda()

optimizer = get_std_optimizer(ff, model_size=40)

# ----- Testing code start ----- Use following to test code without load data -----
# x = torch.ones(1, 512, 96).unsqueeze(1).cuda()  # [batch_size, 1, time_step, channels]
# fake_label_for_testing = torch.tensor([1, 0, 1], dtype=torch.long).cuda()
# from data_pipeline.other.mat_reader import read_mat
# x = read_mat()  # [128, t=500]
# x = torch.Tensor(x)
# x = x.transpose(0, 1).unsqueeze(0).unsqueeze(0).cuda()
# # optimizer.zero_grad()
# # logits = ff(fake_x_for_testing)  # [bs, 40]
# # loss = F.cross_entropy(logits, fake_label_for_testing)
# # loss.backward()
# # optimizer.step()
# cam = ignite_relprop(model=ff, x=x, index=torch.tensor(5, device=x.device))  # [1, 1, 500, 128]
# generate_visualization(x.squeeze(), cam.squeeze())
# ----- Testing code end-----------------------------------------------------------

if __name__ == '__main__':
    step = 0
    global_step = 0
    for epoch in range(11):
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
            optimizer.zero_grad()

            logits = ff(x)  # [bs, 40]
            loss = F.cross_entropy(logits, label)
            loss.backward()
            lr = optimizer.step()

            step += 1
            global_step += 1
            if step % 5 == 0:
                corrects = (torch.argmax(logits, dim=1).data == label.data)
                accuracy = corrects.cpu().int().sum().numpy() / batch_size
                print('epoch:{}/10 step:{}/{} global_step:{} '
                      'loss={:.5f} acc={:.3f} lr={}'.format(epoch, step, int(400 * 100 / batch_size), global_step,
                                                            loss, accuracy, lr))
                summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
                summary.add_scalar(tag='TrainAcc', scalar_value=accuracy, global_step=global_step)

            if step % 100 == 0:
                cam = ignite_relprop(model=ff, x=x[0].unsqueeze(0), index=label[0])  # [1, 1, 512, 96]
                generate_visualization(x[0].squeeze(), cam.squeeze(), save_name='C'+str(label[0].cpu().numpy())+'_'+str(global_step))

        step = 0
    summary.close()
