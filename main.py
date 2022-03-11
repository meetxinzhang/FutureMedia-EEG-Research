# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/5/21 9:02 PM
@desc:
"""
from data_pipeline.dataset import BDFDataset, collate_
from model.integrate import EEGModel
import torch
import torch.nn.functional as F
from utils.learning_rate import get_std_optimizer
from torch.utils.tensorboard import SummaryWriter
summary = SummaryWriter(log_dir='./log/')

gpu = torch.cuda.is_available()
batch_size = 32

dataset = BDFDataset(CVPR2021_02785_path='/home/xin/ACS/hight_io/CVPR2021-02785')
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_, num_workers=10)

model = EEGModel()
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
if gpu:
    model.cuda()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = get_std_optimizer(model, d_model=96)

# ----- Testing code start ----- Use following to test code without load data -----
# fake_x_for_testing = torch.rand(3, 128, 96).cuda()      # [batch_size, time_step, channels]
# fake_label_for_testing = torch.tensor([1, 0, 1], dtype=torch.long).cuda()
# model.train()
# optimizer.zero_grad()
# logits = model(fake_x_for_testing, mask=None)  # [bs, 40]
# loss = F.cross_entropy(logits, fake_label_for_testing)
# loss.backward()
# optimizer.step()
# print(loss.data)
# ----- Testing code end-----------------------------------------------------------


step = 0
for epoch in range(11):
    for x, label in loader:
        if x is None and label is None:
            step += 1
            continue
        if gpu:
            x = x.cuda()
            label = label.cuda()

        model.train()
        optimizer.zero_grad()

        logits = model(x, mask=None)  # [bs, 40]
        loss = F.cross_entropy(logits, label)
        loss.backward()
        lr = optimizer.step()

        step += 1
        if step % 5 == 0:
            corrects = (torch.argmax(logits, dim=1).data == label.data)
            accuracy = corrects.cpu().int().sum().numpy() / batch_size
            print('epoch:{}/10 step:{}/{} loss={:.5f} acc={:.3f} lr={}'.format(epoch, step,
                                                                               int(400 * 100 / batch_size), loss,
                                                                               accuracy, lr))
            summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=step)
            summary.add_scalar(tag='TrainAcc', scalar_value=accuracy, global_step=step)
    step = 0
summary.close()
