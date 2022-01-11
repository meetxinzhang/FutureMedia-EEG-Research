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

gpu = torch.cuda.is_available()
batch_size = 64
epoch_n = 30

dataset = BDFDataset(CVPR2021_02785_path='D:/high_io_dataset/CVPR2021-02785')
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_, num_workers=10)

model = EEGModel()
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
if gpu:
    model.cuda()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = get_std_optimizer(model, d_model=64, factor=2, warmup=4000)

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

if __name__ == '__main__':
    step = 0
    model.train()
    for epoch in range(epoch_n):
        for x, label in loader:
            if x is None and label is None:
                step += 1
                continue
            if gpu:
                x = x.cuda()
                label = label.cuda()

            optimizer.zero_grad()

            logits = model.forward(x, mask=None)  # [bs, 40]
            loss = F.cross_entropy(logits, label)
            loss.backward()
            lr = optimizer.step()

            step += 1
            if step % 10 == 0:
                corrects = (torch.argmax(logits, dim=1).data == label.data)
                accuracy = corrects.cpu().int().sum().numpy() / batch_size
                print('epoch:{}/{} step:{}/{} loss={:.5f} acc={:.5f} lr={}'.format(epoch, epoch_n, step,
                                                                                   int(400 * 100 / batch_size), loss,
                                                                                   accuracy, lr))
        step = 0
