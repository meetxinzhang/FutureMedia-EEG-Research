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

gpu = torch.cuda.is_available()

batch_size = 16
learning_rate = 0.001

dataset = BDFDataset(CVPR2021_02785_path='/media/xin/Raid0/dataset/CVPR2021-02785')
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_, num_workers=10)
model = EEGModel()
if gpu:
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

step = 0
for epoch in range(10):
    for x, label in loader:
        if x is None and label is None:
            step += 1
            continue
        if gpu:
            x = x.cuda()
            label = label.cuda()

        model.train()
        optimizer.zero_grad()

        logits = model(x)  # [bs, 40]
        loss = F.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()

        step += 1
        if step % 5 == 0:
            corrects = (torch.argmax(logits, dim=1).data == label.data)
            accuracy = corrects.cpu().int().sum().numpy() / batch_size
            print('epoch:{}/10 step:{}/{} loss:{:.5f} acc:{:.3f}'.format(epoch, step,
                                                                         int(400*100 / batch_size), loss, accuracy))
    step = 0
