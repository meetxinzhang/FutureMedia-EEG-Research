# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/26 19:04
 @name: 
 @desc:
"""
import torch
import torch.nn.functional as F


# ----- Testing code start ----- Use following to test code without load data -----
# _x = torch.ones(3, 500, 127).unsqueeze(1).cuda()  # [batch_size, 1, time_step, channels]
# _y = torch.tensor([1, 0, 1], dtype=torch.long).cuda()
# optimizer.zero_grad()
# _logits = ff(_x)  # [bs, 40]
# _loss = F.cross_entropy(_logits, _y)
# _loss.backward()
# optimizer.step()
# del _x, _y, _logits, _loss
# _cam = ignite_relprop(model=ff, _x=_x[0].unsqueeze(0), index=_y[0])  # [1, 1, 500, 128]
# generate_visualization(_x[0].squeeze(), _cam.squeeze())
# ----- Testing code end-----------------------------------------------------------

def train(model, x, label, optimizer, batch_size, cal_acc=False):
    x = x.cuda()
    label = label.cuda()

    model.train()
    # # if step % 2 == 0:
    optimizer.zero_grad()
    y = model(x)  # [bs, 40]
    loss = F.cross_entropy(y, label)
    loss.backward()
    optimizer.step()

    accuracy = None
    if cal_acc:
        corrects = (torch.argmax(y, dim=1).data == label.data)
        accuracy = corrects.cpu().int().sum().numpy()

    return loss, accuracy/batch_size


def train_accumulate(model, x, label, optimizer, batch_size, step, accumulation, cal_acc=False):
    x = x.cuda()
    label = label.cuda()

    model.train()
    y = model(x)  # [bs, 40]
    loss = F.cross_entropy(y, label) / accumulation
    loss.backward()

    if (step + 1) % accumulation == 0:
        optimizer.step()
        optimizer.zero_grad()

    accuracy = None
    if cal_acc:
        corrects = (torch.argmax(y, dim=1).data == label.data)
        accuracy = corrects.cpu().int().sum().numpy()

    return loss, accuracy/batch_size


def test(model, x, label, batch_size):
    x = x.cuda()
    label = label.cuda()

    model.eval()
    y = model(x)  # [bs, 40]
    loss = F.cross_entropy(y, label)

    corrects = (torch.argmax(y, dim=1).data == label.data)
    accuracy = corrects.cpu().int().sum().numpy()

    return loss, accuracy/batch_size

