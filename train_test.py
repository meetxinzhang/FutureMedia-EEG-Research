# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/26 19:04
 @name: 
 @desc:
"""
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


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


class XinTrainer:
    def __init__(self, n_epoch, model, optimizer, batch_size, main_gpu, device):
        self.global_step = 0
        self.n = n_epoch
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.main_gpu = main_gpu
        self.device = device

    def train_period_parallel(self, epoch, accumulation, gpu_rank, summary, train_loader, val_iterable):
        for step, (x, label) in enumerate(train_loader):  # [b, 1, 500, 127], [b]
            if x is None and label is None:
                continue

            if step % 10 != 0:
                _, _ = self.train_accumulate(x=x, label=label, step=step, accumulation=accumulation, cal_acc=False)

            else:
                loss, acc = self.train_accumulate(x=x, label=label, step=step, accumulation=accumulation, cal_acc=True)
                x_val, label_val = val_iterable.next()
                loss_val, acc_val = self.validate(x=x_val, label=label_val)

                if gpu_rank == self.main_gpu:
                    if self.device != torch.device("cpu"):
                        torch.cuda.synchronize(self.device)  # 等待所有进程计算完毕

                    lr = self.optimizer.param_groups[0]['lr']
                    print('epoch:{}/{} step:{}/{} lr:{:.4f} loss={:.5f} acc={:.5f} val_loss={:.5f} val_acc={:.5f}'.
                          format(epoch, self.n, step, len(train_loader), lr, loss, acc, loss_val, acc_val))
                    summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=self.global_step)
                    summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=self.global_step)
                    summary.add_scalar(tag='ValLoss', scalar_value=loss_val, global_step=self.global_step)
                    summary.add_scalar(tag='ValAcc', scalar_value=acc_val, global_step=self.global_step)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)
            self.global_step += 1

            # if step % 10 == 0:
            #     cam = ignite_relprop(model=ff, x=x[0].unsqueeze(0), index=label[0])  # [1, 1, 512, 96]
            #     generate_visualization(x[0].squeeze(), cam.squeeze(),
            #                            save_name='S' + str(global_step) + '_C' + str(label[0].cpu().numpy()))

    def train_accumulate(self, x, label, step, accumulation, cal_acc=False):
        x = x.to(self.device)
        label = label.to(self.device)

        # forward pass with `autocast` context manager
        with autocast(enabled=True):
            self.model.train()
            y = self.model(x)  # [bs, 40]
            loss = F.cross_entropy(y, label) / accumulation

        scaler.scale(loss).backward()  # scale gradient and perform backward pass
        # scaler.unscale_(optimizer)  # before gradient clipping the optimizer parameters must be unscaled.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # perform optimization step

        if (step + 1) % accumulation == 0:
            scaler.step(self.optimizer)
            scaler.update()

        accuracy = None
        if cal_acc:
            corrects = (torch.argmax(y, dim=1).data == label.data)
            accuracy = corrects.cpu().int().sum().numpy()

        return loss, accuracy / self.batch_size

    def validate(self, x, label):
        x = x.to(self.device)
        label = label.to(self.device)

        self.model.eval()
        y = self.model(x)  # [bs, 40]
        loss = F.cross_entropy(y, label)

        corrects = (torch.argmax(y, dim=1).data == label.data)
        accuracy = corrects.cpu().int().sum().numpy()

        return loss, accuracy / self.batch_size


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

    return loss, accuracy / batch_size


def train_accumulate(model, x, label, optimizer, batch_size, step, accumulation, cal_acc=False):
    x = x.cuda()
    label = label.cuda()

    # forward pass with `autocast` context manager
    with autocast(enabled=True):
        model.train()
        y = model(x)  # [bs, 40]
        loss = F.cross_entropy(y, label) / accumulation

    scaler.scale(loss).backward()  # scale gradient and perform backward pass
    # scaler.unscale_(optimizer)  # before gradient clipping the optimizer parameters must be unscaled.
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # perform optimization step

    if (step + 1) % accumulation == 0:
        scaler.step(optimizer)
        scaler.update()

    accuracy = None
    if cal_acc:
        corrects = (torch.argmax(y, dim=1).data == label.data)
        accuracy = corrects.cpu().int().sum().numpy()

    return loss, accuracy / batch_size


def test(model, x, label, batch_size):
    x = x.cuda()
    label = label.cuda()

    model.eval()
    y = model(x)  # [bs, 40]
    loss = F.cross_entropy(y, label)

    corrects = (torch.argmax(y, dim=1).data == label.data)
    accuracy = corrects.cpu().int().sum().numpy()

    return loss, accuracy / batch_size
