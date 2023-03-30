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
import torch.distributed as dist
from agent_lrp import ignite_relprop, get_heatmap_gallery
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
    def __init__(self, n_epoch, model, optimizer, train_loader, val_iterable, batch_size, id_exp,
                 summary, gpu_rank, device):
        self.global_step = 0
        self.n = n_epoch
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_iterable = val_iterable
        self.batch_size = batch_size
        self.summary = summary
        self.gpu_rank = gpu_rank
        self.device = device
        self.train_num = len(train_loader)
        self.id_exp = id_exp

    def train_period_parallel(self, epoch, accumulation=1, print_step=10):
        method = self.train_step if accumulation == 1 else self.train_accumulate
        for step, (x, label) in enumerate(self.train_loader):  # [b, 1, 500, 127], [b]
            if x is None and label is None:
                continue

            if step % print_step != 0:
                _, _ = method(x=x, label=label, step=step, accumulation=accumulation, cal_acc=False)

            else:
                loss, acc = method(x=x, label=label, step=step, accumulation=accumulation, cal_acc=True)
                x_val, label_val = self.val_iterable.next()
                loss_val, acc_val = self.validate(x=x_val, label=label_val)

                if self.gpu_rank == 0:
                    torch.cuda.synchronize(self.device)  # 等待所有进程计算完毕
                    if not torch.isfinite(loss):
                        print('WARNING: non-finite loss, ending training ', loss)
                        sys.exit(1)

                    lr = self.optimizer.param_groups[0]['lr']
                    print('epoch:{}/{} step:{}/{} lr:{:.4f} loss={:.5f} acc={:.5f} val_loss={:.5f} val_acc={:.5f}'.
                          format(epoch, self.n, step, self.train_num, lr, loss, acc, loss_val, acc_val))
                    self.summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=self.global_step)
                    self.summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=self.global_step)
                    self.summary.add_scalar(tag='ValLoss', scalar_value=loss_val, global_step=self.global_step)
                    self.summary.add_scalar(tag='ValAcc', scalar_value=acc_val, global_step=self.global_step)
                dist.barrier()  # waite the main process
            self.global_step += 1

    def train_period(self, epoch, accumulation=1, print_step=10):
        method = self.train_step if accumulation == 1 else self.train_accumulate
        for step, (x, label) in enumerate(self.train_loader):  # [b, 1, 500, 127], [b]
            if x is None and label is None:
                continue

            if step % print_step != 0:
                _, _ = method(x, label, step=step, accumulation=accumulation, cal_acc=False)
            else:
                loss, acc = method(x, label, step=step, accumulation=accumulation, cal_acc=True)
                x_val, label_val = self.val_iterable.next()
                loss_val, acc_val = self.validate(x=x_val, label=label_val)

                lr = self.optimizer.param_groups[0]['lr']
                print('epoch:{}/{} step:{}/{} lr:{:.4f} loss={:.5f} acc={:.5f} val_loss={:.5f} val_acc={:.5f}'.
                      format(epoch, self.n, step, self.train_num, lr, loss, acc, loss_val, acc_val))
                self.summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=self.global_step)
                self.summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=self.global_step)
                self.summary.add_scalar(tag='ValLoss', scalar_value=loss_val, global_step=self.global_step)
                self.summary.add_scalar(tag='ValAcc', scalar_value=acc_val, global_step=self.global_step)

            if epoch > 25 and step % 50 == 0:
                cam = ignite_relprop(model=self.model, x=x[0].unsqueeze(0), index=label[0], device=self.device)
                get_heatmap_gallery(cam.squeeze(0),
                            save_name=self.id_exp + '/S' + str(self.global_step) + '_C' + str(label[0].cpu().numpy()))

            self.global_step += 1

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
            accuracy = corrects.cpu().int().sum().numpy() / self.batch_size

        return loss, accuracy

    def train_step(self, x, label, step, accumulation, cal_acc=False):
        x = x.to(self.device)
        label = label.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        y = self.model(x)  # [bs, 40]
        loss = F.cross_entropy(y, label)
        loss.backward()
        self.optimizer.step()

        accuracy = None
        if cal_acc:
            corrects = (torch.argmax(y, dim=1).data == label.data)
            accuracy = corrects.cpu().int().sum().numpy() / self.batch_size
        return loss, accuracy

    def validate(self, x, label):
        x = x.to(self.device)
        label = label.to(self.device)

        self.model.eval()
        y = self.model(x)  # [bs, 40]
        loss = F.cross_entropy(y, label)

        corrects = (torch.argmax(y, dim=1).data == label.data)
        accuracy = corrects.cpu().int().sum().numpy() / self.batch_size

        return loss, accuracy
