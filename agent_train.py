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
import torch.distributed as dist
# from agent_lrp import ignite_relprop, get_heatmap_gallery
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
    def __init__(self, n_epoch, model, optimizer, train_loader, val_loader, batch_size, lr_scheduler,
                 id_exp, summary, gpu_rank, device):
        self.global_step = 0
        self.n = n_epoch
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.val_iterable = IterForever(val_loader)
        self.batch_size = torch.tensor(batch_size).to(device)
        self.summary = summary
        self.gpu_rank = gpu_rank
        self.device = device
        self.train_num = len(train_loader)
        self.id_exp = id_exp
        # print(dist.is_initialized(), 'dist')
        # print(dist.get_rank(), 'rank')
        # print(dist.get_world_size(), 'world_size')

    def train_period_parallel(self, epoch, accumulation=1, print_step=10):
        epoch_loss = []
        epoch_acc = []
        idx = []
        ws = dist.get_world_size()
        for step, (x, label) in enumerate(self.train_loader):  # [b, 1, 500, 127],
            assert len(label) == self.batch_size
            if x is None and label is None:
                continue

            if step % print_step != 0:
                _, _ = self.train_step_accumulate(x=x, label=label, step=step, accumulation=accumulation, cal_acc=False)

            else:
                loss, acc = self.train_step_accumulate(x=x, label=label, step=step, accumulation=accumulation, cal_acc=True)
                dist.reduce(loss, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(acc, op=dist.ReduceOp.SUM, dst=0)

                if self.gpu_rank == 0:
                    loss = loss.item() / ws
                    acc = acc.item() / ws
                    lr = self.optimizer.param_groups[0]['lr']
                    print('epoch:{}/{} step:{}/{} lr:{:.4f} loss={:.5f} acc={:.5f}'.
                          format(epoch, self.n, step, self.train_num, lr, loss, acc))
                    epoch_loss.append(loss)
                    epoch_acc.append(acc)
                    idx.append(self.global_step)
                self.global_step += 1
            # step end
        # epoch end
        self.lr_scheduler.step()

        if self.gpu_rank == 0:
            for i in range(len(epoch_loss)):
                self.summary.add_scalar(tag='TrainLoss', scalar_value=epoch_loss[i], global_step=idx[i])
                self.summary.add_scalar(tag='TrainAcc', scalar_value=epoch_acc[i], global_step=idx[i])
            self.summary.flush()

    def validate_epoch_parallel(self, epoch):
        ws = dist.get_world_size()
        epoch_loss_val = torch.tensor(0).float().to(self.device)
        epoch_acc_val = torch.tensor(0).float().to(self.device)
        n = len(self.val_loader)
        for step, (x_val, label_val) in enumerate(self.val_loader):
            loss_val, acc_val = self.validate_step_accumulate(x=x_val, label=label_val)
            epoch_loss_val += loss_val.item()
            epoch_acc_val += acc_val.item()

        epoch_acc_val = epoch_acc_val / n
        epoch_loss_val = epoch_loss_val / n

        dist.barrier()
        dist.reduce(epoch_loss_val, op=dist.ReduceOp.SUM, dst=0)
        dist.reduce(epoch_acc_val, op=dist.ReduceOp.SUM, dst=0)

        if self.gpu_rank == 0:
            epoch_loss_val /= ws
            epoch_acc_val /= ws
            print(' val epoch:{} val_loss={:.5f} val_acc={:.5f}'.format(epoch, epoch_loss_val, epoch_acc_val))
            self.summary.add_scalar(tag='ValLoss', scalar_value=epoch_loss_val, global_step=epoch)
            self.summary.add_scalar(tag='ValAcc', scalar_value=epoch_acc_val, global_step=epoch)
            self.summary.flush()

    def train_period(self, epoch, accumulation=1, print_step=10):
        for step, (x, label) in enumerate(self.train_loader):  # [b, 1, 500, 127], [b]
            if x is None and label is None:
                continue

            if step % print_step != 0:
                _, _ = self.train_step_accumulate(x, label, step=step, accumulation=accumulation, cal_acc=False)
            else:
                loss, acc = self.train_step_accumulate(x, label, step=step, accumulation=accumulation, cal_acc=True)

                lr = self.optimizer.param_groups[0]['lr']
                print('epoch:{}/{} step:{}/{} lr:{:.4f} loss={:.5f} acc={:.5f}'.
                      format(epoch, self.n, step, self.train_num, lr, loss, acc))
                self.summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=self.global_step)
                self.summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=self.global_step)

            # if epoch > 50 and step % 50 == 0:
            #     cam = ignite_relprop(model=self.model, x=x[0].unsqueeze(0), index=label[0], device=self.device)
            #     get_heatmap_gallery(cam.squeeze(0),
            #                         save_name=self.id_exp + '/S' + str(self.global_step) + '_C' + str(
            #                             label[0].cpu().numpy()))
            self.global_step += 1
        self.lr_scheduler.step()

    def val_period(self, epoch):
        epoch_loss_val = torch.tensor(0).float().to(self.device)
        epoch_acc_val = torch.tensor(0).float().to(self.device)
        n = len(self.val_loader)
        for step, (x_val, label_val) in enumerate(self.val_loader):
            loss_val, acc_val = self.validate_step_accumulate(x=x_val, label=label_val)
            epoch_loss_val += loss_val.item()
            epoch_acc_val += acc_val.item()

        epoch_acc_val = epoch_acc_val / n
        epoch_loss_val = epoch_loss_val / n
        print(' val epoch:{} val_loss={:.5f} val_acc={:.5f}'.format(epoch, epoch_loss_val, epoch_acc_val))
        self.summary.add_scalar(tag='ValLoss', scalar_value=epoch_loss_val, global_step=epoch)
        self.summary.add_scalar(tag='ValAcc', scalar_value=epoch_acc_val, global_step=epoch)
        self.summary.flush()

    def train_step_accumulate(self, x, label, step, accumulation, cal_acc=False):
        x = x.to(self.device)
        label = label.to(self.device)

        # forward pass with `autocast` context manager
        with autocast(enabled=True):
            self.model.train()
            y = self.model(x)  # [bs, 40]
            loss = F.cross_entropy(y, label)
            loss = loss / accumulation

        scaler.scale(loss).backward()  # scale gradient and perform backward pass

        if (step + 1) % accumulation == 0:
            scaler.unscale_(self.optimizer)  # before gradient clipping, the optimizer parameters must be unscaled.
            # optim_paras = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            # perform optimization step
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

        accuracy = None
        if cal_acc:
            corrects = (torch.argmax(y, dim=1) == label).float().sum()
            accuracy = torch.div(corrects, self.batch_size)

        return loss, accuracy

    def validate_step_accumulate(self, x, label):
        x = x.to(self.device)
        label = label.to(self.device)
        with autocast(enabled=True):
            self.model.eval()
            y = self.model(x)  # [bs, 40]
            loss = F.cross_entropy(y, label)

        corrects = (torch.argmax(y, dim=1) == label).float().sum()
        accuracy = torch.div(corrects, self.batch_size)

        return loss, accuracy

    # def train_step(self, x, label, step, accumulation, cal_acc=False):
    #     x = x.to(self.device)
    #     label = label.to(self.device)
    #
    #     self.model.train()
    #     self.optimizer.zero_grad()
    #     y = self.model(x)  # [bs, 40]
    #     loss = F.cross_entropy(y, label)
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     accuracy = None
    #     if cal_acc:
    #         corrects = (torch.argmax(y, dim=1) == label).float().sum()
    #         accuracy = torch.div(corrects, self.batch_size)
    #     return loss, accuracy
