# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 12/5/21 9:02 PM
@desc:
"""
from data_pipeline.dataset import BDFDataset, collate_
from model.field_flow import FieldFlow
import torch
import torch.nn.functional as F
from utils.learning_rate import get_std_optimizer
from torch.utils.tensorboard import SummaryWriter

summary = SummaryWriter(log_dir='./log/')

gpu = torch.cuda.is_available()
batch_size = 3

# dataset = BDFDataset(CVPR2021_02785_path='E:/Datasets/CVPR2021-02785')
# loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_, num_workers=1)

# model = EEGModel()
ff = FieldFlow(num_heads=6, mlp_dilator=2, qkv_bias=False, drop_rate=0.1, attn_drop_rate=0.1,
               batch_size=16, time=512, channels=96, n_classes=40)
# for p in model.parameters():
#     if p.dim() > 1:
#         torch.nn.init.xavier_uniform_(p)
if gpu:
    ff.cuda()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = get_std_optimizer(ff, model_size=96)

# ----- Testing code start ----- Use following to test code without load data -----
fake_x_for_testing = torch.rand(3, 512, 96).unsqueeze(1).transpose(2, 3).cuda()      # [batch_size, time_step, channels]
fake_label_for_testing = torch.tensor([1, 0, 1], dtype=torch.long).cuda()
ff.train()
optimizer.zero_grad()
logits = ff(fake_x_for_testing)  # [bs, 40]
loss = F.cross_entropy(logits, fake_label_for_testing)
loss.backward()
optimizer.step()
print(loss.data)
# ----- Testing code end-----------------------------------------------------------

# if __name__ == '__main__':
#     step = 0
#     global_step = 0
#     for epoch in range(11):
#         for x, label in loader:
#             #  [b, 96, 512], [b]
#             if x is None and label is None:
#                 step += 1
#                 global_step += 1
#                 continue
#             if gpu:
#                 x = x.cuda()
#                 label = label.cuda()
#
#             ff.train()
#             optimizer.zero_grad()
#
#             print('aaaaaaaaaa', x.shape)
#             logits = ff(x)  # [bs, 40]
#             loss = F.cross_entropy(logits, label)
#             loss.backward()
#             lr = optimizer.step()
#
#             step += 1
#             global_step += 1
#             if step % 5 == 0:
#                 corrects = (torch.argmax(logits, dim=1).data == label.data)
#                 accuracy = corrects.cpu().int().sum().numpy() / batch_size
#                 print('epoch:{}/10 step:{}/{} global_step:{} '
#                       'loss={:.5f} acc={:.3f} lr={}'.format(epoch, step, int(400 * 100 / batch_size), global_step,
#                                                             loss, accuracy, lr))
#                 summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
#                 summary.add_scalar(tag='TrainAcc', scalar_value=accuracy, global_step=global_step)
#         step = 0
#     summary.close()
