# # encoding: utf-8
# """
# @author: Xin Zhang
# @contact: zhangxin@szbl.ac.cn
# @time: 12/4/21 2:34 PM
# @desc:
# """
#
# import math
# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim
# import numpy as np
# from scipy.stats import mode
# # from svmutil import *
#
# minus_infinity = float("-Inf")
#
# ##############################################################
# # K-NN classifier
# ##############################################################
# # def kNN(trainSet, trainLabel, sample, k):
# #     trainSetNum = len(trainSet)
# #     distance = np.zeros((1, trainSetNum))
# #     label = np.zeros((1, trainSetNum))
# #     for i in range(trainSetNum):
# #         t, l = trainSet[i], trainLabel[i]
# #         dist = math.sqrt(np.sum(np.power((t-sample), 2)))
# #         distance[0, i] = dist
# #         label[0, i] = l
# #     sortedIdx = np.argsort(distance)
# #     labelSorted = label[0, sortedIdx]
# #     labelMode = mode(labelSorted[0, 0:k])
# #     return labelMode[0]
# #
# # def classifier_KNN(dataset, trainIdx, testIdx, k, channel_idx):
# #     correct = 0.0
# #     if channel_idx is None:
# #         trainSet = [dataset[idx][0].contiguous().view(1, -1).numpy()
# #                     for idx in trainIdx]
# #     else:
# #         trainSet = [dataset[idx][0][:, channel_idx].contiguous().view(1, -1).
# #                     numpy()
# #                     for idx in trainIdx]
# #     trainLabel = [dataset[idx][1] for idx in trainIdx]
# #     for i in range(len(testIdx)):
# #         sample, sample_label = dataset[testIdx[i]]
# # 	sample = sample[:, channel_idx].contiguous().view(1, -1).numpy()
# #         pre_label = kNN(trainSet, trainLabel, sample, k)
# #         if pre_label==sample_label:
# #             correct += 1
# #     return correct/len(testIdx), len(testIdx)
#
# ##############################################################
# # linear SVM classifier
# ##############################################################
# # def SVM_data(dataset, Idx, channel_idx):
# #     data = []
# #     label = []
# #     if channel_idx is None:
# #         Set = [dataset[idx][0].contiguous().view(1, -1).numpy() for idx in Idx]
# #         Label = [dataset[idx][1] for idx in Idx]
# #     else:
# #         Set = [dataset[idx][0][:, channel_idx].contiguous().view(1, -1).numpy()
# #                for idx in Idx]
# #         Label = [dataset[idx][1] for idx in Idx]
# #     for i in range(len(Idx)):
# #         data.append(Set[i].tolist()[0])
# #         label.append(Label[i])
# #     return data, label
#
# # def classifier_SVM(dataset,
# #                    trainIdx,
# #                    valIdx,
# #                    testIdx,
# #                    channel_idx,
# #                    nonclasses,
# #                    pretrain,
# #                    save):
# #     TrainSet, TrainLabel = SVM_data(dataset, trainIdx, channel_idx)
# #     ValSet, ValLabel = SVM_data(dataset, valIdx, channel_idx)
# #     TestSet, TestLabel = SVM_data(dataset, testIdx, channel_idx)
# #     prob = svm_problem(TrainLabel, TrainSet)
# #     if pretrain is None:
# #         if save is None and len(nonclasses)==0:
# #             model = svm_train(prob, "-t 0 -c 10 -m 10000 -e 0.1 -b 0 -q")
# #         else:
# #             model = svm_train(prob, "-t 0 -c 10 -m 10000 -e 0.1 -b 1 -q")
# #     else:
# #         model = svm_load_model(pretrain+".model")
# #     if save is not None:
# #         svm_save_model(save+".model", model)
# #     if pretrain is None and save is None and len(nonclasses)==0:
# #         _, p_acc_v, _ = svm_predict(ValLabel, ValSet, model, "-b 0 -q")
# #         _, p_acc_t, _ = svm_predict(TestLabel, TestSet, model, "-b 0 -q")
# #         return p_acc_v[0]/100.0, p_acc_t[0]/100.0, len(ValSet), len(TestSet)
# #     else:
# #         labels = model.get_labels()
# #         _, _, p_vals_v = svm_predict(ValLabel, ValSet, model, "-b 1 -q")
# #         _, _, p_vals_t = svm_predict(TestLabel, TestSet, model, "-b 1 -q")
# #         p_vals_v = torch.tensor(p_vals_v)
# #         p_vals_t = torch.tensor(p_vals_t)
# #         for i in nonclasses:
# #             if i in labels:
# #                 p_vals_v[:, labels.index(i)] = minus_infinity
# #                 p_vals_t[:, labels.index(i)] = minus_infinity
# #         _, pred_v = p_vals_v.max(1)
# #         _, pred_t = p_vals_t.max(1)
# #         p_acc_v = (pred_v.
# #                    eq(torch.tensor([labels.index(l) for l in ValLabel])).
# #                    sum().
# #                    float()/
# #                    len(ValSet)).item()
# #         p_acc_t = (pred_t.
# #                    eq(torch.tensor([labels.index(l) for l in TestLabel])).
# #                    sum().
# #                    float()/
# #                    len(TestSet)).item()
# #         return p_acc_v, p_acc_t, len(ValSet), len(TestSet)
#
# ##############################################################
# # LSTM classifier
# ##############################################################
# class classifier_LSTM(nn.Module):
#     def __init__(self,
#                  relup,
#                  input_size,
#                  lstm_layers,
#                  lstm_size,
#                  output1_size,
#                  output2_size,
#                  GPUindex):
#         super(classifier_LSTM, self).__init__()
#         self.relup = relup
#         self.lstm_layers = lstm_layers
#         self.lstm_size = lstm_size
#         self.GPUindex = GPUindex
#         self.lstm = nn.LSTM(
#             input_size, lstm_size, num_layers = 1, batch_first = True)
#         self.output1 = nn.Linear(lstm_size, output1_size)
#         self.relu = nn.ReLU()
#         if output2_size is None:
#             self.output2 = None
#         else:
#             self.output2 = nn.Linear(lstm_size, output2_size)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size),
#                      torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
#         if x.is_cuda:
#             lstm_init = (lstm_init[0].cuda(self.GPUindex),
#                          lstm_init[0].cuda(self.GPUindex))
#         lstm_init = (Variable(lstm_init[0]), Variable(lstm_init[1]))
#         x = self.lstm(x, lstm_init)[0][:, -1, :]
#         x = self.output1(x)
#         if self.relup:
#             x = self.relu(x)
#         if self.output2 is not None:
#             x = self.output2(x)
#         return x
#
# # ##############################################################
# # # MLP classifier (2FC)
# # ##############################################################
# # class classifier_MLP(nn.Module):
# #
# #     def __init__(self, input_size, output_size):
# #         super(classifier_MLP, self).__init__()
# #         self.input_size = input_size
# #         self.act = nn.Sigmoid()
# #         self.output1 = nn.Linear(input_size, 128)
# #         self.output2 = nn.Linear(128, output_size)
# #
# #     def forward(self, x):
# #         batch_size = x.size(0)
# #         x = x.view(batch_size, -1)
# #         x = self.output1(x)
# #         x = self.act(x)
# #         x = self.output2(x)
# #         return x
#
# ##############################################################
# # CNN classifier
# ##############################################################
# class classifier_CNN(nn.Module):
#
#     def __init__(self, in_channel, num_points, output_size):
#         super(classifier_CNN, self).__init__()
#         self.channel = in_channel
#         conv1_size = 32
#         conv1_stride = 1
#         self.conv1_out_channels = 8
#         self.conv1_out = int(
#             math.floor(((num_points-conv1_size)/conv1_stride+1)))
#         fc1_in = self.channel*self.conv1_out_channels
#         fc1_out = 40
#         pool1_size = 128
#         pool1_stride = 64
#         pool1_out = int(
#             math.floor(((self.conv1_out-pool1_size)/pool1_stride+1)))
#         dropout_p = 0.5
#         fc2_in = pool1_out*fc1_out
#         self.conv1 = nn.Conv1d(in_channels = 1,
#                                out_channels = self.conv1_out_channels,
#                                kernel_size = conv1_size,
#                                stride = conv1_stride)
#         self.fc1 = nn.Linear(fc1_in, fc1_out)
#         self.pool1 = nn.AvgPool1d(kernel_size = pool1_size,
#                                   stride = pool1_stride)
#         self.activation = nn.ELU()
#         self.dropout = nn.Dropout(p = dropout_p)
#         self.fc2 = nn.Linear(fc2_in, output_size)
#
#     def forward(self, x):
#         batch_size = x.data.shape[0]
#         x = x.permute(0, 2, 1)
#         x = torch.unsqueeze(x, 2)
#         x = x.contiguous().view(-1, 1, x.data.shape[-1])
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = x.view(batch_size,
#                    self.channel,
#                    self.conv1_out_channels,
#                    self.conv1_out)
#         x = x.permute(0, 3, 1, 2)
#         x = x.contiguous().view(batch_size,
#                                 self.conv1_out,
#                                 self.channel*self.conv1_out_channels)
#         x = self.dropout(x)
#         x = self.fc1(x)
#         x = self.dropout(x)
#         x = x.permute(0, 2, 1)
#         x = self.pool1(x)
#         x = x.contiguous().view(batch_size, -1)
#         x = self.fc2(x)
#         return x
#
# ##############################################################
# # Spatial-CNN classifier
# ##############################################################
# # class classifier_SCNN(nn.Module):
# #
# #     def __init__(self, in_channel, num_points, output_size):
# #         super(classifier_SCNN, self).__init__()
# #         if in_channel!=96:
# #             raise RuntimeError("SCNN must have 96 channels")
# #         time_steps = num_points
# #         convT1 = 32
# #         convH1 = 6
# #         convW1 = 6
# #         conv_out_channels1 = 10
# #         conv1_size = (convT1, convH1, convW1)
# #         conv1_stride = 1
# #         conv1_out = (int(
# #             math.floor(((time_steps-conv1_size[0])/conv1_stride+1))), 1, 1)
# #         pool1_size = (16, 1, 1)
# #         pool1_stride = (8, 1, 1)
# #         pool1_out = (int(
# #             math.floor(((conv1_out[0]-pool1_size[0])/pool1_stride[0]+1))), 1, 1)
# #         convT2 = 8
# #         convH2 = conv_out_channels1*12
# #         conv2_out_channels = 80
# #         conv2_size = (convH2, convT2)
# #         conv2_stride = 1
# #         self.conv2_out = (1, int(
# #             math.floor(((pool1_out[0]-conv2_size[1])/conv2_stride+1))))
# #         pool2_size = (1, 16)
# #         pool2_stride = (1, 8)
# #         pool2_out = (1, int(
# #             math.floor(((self.conv2_out[1]-pool2_size[1])/pool2_stride[1]+1))))
# #         fc1_in = pool2_out[1]*pool2_out[0]*conv2_out_channels
# #         fc1_out = output_size
# #         dropout_p = 0.5
# #         self.conv1 = nn.Conv3d(in_channels = 1,
# #                                out_channels = conv_out_channels1,
# #                                kernel_size = conv1_size,
# #                                stride = conv1_stride)
# #         self.conv2 = nn.Conv3d(in_channels = 1,
# #                                out_channels = conv_out_channels1,
# #                                kernel_size = conv1_size,
# #                                stride = conv1_stride)
# #         self.conv3 = nn.Conv3d(in_channels = 1,
# #                                out_channels = conv_out_channels1,
# #                                kernel_size = conv1_size,
# #                                stride = conv1_stride)
# #         self.conv4 = nn.Conv3d(in_channels = 1,
# #                                out_channels = conv_out_channels1,
# #                                kernel_size = conv1_size,
# #                                stride = conv1_stride)
# #         self.conv5 = nn.Conv3d(in_channels = 1,
# #                                out_channels = conv_out_channels1,
# #                                kernel_size = conv1_size,
# #                                stride = conv1_stride)
# #         self.conv6 = nn.Conv3d(in_channels = 1,
# #                                out_channels = conv_out_channels1,
# #                                kernel_size = conv1_size,
# #                                stride = conv1_stride)
# #         self.conv7 = nn.Conv3d(in_channels = 1,
# #                                out_channels = conv_out_channels1,
# #                                kernel_size = conv1_size,
# #                                stride = conv1_stride)
# #         self.conv8 = nn.Conv3d(in_channels = 1,
# #                                out_channels = conv_out_channels1,
# #                                kernel_size = conv1_size,
# #                                stride = conv1_stride)
# #         self.conv9 = nn.Conv3d(in_channels = 1,
# #                                out_channels = conv_out_channels1,
# #                                kernel_size = conv1_size,
# #                                stride = conv1_stride)
# #         self.conv10 = nn.Conv3d(in_channels = 1,
# #                                 out_channels = conv_out_channels1,
# #                                 kernel_size = conv1_size,
# #                                 stride = conv1_stride)
# #         self.conv11 = nn.Conv3d(in_channels = 1,
# #                                 out_channels = conv_out_channels1,
# #                                 kernel_size = conv1_size,
# #                                 stride = conv1_stride)
# #         self.conv12 = nn.Conv3d(in_channels = 1,
# #                                 out_channels = conv_out_channels1,
# #                                 kernel_size = conv1_size,
# #                                 stride = conv1_stride)
# #         self.pool1 = nn.AvgPool3d(kernel_size = pool1_size,
# #                                   stride = pool1_stride)
# #         self.conv = nn.Conv2d(in_channels = 1,
# #                               out_channels = conv2_out_channels,
# #                               kernel_size = conv2_size,
# #                               stride = conv2_stride)
# #         self.pool = nn.AvgPool2d(kernel_size = pool2_size,
# #                                  stride = pool2_stride)
# #         self.fc1 = nn.Linear(fc1_in, fc1_out)
# #         self.activation = nn.ELU()
# #         self.dropout3d = nn.Dropout3d(p = dropout_p)
# #         self.dropout2d = nn.Dropout2d(p = dropout_p)
# #
# #     def forward(self, data):
# #         batch_size = data.data.shape[0]
# #         data = torch.unsqueeze(data, 1)
# #         data1 = data[:, :, :, 0:6, 0:6]
# #         data2 = data[:, :, :, 3:9, 0:6]
# #         data3 = data[:, :, :, 6:12, 0:6]
# #         data4 = data[:, :, :, 0:6, 3:9]
# #         data5 = data[:, :, :, 3:9, 3:9]
# #         data6 = data[:, :, :, 6:12, 3:9]
# #         data7 = data[:, :, :, 0:6, 6:12]
# #         data8 = data[:, :, :, 3:9, 6:12]
# #         data9 = data[:, :, :, 6:12, 6:12]
# #         data10 = data[:, :, :, 0:6, 9:15]
# #         data11 = data[:, :, :, 3:9, 9:15]
# #         data12 = data[:, :, :, 6:12, 9:15]
# #         last_output1 = self.dropout3d(
# #             self.pool1(self.activation(self.conv1(data1))))
# #         last_output2 = self.dropout3d(
# #             self.pool1(self.activation(self.conv2(data2))))
# #         last_output3 = self.dropout3d(
# #             self.pool1(self.activation(self.conv3(data3))))
# #         last_output4 = self.dropout3d(
# #             self.pool1(self.activation(self.conv4(data4))))
# #         last_output5 = self.dropout3d(
# #             self.pool1(self.activation(self.conv5(data5))))
# #         last_output6 = self.dropout3d(
# #             self.pool1(self.activation(self.conv6(data6))))
# #         last_output7 = self.dropout3d(
# #             self.pool1(self.activation(self.conv7(data7))))
# #         last_output8 = self.dropout3d(
# #             self.pool1(self.activation(self.conv8(data8))))
# #         last_output9 = self.dropout3d(
# #             self.pool1(self.activation(self.conv9(data9))))
# #         last_output10 = self.dropout3d(
# #             self.pool1(self.activation(self.conv10(data10))))
# #         last_output11 = self.dropout3d(
# #             self.pool1(self.activation(self.conv11(data11))))
# #         last_output12 = self.dropout3d(
# #             self.pool1(self.activation(self.conv12(data12))))
# #         last_output = torch.cat((last_output1,
# #                                  last_output2,
# #                                  last_output3,
# #                                  last_output4,
# #                                  last_output5,
# #                                  last_output6,
# #                                  last_output7,
# #                                  last_output8,
# #                                  last_output9,
# #                                  last_output10,
# #                                  last_output11,
# #                                  last_output12),
# #                                 1)
# #         last_output = last_output.view(batch_size,
# #                                        last_output.data.shape[1],
# #                                        -1)
# #         last_output = torch.unsqueeze(last_output, 1)
# #         last_output = last_output.contiguous()
# #         last_output = self.conv(last_output)
# #         last_output = self.activation(last_output)
# #         last_output = self.pool(last_output)
# #         last_output = self.dropout2d(last_output)
# #         last_output = last_output.view(batch_size, -1)
# #         last_output = self.fc1(last_output)
# #         return last_output
#
# ##############################################################
# # EEGNet classifier
# ##############################################################
#
# # class classifier_EEGNet(nn.Module):
# #     def __init__(self, spatial, temporal):
# #         super(classifier_EEGNet, self).__init__()
# #         #possible spatial [128, 96, 64, 32, 16, 8]
# #         #possible temporal [1024, 512, 440, 256, 200, 128, 100, 50]
# #         F1 = 8
# #         F2 = 16
# #         D = 2
# #         first_kernel = temporal//2
# #         first_padding = first_kernel//2
# #         self.network = nn.Sequential(
# #             nn.ZeroPad2d((first_padding, first_padding-1, 0, 0)),
# #             nn.Conv2d(in_channels = 1,
# #                       out_channels = F1,
# #                       kernel_size = (1, first_kernel)),
# #             nn.BatchNorm2d(F1),
# #             nn.Conv2d(in_channels = F1,
# #                       out_channels = F1,
# #                       kernel_size = (spatial, 1),
# #                       groups = F1),
# #             nn.Conv2d(in_channels = F1,
# #                       out_channels = D*F1,
# #                       kernel_size = 1),
# #             nn.BatchNorm2d(D*F1),
# #             nn.ELU(),
# #             nn.AvgPool2d(kernel_size = (1, 4)),
# #             nn.Dropout(),
# #             nn.ZeroPad2d((8, 7, 0, 0)),
# #             nn.Conv2d(in_channels = D*F1,
# #                       out_channels = D*F1,
# #                       kernel_size = (1, 16),
# #                       groups = F1),
# #             nn.Conv2d(in_channels = D*F1,
# #                       out_channels = F2,
# #                       kernel_size = 1),
# #             nn.BatchNorm2d(F2),
# #             nn.ELU(),
# #             nn.AvgPool2d(kernel_size = (1, 8)),
# #             nn.Dropout())
# #         self.fc = nn.Linear(F2*(temporal//32), 40)
# #
# #     def forward(self, x):
# #         x = x.unsqueeze(0).permute(1, 0, 3, 2)
# #         x = self.network(x)
# #         x = x.view(x.size()[0], -1)
# #         return self.fc(x)
# #
# #     def cuda(self, gpuIndex):
# #         self.network = self.network.cuda(gpuIndex)
# #         self.fc = self.fc.cuda(gpuIndex)
# #         return self
#
# ##############################################################
# # SyncNet classifier
# ##############################################################
#
# # class classifier_SyncNet(nn.Module):
# #     def __init__(self, spatial, temporal):
# #         super(classifier_SyncNet, self).__init__()
# #         K = min(10, spatial)
# #         Nt = min(40, temporal)
# #         pool_size = Nt
# #         b = np.random.uniform(low = -0.05, high = 0.05, size = (1, spatial, K))
# #         omega = np.random.uniform(low = 0, high = 1, size = (1, 1, K))
# #         zeros = np.zeros(shape = (1, 1, K))
# #         phi_ini = np.random.normal(
# #             loc = 0, scale = 0.05, size = (1, spatial-1, K))
# #         phi = np.concatenate([zeros, phi_ini], axis = 1)
# #         beta = np.random.uniform(low = 0, high = 0.05, size = (1, 1, K))
# #         t = np.reshape(range(-Nt//2, Nt//2),[Nt, 1, 1])
# #         tc = np.single(t)
# #         W_osc = b*np.cos(tc*omega+phi)
# #         W_decay = np.exp(-np.power(tc, 2)*beta)
# #         W = W_osc*W_decay
# #         W = np.transpose(W, (2, 1, 0))
# #         bias = np.zeros(shape = [K])
# #         self.net = nn.Sequential(nn.ConstantPad1d((Nt//2, Nt//2-1), 0),
# #                                  nn.Conv1d(in_channels = spatial,
# #                                            out_channels = K,
# #                                            kernel_size = 1,
# #                                            stride = 1,
# #                                            bias = True),
# #                                 nn.MaxPool1d(kernel_size = pool_size,
# #                                              stride = pool_size),
# #                                 nn.ReLU())
# #         self.net[1].weight.data = torch.FloatTensor(W)
# #         self.net[1].bias.data = torch.FloatTensor(bias)
# #         self.fc = nn.Linear((temporal//pool_size)*K, 40)
# #
# #     def forward(self, x):
# #         x = x.permute(0, 2, 1)
# #         x = self.net(x)
# #         x = x.view(x.size()[0],-1)
# #         x = self.fc(x)
# #         return x
# #
# #     def cuda(self, gpuIndex):
# #         self.net = self.net.cuda(gpuIndex)
# #         self.fc = self.fc.cuda(gpuIndex)
# #         return self
#
# #############################################################################
# # EEG-ChannelNet classifier
# #############################################################################
#
# # class classifier_EEGChannelNet(nn.Module):
# #
# #     def __init__(self, spatial, temporal):
# #         super(classifier_EEGChannelNet, self).__init__()
# #         self.temporal_layers = []
# #         self.temporal_layers.append(nn.Sequential(nn.Conv2d(in_channels = 1,
# #                                     out_channels = 10,
# #                                     kernel_size = (1, 33),
# #                                     stride = (1, 2),
# #                                     dilation = (1, 1),
# #                                     padding = (0, 16)),
# #                                     nn.BatchNorm2d(10),
# #                                     nn.ReLU()))
# #         self.temporal_layers.append(nn.Sequential(nn.Conv2d(in_channels = 1,
# #                                     out_channels = 10,
# #                                     kernel_size = (1, 33),
# #                                     stride = (1, 2),
# #                                     dilation = (1, 2),
# #                                     padding = (0, 32)),
# #                                     nn.BatchNorm2d(10),
# #                                     nn.ReLU()))
# #         self.temporal_layers.append(nn.Sequential(nn.Conv1d(in_channels = 1,
# #                                     out_channels = 10,
# #                                     kernel_size = (1, 33),
# #                                     stride = (1, 2),
# #                                     dilation = (1, 4),
# #                                     padding = (0, 64)),
# #                                     nn.BatchNorm2d(10),
# #                                     nn.ReLU()))
# #         self.temporal_layers.append(nn.Sequential(nn.Conv1d(in_channels = 1,
# #                                     out_channels = 10,
# #                                     kernel_size = (1, 33),
# #                                     stride = (1, 2),
# #                                     dilation = (1, 8),
# #                                     padding = (0, 128)),
# #                                     nn.BatchNorm2d(10),
# #                                     nn.ReLU()))
# #         self.temporal_layers.append(nn.Sequential(nn.Conv1d(in_channels = 1,
# #                                     out_channels = 10,
# #                                     kernel_size = (1, 33),
# #                                     stride = (1, 2),
# #                                     dilation = (1, 16),
# #                                     padding = (0, 256)),
# #                                     nn.BatchNorm2d(10),
# #                                     nn.ReLU()))
# #         self.spatial_layers = []
# #         self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
# #                                    out_channels = 50,
# #                                    kernel_size = (128, 1),
# #                                    stride = (2, 1),
# #                                    padding = (63, 0)),
# #                                    nn.BatchNorm2d(50),
# #                                    nn.ReLU()))
# #         self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
# #                                    out_channels = 50,
# #                                    kernel_size = (64, 1),
# #                                    stride = (2, 1),
# #                                    padding = (31, 0)),
# #                                    nn.BatchNorm2d(50),
# #                                    nn.ReLU()))
# #         self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
# #                                    out_channels = 50,
# #                                    kernel_size = (32, 1),
# #                                    stride = (2, 1),
# #                                    padding = (15, 0)),
# #                                    nn.BatchNorm2d(50),
# #                                    nn.ReLU()))
# #         self.spatial_layers.append(nn.Sequential(nn.Conv2d(in_channels = 50,
# #                                    out_channels = 50,
# #                                    kernel_size = (16, 1),
# #                                    stride = (2, 1),
# #                                    padding = (7, 0)),
# #                                    nn.BatchNorm2d(50),
# #                                    nn.ReLU()))
# #         self.residual_layers = []
# #         self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
# #                                     out_channels = 200,
# #                                     kernel_size = 3,
# #                                     stride = 2,
# #                                     padding = 1),
# #                                     nn.BatchNorm2d(200),
# #                                     nn.ReLU(),
# #                                     nn.Conv2d(in_channels = 200,
# #                                     out_channels = 200,
# #                                     kernel_size = 3,
# #                                     stride = 1,
# #                                     padding = 1),
# #                                     nn.BatchNorm2d(200)))
# #         self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
# #                                     out_channels = 200,
# #                                     kernel_size = 3,
# #                                     stride = 2,
# #                                     padding = 1),
# #                                     nn.BatchNorm2d(200),
# #                                     nn.ReLU(),
# #                                     nn.Conv2d(in_channels = 200,
# #                                     out_channels = 200,
# #                                     kernel_size = 3,
# #                                     stride = 1,
# #                                     padding = 1),
# #                                     nn.BatchNorm2d(200)))
# #         self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
# #                                     out_channels = 200,
# #                                     kernel_size = 3,
# #                                     stride = 2,
# #                                     padding = 1),
# #                                     nn.BatchNorm2d(200),
# #                                     nn.ReLU(),
# #                                     nn.Conv2d(in_channels = 200,
# #                                     out_channels = 200,
# #                                     kernel_size = 3,
# #                                     stride = 1,
# #                                     padding = 1),
# #                                     nn.BatchNorm2d(200)))
# #         self.residual_layers.append(nn.Sequential(nn.Conv2d(in_channels = 200,
# #                                     out_channels = 200,
# #                                     kernel_size = 3,
# #                                     stride = 2,
# #                                     padding = 1),
# #                                     nn.BatchNorm2d(200),
# #                                     nn.ReLU(),
# #                                     nn.Conv2d(in_channels = 200,
# #                                     out_channels = 200,
# #                                     kernel_size = 3,
# #                                     stride = 1,
# #                                     padding = 1),
# #                                     nn.BatchNorm2d(200)))
# #         self.shortcuts = []
# #         self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
# #                               out_channels = 200,
# #                               kernel_size = 1,
# #                               stride = 2),
# #                               nn.BatchNorm2d(200)))
# #         self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
# #                               out_channels = 200,
# #                               kernel_size = 1,
# #                               stride = 2),
# #                               nn.BatchNorm2d(200)))
# #         self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
# #                               out_channels = 200,
# #                               kernel_size = 1,
# #                               stride = 2),
# #                               nn.BatchNorm2d(200)))
# #         self.shortcuts.append(nn.Sequential(nn.Conv2d(in_channels = 200,
# #                               out_channels = 200,
# #                               kernel_size = 1,
# #                               stride = 2),
# #                               nn.BatchNorm2d(200)))
# #         spatial_kernel = 3
# #         temporal_kernel = 3
# #         if spatial == 128:
# #             spatial_kernel = 3
# #         elif spatial==96:
# #             spatial_kernel = 3
# #         elif spatial==64:
# #             spatial_kernel = 2
# #         else:
# #             spatial_kernel = 1
# #         if temporal == 1024:
# #             temporal_kernel = 3
# #         elif temporal == 512:
# #             temporal_kernel = 3
# #         elif temporal == 440:
# #             temporal_kernel = 3
# #         elif temporal == 50:
# #             temporal_kernel = 2
# #         self.final_conv = nn.Conv2d(in_channels = 200,
# #                                     out_channels = 50,
# #                                     kernel_size = (spatial_kernel,
# #                                                    temporal_kernel),
# #                                     stride = 1,
# #                                     dilation = 1,
# #                                     padding = 0)
# #         spatial_sizes = [128, 96, 64, 32, 16, 8]
# #         spatial_outs = [2, 1, 1, 1, 1, 1]
# #         temporal_sizes = [1024, 512, 440, 256, 200, 128, 100, 50]
# #         temporal_outs = [30, 14, 12, 6, 5, 2, 2, 1]
# #         inp_size = (50*
# #                     spatial_outs[spatial_sizes.index(spatial)]*
# #                     temporal_outs[temporal_sizes.index(temporal)])
# #         self.fc1 = nn.Linear(inp_size, 1000)
# #         self.fc2 = nn.Linear(1000, 40)
# #
# #     def forward(self, x):
# #         x = x.unsqueeze(0).permute(1, 0, 3, 2)
# #         y = []
# #         for i in range(5):
# #             y.append(self.temporal_layers[i](x))
# #         x = torch.cat(y, 1)
# #         y=[]
# #         for i in range(4):
# #             y.append(self. spatial_layers[i](x))
# #         x = torch.cat(y, 1)
# #         for i in range(4):
# #             x = F.relu(self.shortcuts[i](x)+self.residual_layers[i](x))
# #         x = self.final_conv(x)
# #         x = x.view(x.size()[0], -1)
# #         x = self.fc1(x)
# #         x = F.relu(x)
# #         x = self.fc2(x)
# #         return x
# #
# #     def cuda(self, gpuIndex):
# #         for i in range(len(self.temporal_layers)):
# #             self.temporal_layers[i] = self.temporal_layers[i].cuda(gpuIndex)
# #         for i in range(len(self.spatial_layers)):
# #             self.spatial_layers[i] = self.spatial_layers[i].cuda(gpuIndex)
# #         for i in range(len(self.residual_layers)):
# #             self.residual_layers[i] = self.residual_layers[i].cuda(gpuIndex)
# #         for i in range(len(self.shortcuts)):
# #             self.shortcuts[i] = self.shortcuts[i].cuda(gpuIndex)
# #         self.final_conv = self.final_conv.cuda(gpuIndex)
# #         self.fc1 = self.fc1.cuda(gpuIndex)
# #         self.fc2 = self.fc2.cuda(gpuIndex)
# #         return self
#
# ##############################################################
# # Network trainer
# ##############################################################
# def net_trainer(
#         net, loaders, opt, channel_idx, nonclasses, pretrain, train, save):
#     optimizer = getattr(torch.optim,
#                         opt.optim)(net.parameters(),
#                                    lr = opt.learning_rate)
#     if pretrain is not None:
#         net.load_state_dict(torch.load(pretrain+".pth", map_location = "cpu"))
#     # Setup CUDA
#     if not opt.no_cuda:
#         net.cuda(opt.GPUindex)
#     # Start training
#     if train:
#         for epoch in range(1, opt.epochs+1):
#             print("epoch", epoch)
#             # Initialize loss/accuracy variables
#             losses = {"train": 0.0, "val": 0.0, "test": 0.0}
#             corrects = {"train": 0.0, "val": 0.0, "test": 0.0}
#             counts = {"train": 0.0, "val": 0.0, "test": 0.0}
#             # Adjust learning rate for SGD
#             if opt.optim=="SGD":
#                 lr = opt.learning_rate*(opt.learning_rate_decay_by**
#                                         (epoch//opt.learning_rate_decay_every))
#                 for param_group in optimizer.param_groups:
#                     param_group["lr"] = lr
#             # Process each split
#             for split in ("train", "val", "test"):
#                 # Set network mode
#                 if split=="train":
#                     net.train()
#                 else:
#                     net.eval()
#                 # Process all split batches
#                 for i, (input, target) in enumerate(loaders[split]):
#                     # Check CUDA
#                     if not opt.no_cuda:
#                         if channel_idx is None:
#                             input = input.cuda(opt.GPUindex, async = True)
#                             target = target.cuda(opt.GPUindex, async = True)
#                         else:
#                             input = input[:, :, channel_idx].cuda(
#                                 opt.GPUindex, async = True)
#                             target = target.cuda(opt.GPUindex, async = True)
#                     # Wrap for autograd
#                     if split == "train":
#                         input = Variable(input)
#                         target = Variable(target)
#                         # Forward
#                         output = net(input)
#                         loss = F.cross_entropy(output, target)
#                         losses[split] += loss.item()
#                         # Compute accuracy
#                         output.data[:, nonclasses] = minus_infinity
#                         _, pred = output.data.max(1)
#                         corrects[split] += pred.eq(target.data).sum().float()
#                         counts[split] += input.data.size(0)
#                         # Backward and optimize
#                         optimizer.zero_grad()
#                         loss.backward()
#                         optimizer.step()
#                     else:
#                         with torch.no_grad():
#                             # Forward
#                             output = net(input)
#                             loss = F.cross_entropy(output, target)
#                             losses[split] += loss.item()
#                             # Compute accuracy
#                             output.data[:, nonclasses] = minus_infinity
#                             _, pred = output.data.max(1)
#                             corrects[split] += (
#                                 pred.eq(target.data).sum().float())
#                             counts[split] += input.data.size(0)
#         if save is not None:
#             torch.save(net.state_dict(), save+".pth")
#     else:
#         # Initialize loss/accuracy variables
#         losses = {"val": 0.0, "test": 0.0}
#         corrects = {"val": 0.0, "test": 0.0}
#         counts = {"val": 0.0, "test": 0.0}
#         # Process each split
#         for split in ("val", "test"):
#             # Set network mode
#             net.eval()
#             # Process all split batches
#             for i, (input, target) in enumerate(loaders[split]):
#                 # Check CUDA
#                 if not opt.no_cuda:
#                     if channel_idx is None:
#                         input = input.cuda(opt.GPUindex, async = True)
#                         target = target.cuda(opt.GPUindex, async = True)
#                     else:
#                         input = input[:, :, channel_idx].cuda(
#                             opt.GPUindex, async = True)
#                         target = target.cuda(opt.GPUindex, async = True)
#                 with torch.no_grad():
#                     # Forward
#                     output = net(input)
#                     loss = F.cross_entropy(output, target)
#                     losses[split] += loss.item()
#                     # Compute accuracy
#                     output.data[:, nonclasses] = minus_infinity
#                     _, pred = output.data.max(1)
#                     corrects[split] += (
#                         pred.eq(target.data).sum().float())
#                     counts[split] += input.data.size(0)
#     return ((corrects["val"]/counts["val"]).data.cpu().item(),
#             (corrects["test"]/counts["test"]).data.cpu().item(),
#             int(counts["val"]),
#             int(counts["test"]))
