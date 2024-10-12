import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from util import sample_and_group


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        BtimesN, _, k = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool1d(x, 1).view(BtimesN,
                                             -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)
        attention = F.softmax(energy / np.sqrt(x_q.size(-1)), dim=-2)
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        x = x + x_r

        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=128):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        batch_size, _, N = x.size()
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class TSegFormer(nn.Module):
    def __init__(self, args, part_num=33):
        super(TSegFormer, self).__init__()
        self.part_num = part_num
        self.args = args

        self.conv1 = nn.Conv1d(7, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)
        self.pt_last = Point_Transformer_Last(args)
        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.label_conv = nn.Sequential(nn.Conv1d(2, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.convs4 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp4 = nn.Dropout(0.5)
        self.convs5 = nn.Conv1d(512, 256, 1)
        self.convs6 = nn.Conv1d(256, 2, 1)
        self.bns4 = nn.BatchNorm1d(512)
        self.bns5 = nn.BatchNorm1d(256)

    def forward(self, x, cls_label):
        xyz = x.permute(0, 2, 1)[:, :, 0:3]
        batch_size, _, N = x.size()

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=N, radius=0.15, nsample=32, xyz=xyz,
                                                points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=N, radius=0.2, nsample=32, xyz=new_xyz,
                                                points=feature)
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = self.conv_fuse(x)

        x_max_feature = F.adaptive_max_pool1d(x, 1).repeat(1, 1, N)
        x_avg_feature = F.adaptive_avg_pool1d(x, 1).repeat(1, 1, N)

        cls_label_one_hot = cls_label.view(batch_size, 2, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat([x_max_feature, x_avg_feature, cls_label_feature], 1)
        all_fea = torch.cat((x, x_global_feature), 1)

        x = F.relu(self.bns1(self.convs1(all_fea)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        y = F.relu(self.bns4(self.convs4(all_fea)))
        y = self.dp4(y)
        y = F.relu(self.bns5(self.convs5(y)))
        y = self.convs6(y)  

        return x, y