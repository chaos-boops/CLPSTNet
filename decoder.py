# -*- coding = utf-8 -*-
# @Time:2022/10/20 10:38
# @Author : ZHANGTONG
# @File:decoder.py
# @Software:PyCharm

from torch import nn
import torch

class decode_Network(nn.Module):
    def __init__(self):
        super(decode_Network, self).__init__()
        self.inter_channels = 64
        self.grow_rate = 32

        self.origin_conv = conv3x3(3, 32, stride=1, padding=1, dilation=1, groups=1)
        self.origin_bn = nn.BatchNorm2d(32)
        self.origin_act = nn.LeakyReLU()

        self.res1 = conv1x1(32, 192, stride=1)

        self.s1_branch1 = nn.Sequential(
            conv1x1(32, 64, stride=1),
            nn.ReLU(inplace=True))
        self.s1_branch2 = nn.Sequential(
            conv3x3(32, 64, stride=1, padding=1, dilation=1, groups=1),
            nn.ReLU(inplace=True))
        self.s1_branch3 = nn.Sequential(
            conv1x1(32, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=3, padding=3, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        self.s1_branch4 = nn.Sequential(
            conv1x1(32, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=6, padding=6, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        # self.td1 = Transition(192, 192//2)  #96
        self.res2 = conv1x1(192, 288, stride=1)

        self.s2_branch1 = nn.Sequential(
            conv1x1(192, 64, stride=1),
            nn.ReLU(inplace=True))
        self.s2_branch2 = nn.Sequential(
            conv1x1(192, 96, stride=1),
            nn.ReLU(inplace=True),
            conv3x3(96, 128, stride=1, padding=1, dilation=1, groups=1),
            nn.ReLU(inplace=True))
        self.s2_branch3 = nn.Sequential(
            conv1x1(192, 16, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True))
        self.s2_branch4 = nn.Sequential(
            conv1x1(192, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=6, padding=6, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        self.s2_branch5 = nn.Sequential(
            conv1x1(192, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=12, padding=12, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())

        # self.td2 = Transition(272,272//2)
        self.res3 = conv1x1(288, 512, stride=1)

        self.s3_branch1 = nn.Sequential(
            conv1x1(288, 128, stride=1),
            nn.ReLU(inplace=True))
        self.s3_branch2 = nn.Sequential(
            conv1x1(288, 128, stride=1),
            nn.ReLU(inplace=True),
            conv3x3(128, 256, stride=1, padding=1, dilation=1, groups=1),
            nn.ReLU(inplace=True))
        self.s3_branch3 = nn.Sequential(
            conv1x1(288, 32, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True))
        self.s3_branch4 = nn.Sequential(
            conv1x1(288, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=12, padding=12, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        self.s3_branch5 = nn.Sequential(
            conv1x1(288, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=18, padding=18, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        # self.td3 = Transition(288, 288 // 2)

        self.conv_final = nn.Conv2d(512, 6, kernel_size=3, padding=1, stride=1)
        # self.sigmo = nn.Sigmoid()

    def forward(self, input):
        res = input
        out = self.origin_act(self.origin_bn(self.origin_conv(input)))
        res = self.res1(out)  # 32,192//2

        s1_branch1 = self.s1_branch1(out)
        s1_branch2 = self.s1_branch2(out)
        s1_branch3 = self.s1_branch3(out)
        s1_branch4 = self.s1_branch4(out)
        # B x 192 x 128 x128
        out = torch.cat((s1_branch1, s1_branch2, s1_branch3, s1_branch4), dim=1)
        out += res

        res = self.res2(out)
        s2_branch1 = self.s2_branch1(out)
        s2_branch2 = self.s2_branch2(out)
        s2_branch3 = self.s2_branch3(out)
        s2_branch4 = self.s2_branch4(out)
        s2_branch5 = self.s2_branch5(out)
        # B x 288 x 128 x 128
        out = torch.cat((s2_branch1, s2_branch2, s2_branch3, s2_branch4, s2_branch5), dim=1)
        out += res

        res = self.res3(out)
        s3_branch1 = self.s3_branch1(out)
        s3_branch2 = self.s3_branch2(out)
        s3_branch3 = self.s3_branch3(out)
        s3_branch4 = self.s3_branch4(out)
        s3_branch5 = self.s3_branch5(out)
        # B x 512 x 128 x 128
        out = torch.cat((s3_branch1, s3_branch2, s3_branch3, s3_branch4, s3_branch5), dim=1)
        out += res

        output = self.conv_final(out)
        return output