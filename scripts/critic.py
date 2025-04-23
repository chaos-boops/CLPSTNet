"""xuNet"""
# This implementation is based on the work by Zhang et al.
# Reference: WZhang R, Dong S, Liu J. Invisible steganography via generative adversarial networks[J]. Multimedia tools and applications, 2019, 78(7): 8559-8575.

import torch.nn as nn
import torch
import torch.nn.functional as F


class XuNet(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(XuNet, self).__init__()
        # self.preprocessing = kv_conv2d()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=kernel_size, padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(5, 2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(5, 2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.AvgPool2d(kernel_size=5, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        # self.pool5 = nn.AvgPool2d(kernel_size=16, stride=1)
        # self.pool6 = nn.AvgPool2d(kernel_size=4, stride=4)
        # self.pool7 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(3840, 128)
        self.fc2 = nn.Linear(128, 2)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x  # self.preprocessing(x)
        out = self.conv1(out)

        out = torch.abs(out)
        # print(out)
        out = self.relu(self.bn1(out))
        out = self.pool1(out)

        out = self.tanh(self.bn2(self.conv2(out)))
        out = self.pool2(out)

        out = self.relu(self.bn3(self.conv3(out)))
        out = self.pool3(out)

        out = self.relu(self.bn4(self.conv4(out)))
        out = self.pool4(out)

        out = self.relu(self.bn5(self.conv5(out)))

        _, _, x, y = out.size()
        out1 = F.avg_pool2d(out, kernel_size=int(x / 4), stride=int(x / 4))
        out2 = F.avg_pool2d(out, kernel_size=int(x / 2), stride=int(x / 2))
        out3 = F.avg_pool2d(out, kernel_size=(x, y), stride=1)

        # print(out1.size())
        # print(out2.size())
        # print(out3.size())

        # out1 = out1.view(out1.size(0), -1)
        # out2 = out2.view(out2.size(0), -1)
        # out3 = out3.view(out3.size(0), -1)
        # print(out1.size())
        # print(out2.size())
        # print(out3.size())
        out1 = out1.reshape(out1.size(0), 3200)
        out2 = out2.reshape(out2.size(0), 512)
        out3 = out3.reshape(out3.size(0), 128)

        out = torch.cat([out1, out2, out3], dim=1)

        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return self.sigmoid(out)
