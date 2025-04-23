# -*- coding = utf-8 -*-
# @Time:2022/10/20 10:36
# @Author : ZHANGTONG
# @File:clpstnet.py
# @Software:PyCharm
from torch import nn
import torch

def conv3x3(in_channels, out_channels, stride, padding, dilation, groups):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                     groups=groups, bias=False)


def conv1x1(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Conv_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv_block, self).__init__()
        self.conv3_3 = nn.Conv2d(input_channels, output_channels, kernel_size=3)

    def forward(self, x):
        out = self.DB(x)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channles):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU()
        self.conv = nn.Conv2d(in_channels, out_channles, kernel_size=1, stride=1, bias=False)

    def forward(self, input):
        out = self.conv(self.relu(self.bn(input)))
        return out


class convBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding=1, stride=1):
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class UpconvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding=1, stride=1):
        super(UpconvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                                         padding=kernel_size // 2 - 1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.upconv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Residual_convBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding=1, stride=1):
        super(Residual_convBlock, self).__init__()
        self.DB = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU())
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding='same')

    def forward(self, x):
        out = self.DB(x)
        x0 = self.conv1(x)
        # print(out.size())
        # print(x0.size())
        out = out + x0
        return out


class Progressive_BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Progressive_BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Progressive_Block1(nn.Module):
    def __init__(self, in_channels, channel_1x1, channel_3x3):
        super(Progressive_Block1, self).__init__()
        self.secret_deep = 6
        self.inter_channels = 64
        self.grow_rate = 32
        self.progressive_block1_branch1 = Progressive_BasicBlock(in_channels, channel_1x1, kernel_size=1)
        self.progressive_block1_branch2 = nn.Sequential(
            Progressive_BasicBlock(in_channels, channel_1x1, kernel_size=1),
            Progressive_BasicBlock(channel_1x1, channel_3x3, kernel_size=3))
        self.progressive_block1_branch3 = nn.Sequential(
            Progressive_BasicBlock(in_channels, self.inter_channels, kernel_size=1),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=3, padding=6, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        self.progressive_block1_branch4 = nn.Sequential(
            Progressive_BasicBlock(in_channels, self.inter_channels, kernel_size=1),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=6, padding=12, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        self.out_channels = channel_1x1 + channel_3x3 + 2 * self.grow_rate
        self.residual_connection = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

    def forward(self, x, secret, carrier):
        res = self.residual_connection(x)
        branch1 = self.progressive_block1_branch1(x)
        branch2 = self.progressive_block1_branch2(x)
        branch3 = self.progressive_block1_branch3(x)
        branch4 = self.progressive_block1_branch4(x)
        outputs = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        outputs += res
        return outputs


class Progressive_Block2(nn.Module):
    def __init__(self, in_channels, channel_1x1, channel_3x3_1, channel_3x3, channel_5x5_1, channel_5x5, dilation1,
                 dilation2):
        super(Progressive_Block2, self).__init__()
        self.secret_deep = 6
        self.inter_channels = 64
        self.grow_rate = 32

        self.progressive_block2_branch1 = Progressive_BasicBlock(in_channels, channel_1x1, kernel_size=1)
        self.progressive_block2_branch2 = nn.Sequential(
            Progressive_BasicBlock(in_channels, channel_3x3_1, kernel_size=1),
            Progressive_BasicBlock(channel_3x3_1, channel_3x3, kernel_size=3))
        self.progressive_block2_branch3 = nn.Sequential(
            Progressive_BasicBlock(in_channels, channel_5x5_1, kernel_size=1),
            Progressive_BasicBlock(channel_5x5_1, channel_5x5, kernel_size=3),
            Progressive_BasicBlock(channel_5x5, channel_5x5, kernel_size=3))
        self.progressive_block2_branch4 = nn.Sequential(
            Progressive_BasicBlock(in_channels, self.inter_channels, kernel_size=1),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=dilation1, padding=dilation1, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        self.progressive_block2_branch5 = nn.Sequential(
            Progressive_BasicBlock(in_channels, self.inter_channels, kernel_size=1),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=dilation2, padding=dilation2, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())

        self.output_channels = channel_1x1 + channel_3x3 + channel_5x5 + 2 * self.grow_rate
        self.residual_connection = nn.Conv2d(in_channels, self.output_channels, kernel_size=1)

    def forward(self, x):
        res = self.residual_connection(x)
        branch1 = self.progressive_block2_branch1(x)
        # print(branch1.size())
        branch2 = self.progressive_block2_branch2(x)
        # print(branch2.size())
        branch3 = self.progressive_block2_branch3(x)
        # print(branch3.size())
        branch4 = self.progressive_block2_branch4(x)
        # print(branch4.size())
        branch5 = self.progressive_block2_branch5(x)
        # print(branch5.size())
        outputs = torch.cat((branch1, branch2, branch3, branch4, branch5), dim=1)
        outputs += res
        return outputs


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=32, drop_rate=0.1, dilation_rate=1):
        super(DenseLayer, self).__init__()
        self.dl = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(growth_rate * 4),
            nn.LeakyReLU(),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, stride=1, dilation=dilation_rate, padding=1,
                      bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.LeakyReLU(),
            nn.Dropout2d(0.1), )

    def forward(self, input):
        out = self.dl(input)
        out = torch.cat([input, out], dim=1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, grow_rate=32):
        super(DenseBlock, self).__init__()
        self.dense = self.make_dense(num_layers, in_channels, grow_rate)

    def make_dense(self, num_layers, in_channels, grow_rate):
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * grow_rate, grow_rate))
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.dense(input)
        return out


class Encoder_Network(nn.Module):
    def __init__(self):
        super(Encoder_Network, self).__init__()
        self.secret_deep = 6
        self.inter_channels = 64
        self.grow_rate = 32

        # self.origin_conv = nn.Conv2d(3+self.secret_deep, 16, kernel_size=3, stride=1, padding=1)
        # self.origin_norm = nn.BatchNorm2d(16)

        # self.convBlock1 = Residual_convBlock(16,32)
        # self.convBlock2 = Residual_convBlock(32,64)
        # self.convBlock3 = Residual_convBlock(64,128)
        # self.convBlock4 = Residual_convBlock(128,256)
        # self.convBlock5 = Residual_convBlock(256,512)
        self.conBlock1 = Residual_convBlock(9, 16, kernel_size=3)
        self.conBlock2 = Residual_convBlock(16, 32, kernel_size=3)
        self.conBlock3 = Residual_convBlock(32, 64, kernel_size=3)
        self.conBlock4 = Residual_convBlock(64, 96, kernel_size=3)

        # self.downConv = convBlock(96,128,kernel_size=3,stride=2)
        # 128 *64 *64

        """down ConvBlock 01
        Iuput : 128,128,128
        Output: 352,64,64
        """
        self.DenseBlock1 = DenseBlock(4, 96)
        self.downBlock1 = convBlock(224, 192, kernel_size=3, stride=2)
        self.progressive_block_1 = Progressive_Block2(in_channels=192, channel_1x1=64, channel_3x3_1=96,
                                                      channel_3x3=128, channel_5x5_1=32, channel_5x5=96,
                                                      dilation1=3, dilation2=6)
        # output_channels = 64 +128+96+64 =352

        """down ConvBlock 02
        Iuput : 352,64,64
        Output: 528,32,32
        """
        self.DenseBlock2 = DenseBlock(4, 352)  # 352+128 = 480
        self.downBlock2 = convBlock(480, 480, kernel_size=3, stride=2)
        self.progressive_block_2 = Progressive_Block2(in_channels=480, channel_1x1=192, channel_3x3_1=96,
                                                      channel_3x3=208, channel_5x5_1=24, channel_5x5=64,
                                                      dilation1=6, dilation2=12)

        """down ConvBlock 03
        Iuput : 528,32,32
        Output: 512,16,16
        """
        self.DenseBlock3 = DenseBlock(4, 528)  # out_channels = 512 +32*4 = 656
        self.downBlock3 = convBlock(656, 512, kernel_size=3, stride=2)  # out = 528 * 16 * 16
        self.progressive_block_3 = Progressive_Block2(in_channels=512, channel_1x1=160, channel_3x3_1=112,
                                                      channel_3x3=224, channel_5x5_1=24, channel_5x5=64,
                                                      dilation1=12, dilation2=18)  # 512

        """multi-scale transtionBlock design
        Iuput : 512,16,16
        Output: 512,16,16
        """
        self.transtionBlock1 = Progressive_Block2(in_channels=512, channel_1x1=128, channel_3x3_1=128,
                                                  channel_3x3=256, channel_5x5_1=24, channel_5x5=64,
                                                  dilation1=3, dilation2=6)
        self.transtionBlock2 = Progressive_Block2(in_channels=512, channel_1x1=112, channel_3x3_1=144,
                                                  channel_3x3=288, channel_5x5_1=32, channel_5x5=64,
                                                  dilation1=12, dilation2=18)

        """up convBlock 01"""
        self.concat_down1 = convBlock(512, 256, kernel_size=1, padding='same')
        self.DenseBlock4 = DenseBlock(4, 528 + 256)  # out_channels = 528+256+128=912
        self.td1 = Transition(912, 512)
        self.progressive_block_4 = Progressive_Block2(in_channels=512, channel_1x1=128, channel_3x3_1=128,
                                                      channel_3x3=256, channel_5x5_1=24, channel_5x5=64,
                                                      dilation1=3, dilation2=6)  # 512
        self.upBlock1 = UpconvBlock(512, 256, kernel_size=4, stride=2)  # 32 *32 * 480

        """up convBlock 02"""
        self.concat_down2 = convBlock(528, 256, kernel_size=1, padding='same')
        self.DenseBlock5 = DenseBlock(4, 256 + 256)  # out_channels = 512 +32*4 = 640
        self.td2 = Transition(640, 480)
        self.progressive_block_5 = Progressive_Block2(in_channels=480, channel_1x1=192, channel_3x3_1=96,
                                                      channel_3x3=208, channel_5x5_1=24, channel_5x5=64,
                                                      dilation1=6, dilation2=12)
        self.upBlock2 = UpconvBlock(528, 192, kernel_size=4, stride=2)  # 64 *64 * 480

        """up convBlock 03"""
        self.concat_down3 = convBlock(352, 180, kernel_size=1, padding='same')
        self.DenseBlock6 = DenseBlock(4, 192 + 180)  # out_channels = 512 +32*4 = 640
        self.td3 = Transition(500, 320)
        self.progressive_block_6 = Progressive_Block2(in_channels=320, channel_1x1=128, channel_3x3_1=128,
                                                      channel_3x3=192, channel_5x5_1=32, channel_5x5=96,
                                                      dilation1=12, dilation2=18)
        self.upBlock3 = UpconvBlock(480, 256, kernel_size=4, stride=2)  # 128 *128 * 224

        # self.conBlock6 = convBlock(6, 16, kernel_size=3)
        self.conBlock7 = Residual_convBlock(256, 128, kernel_size=3)
        self.conBlock8 = Residual_convBlock(128, 64, kernel_size=3)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding='same')

    def forward(self, x, secret, carrier):
        out = self.conBlock1(x)
        # out =torch.cat((out,secret),dim=1)

        out = self.conBlock2(out)
        # out =torch.cat((out,secret),dim=1)
        out = self.conBlock3(out)
        # out =torch.cat((out,secret),dim=1)
        out = self.conBlock4(out)
        # print(out.size())

        out = self.DenseBlock1(out)
        out = self.downBlock1(out)
        out = self.progressive_block_1(out)
        concat_connection1 = out

        out = self.DenseBlock2(out)
        out = self.downBlock2(out)
        out = self.progressive_block_2(out)
        concat_connection2 = out

        out = self.DenseBlock3(out)
        # print(out.size())
        out = self.downBlock3(out)
        out = self.progressive_block_3(out)
        concat_connection3 = out

        out = self.transtionBlock1(out)
        out = self.transtionBlock2(out)

        # print(concat_connection3.size())
        concat_connection3 = self.concat_down1(concat_connection3)
        # print(out.size())
        # print(concat_connection3.size())
        out = torch.cat((out, concat_connection3), dim=1)
        out = self.DenseBlock4(out)
        out = self.td1(out)
        out = self.progressive_block_4(out)
        # print(out.size())
        out = self.upBlock1(out)

        # print(concat_connection2.size())
        concat_connection2 = self.concat_down2(concat_connection2)
        # print(concat_connection2.size())
        # print(out.size())

        out = torch.cat((out, concat_connection2), dim=1)
        out = self.DenseBlock5(out)
        out = self.td2(out)
        out = self.progressive_block_5(out)
        out = self.upBlock2(out)

        concat_connection1 = self.concat_down3(concat_connection1)
        out = torch.cat((out, concat_connection1), dim=1)
        out = self.DenseBlock6(out)
        out = self.td3(out)
        out = self.progressive_block_6(out)
        out = self.upBlock3(out)

        out = self.conBlock7(out)
        out = self.conBlock8(out)
        out = self.final_conv(out)
        out += carrier
        return out