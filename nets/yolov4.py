# -*- coding: utf-8 -*-
# @Time    : 2021/8/4 0004 下午 18:09
# @Author  : Ruihao
# @FileName: yolov4.py
# @ProjectName:yolov4_own

# 引入有序字典
from collections import OrderedDict

import torch
import torch.nn as nn

# 从CSPdarknet.py导入函数darknet53
from nets.CSPdarknet import darknet53

#---------------------------------------------------#
# 定义CBL:conv + bn + relu结构的卷积块
#---------------------------------------------------#
def conv2d(filter_in, filter_out, kernel_size, stride = 1):
    # 设置填充大小
    pad = (kernel_size -1) // 2 if kernel_size else 0
    # 借助Sequential和有序字典，搭建CBL卷积结构
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, padding=pad, stride=stride, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
# Spp结构，设置不同大小池化核进行池化，并Cat多个池化结构
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes = [5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList(
            [nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    # 定义前向传播过程
    def forward(self, x):
        # 进行多个池化
        features = [maxpool(x) for maxpool in self.maxpools[::-1]] # 池化后仍是一个
        print("features", features)
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
# 卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1), # CBL Conv结构
            nn.Upsample(scale_factor=2, mode='nearest') # nearest方法进行上采样
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(in_filters, filters_list):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1), # CBL, ks = 1
        conv2d(filters_list[0], filters_list[1], 3), # CBL, ks = 1
        conv2d(filters_list[1], filters_list[0], 1), # CBL, ks = 1
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(in_filters, filters_list):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1), # CBL, ks = 1
        conv2d(filters_list[0], filters_list[1], 3), # CBL, ks = 1
        conv2d(filters_list[1], filters_list[0], 1), # CBL, ks = 1
        conv2d(filters_list[0], filters_list[1], 3),  # CBL, ks = 1
        conv2d(filters_list[1], filters_list[0], 1),  # CBL, ks = 1
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#   CBL + Conv2d
#---------------------------------------------------#
def yolo_head(in_filters, filters_list):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1)
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    '''
    :param num_anchors: Anchor的数量
            num_classes:类别的数量
    '''
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        # ---------------------------------------------------#
        # 有CSPdarknet53得到三个特征层
        # 用来进行特征传播和级联
        # 输出特征shape:
        # 52 52 256
        # 26 26 512
        # 13 13 1024
        # ---------------------------------------------------#
        self.channel_low = [512, 1024] # 对应模型架构Neck底层的通道数
        self.channel_mid = [256, 512] # 对应模型架构Neck中层的通道数
        self.channel_high = [128, 256] # 对应模型架构Neck高层的通道数

        self.backbone = darknet53(None)

        # Neck部分

        #--------------------------------------------------#
        # First Neck Layer
        # --------------------------------------------------#
        # CBL*3 + SPP + CBL*3
        self.conv1 = make_three_conv(1024, self.channel_low)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv(2048, self.channel_low)

        # CBL + Upsamping = Upsample
        self.upsample1 = Upsample(512, 256)
        # Get feature form P4, and user 1x1 conv to modify channels
        self.conv_for_P4 = conv2d(512, 256, 1)
        # Five conv after concat
        self.make_five_conv1 = make_five_conv(512, self.channel_mid)

        self.upsample2 = Upsample(256, 128)
        # Get feature form P3, and user 1x1 conv to modify channels
        self.conv_for_P3 = conv2d(256, 128, 1)
        # Five conv after concat
        self.make_five_conv2 = make_five_conv(256, self.channel_high)

        #--------------------------------------------------#
        # First Head Layer
        # --------------------------------------------------#
        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolohead3 = yolo_head(128, [self.channel_high[1], final_out_filter2])

        #--------------------------------------------------#
        # Second Neck Layer
        # --------------------------------------------------#
        # CBL * 1
        self.down_sample1 = conv2d(128, 256, 3, stride=2)
        # CBL * 5, out: 256
        self.make_five_conv3 = make_five_conv(512, [256, 512])

        #--------------------------------------------------#
        # Second Head Layer
        # --------------------------------------------------#
        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        final_out_filter1 = num_anchors * (5 + num_classes)
        self.yolohead2 = yolo_head(256, [self.channel_mid[1], final_out_filter1])

        # --------------------------------------------------#
        # Third Neck Layer
        # --------------------------------------------------#
        # CBL * 1
        self.down_sample2 = conv2d(256, 512, 3, stride=2)
        # CBL * 5, out channels:512
        self.make_five_conv4 = make_five_conv(1024, [512, 1024])

        # --------------------------------------------------#
        # Third Head Layer
        # --------------------------------------------------#
        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        final_out_filter0 = num_anchors * (5 + num_classes)
        self.yolohead1 = yolo_head(512, [self.channel_low[1], final_out_filter0])

    def forward(self, x):
        # output of backbone
        x2, x1, x0 = self.backbone(x)

        #--------------------------------------------------#
        # First Neck Layer
        # --------------------------------------------------#
        # 这不是标准的SPP步骤, 少了一个1x1 conv
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
        P5 = self.conv1(x0) # CBL * 3
        P5 = self.SPP(P5) # SPP
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.conv2(P5) # CBL * 3

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5) # 1x1 conv + upsamping

        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1) # CBL * 1
        # P5_upsample + P4
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4, P5_upsample], axis = 1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 = 52,52,256
        P3 = torch.cat([P3, P4_upsample], axis=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.make_five_conv2(P3)

        #--------------------------------------------------#
        # Second Neck Layer
        # --------------------------------------------------#
        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample, P4], axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv3(P4)

        #--------------------------------------------------#
        # Third Neck Layer
        # --------------------------------------------------#
        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample, P5], axis=1)
        print("P5_1.shape", P5.shape)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.make_five_conv4(P5)
        print("P5_2.shape", P5.shape)

        #--------------------------------------------------#
        # First Head Layer
        # y3=(batch_size,75,52,52)
        # --------------------------------------------------#
        out_top = self.yolohead3(P3)

        #--------------------------------------------------#
        # Second Head Layer
        # y2=(batch_size,75,26,26)
        # --------------------------------------------------#
        out_mid = self.yolohead2(P4)

        #--------------------------------------------------#
        # Third Head Layer
        # y1=(batch_size,75,13,13)
        # --------------------------------------------------#
        out_low = self.yolohead1(P5)

        return out_low, out_mid, out_top

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(1,3,416,416))
    print(x)
    num_anchors, num_classes = 9, 10
    model = YoloBody(num_anchors, num_classes)
    out_low, out_mid, out_top = model(x)
    print('Output shape:',out_low.shape)











