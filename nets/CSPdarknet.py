# -*- coding: utf-8 -*-
# @Time    : 2021/8/4 0004 上午 10:30
# @Author  : Ruihao
# @FileName: CSPdarknet.py
# @ProjectName:yolov4_own

# 导入torch包
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入数学运算包
import math

#-------------------------------------------------#
#   MISH激活函数
#   Mish激活函数公式：
#   Mish = x * tanh(ln(1 + e ^ x))
#   Torch实现：
#   Softplus(x) = 1 / β∗log(1 + exp(β∗x)), note: β = 1
#-------------------------------------------------#
class Mish(nn.Module):
    '''
    描述：Mish激活函数
    参数 x：输入tensor
    '''
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#---------------------------------------------------#
#   基础卷积块
#---------------------------------------------------#
class BasicConv(nn.Module):
    '''
    ：Description
        基础卷积块：conv2d + BN + Mish
    :param x:
    :return x:
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
        super(BasicConv, self).__init__()
        # padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        # print("x.shape", x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   CSPdarknet的结构块的组成部分Resblock
#   内部堆叠的残差块
#---------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels = None):
        super(Resblock, self).__init__()
        # 多个残差块时，内部的通道数不变，in_channels == out_channels == hidden_channels
        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


#-----------------------------------------------------#
# CSPdarknet的结构：大残差块 + 内部多个小残差块
# 架构分解：
# 1.先使用步长为2，无填充的2x2卷积，压缩长宽
# 2.建立大残差块边,绕过多个内部残差块
# 3.建立残差块主干，进行num_blocks次循环，建立内部残差结构
#-----------------------------------------------------#
class Resblock_body(nn.Module):
    '''
    :param in_channels:输入通道数
    :param out_channels:输出通道数
    :param num_blocks:resblock数量
    :param first:是否仅含一个resblock
    :return x:feature
    '''
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        #--------------------------------------------#
        # Step 1:步长为2，无填充的的2x2卷积块，压缩长宽
        #--------------------------------------------#
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        # first为主干num_blocks = 1, 仅包含一个卷积块,内部通道数减半
        if first:
            #------------------------------------------------------------------#
            # Step2：建立大残差块边self.split_conv0,绕过多个内部残差块, ks = 1
            #------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)

            # ------------------------------------------------------------------#
            # Step3: 建立残差块主干，进行num_blocks次循环，建立内部残差结构
            # ------------------------------------------------------------------#

            # 残差块的1x1卷积
            self.split_conv1 = BasicConv(out_channels, out_channels, 1) # ks = 1
            # 残差块:(Resblock + BasicConv) * 1
            self.blocks_conv = nn.Sequential(
                Resblock(channels= out_channels, hidden_channels= out_channels//2),
                BasicConv(out_channels, out_channels, 1), # ks = 1
            )
            # 合并后，使用1x1卷积整合通道
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        # 主干包含num_blocks卷积块
        else:
            #------------------------------------------------------------------#
            # Step2：建立大残差块边self.split_conv0,绕过多个内部残差块, ks = 1
            #------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)

            # ------------------------------------------------------------------#
            # Step3: 建立残差块主干，进行num_blocks次循环，建立内部残差结构
            # ------------------------------------------------------------------#

            # 残差块的1x1卷积
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1) # ks = 1
            # 残差块:(Resblock * num_blocks + BasicConv)
            self.blocks_conv = nn.Sequential(
                *[Resblock(channels= out_channels // 2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1), # ks = 1
            )
            # 合并后，使用1x1卷积整合通道
            self.concat_conv = BasicConv(out_channels, out_channels, 1)
    def forward(self, x):
        x = self.downsample_conv(x)

        # 大残差边
        x0 = self.split_conv0(x)

        # 主体残差结构
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        # concat 大残差边与主体残差结构
        x = torch.cat([x1, x0], dim=1)
        # 1x1卷积整合通道
        x = self.concat_conv(x)

        return x

#---------------------------------------------------#
#   搭建CSPdarknet53 主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
class CSPDarkNet(nn.Module):
    '''
    :param layer:每层的resblo数量
    :return: three feature layers
    '''
    def __init__(self, layer):
        super(CSPDarkNet, self).__init__()
        # 起始输入通道数
        self.inplanes = 32
        # 416，416，3 -> 416，416，32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)

        # 设置后续通道数list
        self.feature_channels = [64, 128, 256, 512, 1024]
        # 将Resblock_body添加到Module集合中，torch会自动将其注册到当前网络模型
        self.stages = nn.ModuleList([
            # 416，416，32 -> 208，208， 64
            Resblock_body(self.inplanes, self.feature_channels[0], layer[0], first=True),
            # 208，208， 64 -> 104，104, 128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layer[1], first=False),
            # 104，104, 128 -> 52, 52, 256
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layer[2], first=False),
            # 52, 52, 256 -> 26, 26, 512
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layer[3], first=False),
            # 26, 26, 512 -> 13, 13, 1024
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layer[4], first=False),
        ])

        self.num_features = 1

        # ????????????????????????????????#
        # 这部分什么功能？
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        print("x.shape", x.shape)

        x = self.stages[0](x)
        print("x1.shape", x.shape)
        x = self.stages[1](x)
        print("x2.shape", x.shape)
        out3 = self.stages[2](x)
        print("out3.shape", out3.shape)
        out4 = self.stages[3](out3)
        print("out4.shape", out4.shape)
        out5 = self.stages[4](out4)
        print("out5.shape", out5.shape)

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    # 实例化CSPDarkNet
    model = CSPDarkNet([1, 2, 8, 8, 4])
    # 加载预训练模型
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        # 加载失败
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(1,3,416,416))
    print(x)
    model = darknet53(None)
    out_low, out_mid, out_top = model(x)
    print('Darknet Output shape:',out_low.shape)












