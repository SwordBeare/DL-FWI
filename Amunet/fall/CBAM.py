from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# cbam-unet
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


'''------------- CBAM模块-----------------------------'''


class CBAM(nn.Module):
    # in_planes:输入特征图通道数
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# 卷积，在uent中卷积一般成对使用
class DoubleConv(nn.Sequential):
    # 输入通道数， 输出通道数， mid_channels为成对卷积中第一个卷积层的输出通道数
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            # 3*3卷积，填充为1，卷积之后输入输出的特征图大小一致
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


# 下采样
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            # 1.最大池化的窗口大小为2， 步长为2
            nn.MaxPool2d(2, stride=2),
            # 2.两个卷积
            DoubleConv(in_channels, out_channels)
        )


# 上采样
class Up(nn.Module):
    # bilinear是否采用双线性插值
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # 使用双线性插值上采样
            # 上采样率为2，双线性插值模式
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 使用转置卷积上采样
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        # 上采样之后的特征图与要拼接的特征图，高度方向的差值
        diff_y = x2.size()[2] - x1.size()[2]
        # 上采样之后的特征图与要拼接的特征图，宽度方向的差值
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        # 1.填充差值
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # 2.拼接
        x = torch.cat([x2, x1], dim=1)
        # 3.卷积，两次卷积
        x = self.conv(x)
        return x


# 最后的1*1输出卷积
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class CBAM_UNet(nn.Module):
    # 参数: 输入通道数， 分割任务个数， 是否使用双线插值， 网络中第一个卷积通道个数
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(CBAM_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)

        # 下采样，参数:输入通道,输出通道
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        # 如果采用双线插值上采样为 2，采用转置矩阵上采样为 1
        factor = 2 if bilinear else 1
        # 最后一个下采样，如果是双线插值则输出通道为512，否则为1024
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        # 上采样，参数:输入通道,输出通道
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        # 最后的1*1输出卷积
        self.out_conv = OutConv(base_c, num_classes)
        # CBAM模块
        self.CBAM = CBAM(base_c)

    # 正向传播过程
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. 定义最开始的两个卷积层
        x1 = self.in_conv(x)
        # 2. contracting path（收缩路径）
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # 3. expanding path（扩展路径）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # cbam模块
        x = self.CBAM(x)
        # 4. 最后1*1输出卷积
        logits = self.out_conv(x)

        return nn.Sigmoid()(logits)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CBAM_UNet().to(device)
    # 打印网络结构和参数
    summary(net, (3, 256, 256))