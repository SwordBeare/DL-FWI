# -*- coding: utf-8 -*-
"""
Comparison of the network

Created on Sep 2024

@author: Jian An (2569222191@qq.com)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
'''------------- CBAM模块-----------------------------'''
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
class DeformConv2d(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation  # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset




class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Non-square convolution with flexible definition
        (Similar to InversionNet)

        :param in_fea:       Number of channels for convolution layer input
        :param out_fea:      Number of channels for convolution layer output
        :param kernel_size:  Size of the convolution kernel
        :param stride:       Convolution stride
        :param padding:      Convolution padding
        :param activ_fuc:    Activation function
        '''
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(activ_fuc)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SeismicRecordDownSampling(nn.Module):
    '''
    Downsampling module for seismic records
    '''
    def __init__(self, shot_num):
        super().__init__()

        self.pre_dim_reducer1 = ConvBlock(shot_num, 8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer2 = ConvBlock(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer3 = ConvBlock(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer4 = ConvBlock(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer5 = ConvBlock(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer6 = ConvBlock(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):

        width = x.shape[3]
        new_size = [width * 8, width]
        dimred0 = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

        dimred1 = self.pre_dim_reducer1(dimred0)
        dimred2 = self.pre_dim_reducer2(dimred1)
        dimred3 = self.pre_dim_reducer3(dimred2)
        dimred4 = self.pre_dim_reducer4(dimred3)
        dimred5 = self.pre_dim_reducer5(dimred4)
        dimred6 = self.pre_dim_reducer6(dimred5)

        return dimred6

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        '''
        Convolution with two basic operations

        :param in_size:         Number of channels of input
        :param out_size:        Number of channels of output
        :param is_batchnorm:    Whether to use BN
        '''
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )

    def forward(self, inputs):
        '''

        :param inputs:          Input Image
        :return:
        '''
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class unetConv11(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        '''
        Convolution with two basic operations

        :param in_size:         Number of channels of input
        :param out_size:        Number of channels of output
        :param is_batchnorm:    Whether to use BN
        '''
        super(unetConv11, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(DeformConv2d(in_size, out_size),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )

    def forward(self, inputs):
        '''

        :param inputs:          Input Image
        :return:
        '''
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, output_lim):
        '''
        Upsampling Unit
        [Affiliated with FCNVMB]

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        '''
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
        self.output_lim = output_lim
        # Transposed convolution
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        '''

        :param inputs1:      Layer of the selected coding area via skip connection
        :param inputs2:      Current network layer based on network flows
        :return:
        '''
        outputs2 = self.up(inputs2)
        input2 = F.interpolate(outputs2, size=self.output_lim, mode='bilinear', align_corners=False)

        return self.conv(torch.cat([inputs1, input2], 1))


class C_unet70(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        '''
        Network architecture of FCNVMB

        :param n_classes:       Number of channels of output (any single decoder)
        :param in_channels:     Number of channels of network input
        :param is_deconv:       Whether to use deconvolution
        :param is_batchnorm:    Whether to use BN
        '''
        super(C_unet70, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

        filters = [64, 128, 256, 512, 1024]

        self.conv0 = SeismicRecordDownSampling(5)

        self.conv1 = unetConv2(32, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.cbam = CBAM(1024)

        self.up4 = unetUp(filters[4], filters[3], self.is_deconv, (9, 9))
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv, (18, 18))
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv, (35, 35))
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv, (70, 70))
        self.final = nn.Conv2d(filters[0], self.n_classes, 1)


    def forward(self, inputs):
        '''

        :param inputs:          Input Image
        :param label_dsp_dim:   Size of the network output image (velocity open_fwi_data size)
        :return:
        '''
        data0 = self.conv0(inputs)

        data11 = self.conv1(data0)
        data1  = self.down(data11)

        data21 = self.conv2(data1)
        data2 = self.down(data21)

        data31 = self.conv3(data2)
        data3 = self.down(data31)

        data41 = self.conv4(data3)
        data4 = self.down(data41)

        center = self.center(data4)
        center = self.cbam(center)

        up4 = self.up4(data41, center)
        up3 = self.up3(data31, up4)
        up2 = self.up2(data21, up3)
        up1 = self.up1(data11, up2)

        out = self.final(up1)
        return out


if __name__ == '__main__':
    # FCNVMB
    x = torch.zeros((2, 5, 1000, 70))
    model = C_unet70(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True)
    out = model(x)
    print("out: ", out.size())

