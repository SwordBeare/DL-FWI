# -*- coding: utf-8 -*-
"""
Comparison of the network

Created on Sep 2024

@author: Jian An (2569222191@qq.com)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class unet70(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        '''
        Network architecture of FCNVMB

        :param n_classes:       Number of channels of output (any single decoder)
        :param in_channels:     Number of channels of network input
        :param is_deconv:       Whether to use deconvolution
        :param is_batchnorm:    Whether to use BN
        '''
        super(unet70, self).__init__()
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

        up4 = self.up4(data41, center)
        up3 = self.up3(data31, up4)
        up2 = self.up2(data21, up3)
        up1 = self.up1(data11, up2)

        out = self.final(up1)
        return out


if __name__ == '__main__':
    # FCNVMB
    x = torch.zeros((2, 5, 1000, 70))
    model = unet70(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True)
    out = model(x)
    print("out: ", out.size())

