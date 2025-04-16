# -*- coding: utf-8 -*-
"""
Comparison of the network

Created on Sep 2024

@author: Jian An (2569222191@qq.com)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class unetConv_esa(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        '''
        Convolution with two basic operations

        :param in_size:         Number of channels of input
        :param out_size:        Number of channels of output
        :param is_batchnorm:    Whether to use BN
        '''
        super(unetConv_esa, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(wide_conv(in_size, out_size),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
            self.esa = ESA( out_size, is_deconv=True)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(wide_conv(in_size, out_size),
                                       nn.ReLU(inplace=True), )
            self.esa = ESA(out_size, is_deconv=False)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )

    def forward(self, inputs):
        '''

        :param inputs:          Input Image
        :return:
        '''
        outputs = self.conv1(inputs)
        outputs = self.esa(outputs)
        outputs = self.conv2(outputs)
        return outputs

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
            self.conv1 = nn.Sequential(wide_conv(in_size, out_size),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(wide_conv(in_size, out_size),
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


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        '''
        Downsampling Unit
        [Affiliated with FCNVMB]

        :param in_size:         Number of channels of input
        :param out_size:        Number of channels of output
        :param is_batchnorm:    Whether to use BN
        '''
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        '''

        :param inputs:          Input Image
        :return:
        '''
        outputs = self.conv(inputs)
        outputs = self.down(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        '''
        Upsampling Unit
        [Affiliated with FCNVMB]

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        '''
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
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
        offset1 = (outputs2.size()[2] - inputs1.size()[2])
        offset2 = (outputs2.size()[3] - inputs1.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]

        # Skip and concatenate
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class wide_conv(nn.Module):
    def __init__(self, in_size, out_size):
        super(wide_conv, self).__init__()
        self.conv_1 = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_3 = nn.Conv2d(in_size, out_size, 3, 1, 1)
        self.conv_5 = nn.Conv2d(in_size, out_size, 5, 1, 2)

    def forward(self, inputs):
        outputs_1 = self.conv_1(inputs)
        outputs_3 = self.conv_3(inputs)
        outputs_5 = self.conv_5(inputs)
        outputs_all = outputs_1 + outputs_3 + outputs_5
        return outputs_all


class ESA(nn.Module):
    def __init__(self, size, is_deconv=True):
        '''
        ESA Spatial Attention Unit

        :param size: Number of channels of input and output
        :param is_deconv: Whether to use deconvolution
        '''
        super(ESA, self).__init__()
        para_size = int(size / 2)
        self.conv_1 = nn.Conv2d(size, para_size, 1, 1, 0)
        self.conv_2 = nn.Conv2d(para_size, para_size, 3, 1, 1)
        self.conv_3 = nn.Conv2d(para_size, size, 3, 2, 1)  # Downsampling
        self.conv_4 = nn.Conv2d(size, size, 3, 1, 1)

        if is_deconv:
            self.up = nn.ConvTranspose2d(size, para_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_5 = nn.Conv2d(para_size, size, 1, 1, 0)

    def forward(self, inputs):
        output_1 = self.conv_1(inputs)
        output_2 = self.conv_2(output_1)
        output_3 = self.conv_3(output_2)
        output_4 = self.conv_4(output_3)
        output_5 = self.up(output_4)
        output_5 = output_5[:, :, :25, :19]
        output_5 = output_5 + output_1
        output_6 = self.conv_5(output_5)
        output_6 = F.sigmoid(output_6)
        output = output_6 + inputs
        return output


class AJ_net_esa_widecon(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        '''
        Network architecture of FCNVMB

        :param n_classes:       Number of channels of output (any single decoder)
        :param in_channels:     Number of channels of network input
        :param is_deconv:       Whether to use deconvolution
        :param is_batchnorm:    Whether to use BN
        '''
        super(AJ_net_esa_widecon, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        filters = [64, 128, 256, 512, 1024]

        self.down0 = nn.Conv2d(self.in_channels, filters[0], 1, 1, 0)

        self.down1 = unetDown(filters[0], filters[0], self.is_batchnorm)

        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv_esa(filters[3], filters[4], self.is_batchnorm)
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], self.n_classes, 1)

    def forward(self, inputs, label_dsp_dim):
        '''

        :param inputs:          Input Image
        :param label_dsp_dim:   Size of the network output image (velocity open_fwi_data size)
        :return:
        '''
        down0 = self.down0(inputs)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.center(down4)
        up4 = self.up4(down4, center)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        up1 = self.up1(down1, up2)
        up1 = up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()

        return self.final(up1)


if __name__ == '__main__':
    # FCNVMB
    x = torch.zeros((5, 29, 400, 301))
    model = AJ_net_esa_widecon(n_classes=1, in_channels=29, is_deconv=True, is_batchnorm=True)
    out = model(x,[201,301])
    print("out: ", out.size())
