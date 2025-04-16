import torch
import torch.nn as nn
import torch.nn.functional as F


class Multi_scale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Multi_scale, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, int(out_channel / 4), 3, 1, 1),
                                   nn.BatchNorm2d(int(out_channel / 4)),
                                   nn.ReLU(inplace=True), )

        self.conv3 = nn.Sequential(nn.Conv2d(int(out_channel / 4), int(out_channel / 4), 3, 1, 3, dilation=3),
                                   nn.BatchNorm2d(int(out_channel / 4)),
                                   nn.ReLU(inplace=True), )

        self.conv4 = nn.Sequential(nn.Conv2d(int(out_channel / 4), int(out_channel / 2), 3, 1, 5, dilation=5),
                                   nn.BatchNorm2d(int(out_channel / 2)),
                                   nn.ReLU(inplace=True), )

    def forward(self, inputs):
        x0 = self.conv1(inputs)

        x1 = self.conv2(inputs)
        x2 = self.conv3(x1)
        x_temp = x1 + x2

        x3 = self.conv4(x_temp)
        x_temp_2 = x3 + inputs

        x_result = torch.cat([x1, x_temp, x_temp_2], 1)

        output = x0 + x_result
        return output


class eca(nn.Module):
    # 这里用的是一维卷积
    # kernel_size是图像的大小
    # kernel_size_1是一维卷积适应性的大小
    def __init__(self, in_out_channel, kernel_size, kernel_size_1):
        super(eca, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_out_channel, in_out_channel, kernel_size, 1, 0,
                                        groups=in_out_channel, bias=False)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size_1, padding=(kernel_size_1 - 1) // 2,
                              bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        y = self.depthwise_conv(inputs)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        attention = self.softmax(y)

        result = inputs * attention.expand_as(inputs)
        result = result + inputs
        return result


class Gfm(nn.Module):
    def __init__(self, in_channel, k, s, av_size):
        super(Gfm, self).__init__()
        self.conv_n_64 = nn.Conv2d(in_channel, 64, 1, 1, 0)
        self.conv_16_64 = nn.Conv2d(16, 64, 1, 1, 0)
        self.conv_n_16 = nn.Conv2d(in_channel, 16, 1, 1, 0)

        self.conv_n = nn.Conv2d(16, 16, k, s, 1)

        self.av_size = av_size

    def forward(self, inputs):
        x1 = F.avg_pool2d(inputs, kernel_size=self.av_size, stride=self.av_size, ceil_mode=True)
        x1 = self.conv_n_64(x1)

        x2 = self.conv_n_16(inputs)
        x2 = self.conv_n(x2)
        x2 = self.conv_16_64(x2)

        x = x1 + x2

        return x

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
        outputs1 = self.up(inputs1)
        offset1 = (outputs1.size()[2] - inputs2.size()[2])
        offset2 = (outputs1.size()[3] - inputs2.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]

        # Skip and concatenate
        outputs2 = F.pad(inputs2, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
class MHSAunet(nn.Module):
    def __init__(self, in_channel):
        super(MHSAunet, self).__init__()
        self.is_deconv = True
        filters = [64, 128, 256, 512, 1024]
        # 这里用3,3卷积还是1,1卷积
        self.down0 = nn.Sequential(nn.Conv2d(in_channel, filters[0], 3, 1, 1),
                                   nn.BatchNorm2d(filters[0]),
                                   nn.ReLU(inplace=True), )

        self.Msm1 = Multi_scale(filters[0], filters[1])
        self.eca1 = eca(filters[1], (400, 301), 3)
        self.poll = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.Msm2 = Multi_scale(filters[1], filters[2])
        self.eca2 = eca(filters[2], (200, 151), 5)
        # self.poll = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.Msm3 = Multi_scale(filters[2], filters[3])
        self.eca3 = eca(filters[3], (100, 76), 5)
        # self.poll = nn.MaxPool2d(2, 2, ceil_mode=True)

        # self.Msm4 = Multi_scale(filters[3], filters[4])
        self.down1 = nn.Sequential(nn.Conv2d(filters[3], filters[3], 3, 1, 1),
                                   nn.BatchNorm2d(filters[3]),
                                   nn.ReLU(inplace=True), )
        self.eca4 = eca(filters[3], (50, 38), 5)
        self.poll = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.gfm1 = Gfm(128,9,8,8)
        self.gfm2 = Gfm(256,5,4,4)
        self.gfm3 = Gfm(512,3,2,2)

        self.up =nn.Conv2d(704, 1024, 3, 1,1)

        self.up1 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up2 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up3 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up4 = unetUp(filters[1], filters[0], self.is_deconv)

        self.final = nn.Conv2d(filters[0], 1, 1)
    def forward(self, inputs,label_dsp_dim):
        x0 = self.down0(inputs)

        x = self.Msm1(x0)
        x = self.eca1(x)
        x1 = self.poll (x)

        x = self.Msm2(x1)
        x = self.eca2(x)
        x2 = self.poll(x)

        x = self.Msm3(x2)
        x = self.eca3(x)
        x3 = self.poll(x)

        x = self.down1(x3)
        x = self.eca4(x)
        x4 = self.poll(x)

        x_skip1 = self.gfm1(x1)
        x_skip2 = self.gfm2(x2)
        x_skip3 = self.gfm3(x3)

        x = torch.cat([x4, x_skip1,x_skip2,x_skip3], 1)
        x = self.up(x)

        x = self.up1(x,x3)
        x = self.up2(x,x2)
        x = self.up3(x,x1)
        x = self.up4(x,x0)

        x = x[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()

        x = self.final(x)

        return x

if __name__ == '__main__':

    data = torch.randn(2, 29, 400, 301)
    model = MHSAunet(29)
    result = model(data,[201, 301])



print('213')
