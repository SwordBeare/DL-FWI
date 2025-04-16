# -*- coding: utf-8 -*-
"""
Comparison of the network

Created on Sep 2024

@author: Jian An (2569222191@qq.com)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class unetConv2(nn.Module):
    def __init__(self, in_channel, out_channel, is_batchnorm, is_MHSA, MHSA_size):
        '''
        Convolution with two basic operations

        :param in_channel:         Number of channels of input
        :param out_channel:        Number of channels of output
        :param is_batchnorm:    Whether to use BN
        '''
        super(unetConv2, self).__init__()

        if MHSA_size is None:
            MHSA_size = [4, 4]
        self.curr_h, self.curr_w = MHSA_size
        self.is_MHSA = is_MHSA

        # test use conv
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True),
                                   )

        if self.is_MHSA:
            self.MHSA = MHSA(out_channel,heads=4,curr_h=self.curr_h,curr_w=self.curr_w)


        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True),
                                   )

    def forward(self, inputs):
        '''

        :param inputs:          Input Image
        :return:
        '''
        outputs = self.conv1(inputs)
        if self.is_MHSA:
            N, C, H, W = outputs.shape
            P_h, P_w = H // self.curr_h, W // self.curr_w  # 4,4 or 5,5
            outputs = outputs.reshape(N * P_h * P_w, C, self.curr_h, self.curr_w)
            outputs = self.MHSA(outputs)
            outputs = outputs.permute(0, 3, 1, 2)
            N1, C1, H1, W1 = outputs.shape
            outputs = outputs.reshape(N, C1, int(H), int(W))


        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm = True, is_MHSA=True, MHSA_size=None):
        '''
        Downsampling Unit
        [Affiliated with FCNVMB]

        :param in_size:         Number of channels of input
        :param out_size:        Number of channels of output
        :param is_batchnorm:    Whether to use BN
        '''
        super(unetDown, self).__init__()

        self.conv = unetConv2(in_size, out_size, is_batchnorm,is_MHSA,MHSA_size)
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
    def __init__(self, in_size, out_size, is_deconv,is_batchnorm = True,is_MHSA=True, MHSA_size=None):
        '''
        Upsampling Unit
        [Affiliated with FCNVMB]

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        '''
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size,is_batchnorm,is_MHSA,MHSA_size)
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
        result = self.conv(torch.cat([outputs1, outputs2], 1))
        return result


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
        para_size = size / 2
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
        output_5 = output_5 + output_1
        output_6 = self.conv_5(output_5)
        output_6 = F.sigmoid(output_6)
        output = output_6 + inputs
        return output


# MHSA module
class MHSA(nn.Module):

    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type='relative', use_pos=True):
        super(MHSA, self).__init__()
        self.q_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.k_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.v_proj = GroupPointWise(in_channels, heads, proj_factor=1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type == 'relative':
            self.self_attention = RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads=True)
        else:
            raise NotImplementedError

    def forward(self, input):
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        o = self.self_attention(q=q, k=k, v=v)
        return o


class GroupPointWise(nn.Module):
    """"""

    def __init__(self, in_channels, heads=4, proj_factor=1, target_dimension=None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        self.w = nn.Parameter(
            torch.Tensor(in_channels, heads, proj_channels // heads)
        )

        nn.init.normal_(self.w, std=0.01)

    def forward(self, input):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        input = input.permute(0, 2, 3, 1).float()
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        self.w.to()
        out = torch.einsum('bhwc,cnp->bnhwp', input, self.w)
        return out


class RelPosSelfAttention(nn.Module):
    """Relative Position Self Attention"""

    def __init__(self, h, w, dim, relative=True, fold_heads=False):
        super(RelPosSelfAttention, self).__init__()
        self.relative = relative
        self.fold_heads = fold_heads
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)

    def forward(self, q, k, v):
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits += self.relative_logits(q)
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        # Relative logits in width dimension.
        rel_logits_w = self.relative_logits_1d(q, self.rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        # Relative logits in height dimension
        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim = q.shape
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits = rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _ = x.shape
        col_pad = torch.zeros((bs, heads, length, 1), dtype=x.dtype).cuda()
        x = torch.cat([x, col_pad], dim=3)
        flat_x = torch.reshape(x, [bs, heads, -1]).cuda()
        flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype).cuda()
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x = final_x[:, :, :length, length - 1:]
        return final_x


# FCNVMB Model
class MHSANET(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        '''
        Network architecture of FCNVMB

        :param n_classes:       Number of channels of output (any single decoder)
        :param in_channels:     Number of channels of network input
        :param is_deconv:       Whether to use deconvolution
        :param is_batchnorm:    Whether to use BN
        '''
        super(MHSANET, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        filters = [32, 64, 128, 256, 512, 1024]

        #The first convolution changes the channel to 32
        self.down0 = nn.Sequential(nn.Conv2d(self.in_channels, filters[0], 1, 1, 0),
                                   nn.BatchNorm2d(filters[0]),
                                   nn.ReLU(inplace=True), )

        # The second module changes channel to 64 and MHSA is added
        self.down1 = unetDown(filters[0], filters[0], self.is_batchnorm, True, MHSA_size=[5, 5])
        # The third module changes channel to 128 and MHSA is added
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm, True, MHSA_size=[5, 5])
        # The fourth module changes channel to 256 and MHSA is added
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm, True, MHSA_size=[5, 5])
        # The Fifth module changes channel to 512 and MHSA is added
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm, True, MHSA_size=[5, 5])
        # The Sixth module changes channel to 1024 and MHSA is added
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm, True, MHSA_size=[5, 5])

        self.up4 = unetUp(filters[4], filters[3], self.is_deconv,True, MHSA_size=[5, 5])
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv,True, MHSA_size=[5, 5])
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv,True, MHSA_size=[5, 5])
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv,True, MHSA_size=[5, 5])

        self.final = nn.Conv2d(filters[0], self.n_classes, 1)

    def forward(self, inputs, label_dsp_dim):
        '''

        :param inputs:          Input Image
        :param label_dsp_dim:   Size of the network output image (velocity open_fwi_data size)
        :return:
        '''

        inputs = F.interpolate(inputs, size=(400, 400), mode='bilinear', align_corners=False)
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
    # MHSA
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    in_channels = 29
    h = 400
    w = 301

    data = torch.randn(2, in_channels, h, w).to(device)

    # model = unetDown(32,64,True,False,[4,4]).to(device)
    model = MHSANET(n_classes=1, in_channels=29, is_deconv=True, is_batchnorm=True).to(device)

    out = model(data,[201, 301])


    print(out.size())


