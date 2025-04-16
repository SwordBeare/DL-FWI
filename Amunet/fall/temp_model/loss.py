import math

import numpy as np
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2

mpl.use('TkAgg')
import torch.nn as nn





def plot(gt, pre):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(gt)

    ax[1].imshow(pre)
    plt.show()


import numpy as np
import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from math import exp
import numpy as np


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

def extract_contours(para_image):
    '''
    Use Canny to extract contour features

    :param image:       Velocity open_fwi_data (numpy)
    :return:            Binary contour structure of the velocity open_fwi_data (numpy)
    '''

    image = para_image

    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image_to_255 = norm_image * 255
    norm_image_to_255 = norm_image_to_255.astype(np.uint8)
    canny = cv2.Canny(norm_image_to_255, 10, 15)
    bool_canny = np.clip(canny, 0, 1)
    return bool_canny


def extract_contours_patch(para_image):
    '''
    输入输出都是tensor数据
    :param para_image: tensor cpu
    :return: tensor cpu
    '''
    edge = para_image.detach().numpy()
    batch = edge.shape[0]
    dim = edge.shape[2], edge.shape[3]

    # result = []
    result = np.zeros([batch, 1, dim[0], dim[1]])
    for i in range(batch):
        temp_edge = extract_contours(edge[i, 0])
        result[i, 0, ...] = temp_edge
    result = torch.from_numpy(result)
    return result


class loss_all:
    def __init__(self,dim, weights=[1, 1]):
        self.criterion1 = nn.MSELoss()
        # ew = torch.from_numpy(np.array(entropy_weight).astype(np.float32)).cuda()
        self.criterion2 = nn.MSELoss()
        self.weights = weights
        self.dim =dim

    def __call__(self, out, gt):
        '''
        :param out: Output of the first decoder
        :param gt: Velocity open_fwi_data
        :return:
        '''
        ex_out = extract_contours_patch(out.to('cpu'))
        ex_gt = extract_contours_patch(gt.to('cpu'))

        loss1 = self.criterion1(out, gt)* (1-ssim(out, gt))
        loss2 = self.criterion2(torch.squeeze(ex_out), torch.squeeze(ex_gt))
        loss = self.weights[0] * loss1
        # loss = self.weights[0] * loss1 + self.weights[1] * loss2
        return loss

        # cross = self.criterion2(outputs2, torch.squeeze(targets2).long())


if __name__ == '__main__':
    import scipy.io as scio

    # data = scio.loadmat(r'E:\Allresult\datas\SEGSimulation\test_data\seismic\seismic1604')['data']
    # gt = scio.loadmat(r'E:\Allresult\datas\SEGSimulation\test_data\vmodel\vmodel1604')['data']



    # gt = torch.tensor(gt)
    # gt = gt.unsqueeze(0).unsqueeze(0)
    # # pre = pre.unsqueeze(0).unsqueeze(0)
    # batch_size = 1

    gt = torch.rand(3,1, 201, 301)*1000
    pre = torch.rand(3,1, 201, 301)*100
    # plot(gt[0,0], pre[0,0])
    batch_size = 3
    model = loss_all(1)
    out = model(gt,pre)

    F.mse_loss(gt, pre, reduction='sum')

    model = loss_all(batch_size *gt.size()[2]*gt.size()[3])
    result = model(pre, gt)


    print('214')
