from math import exp
import torch
import torch.nn.functional as F

def gaussian(window_size, sigma):
    # 计算公式:e^(-x^2)/(2*sigma^2)，其中x表示距离中心点的距离，sigma默认1.5
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    # 数据归一化
    return gauss / gauss.sum()


# 计算滑动窗口权重
def create_window(window_size, channel):
    # 利用滑动窗口尺寸先计算一个一维，并且服从正态分布的数据
    # 注意这里利用unsqueeze函数扩了一下维度，从行向量变为了列向量
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # 列向量乘以行向量，变为n*n的矩阵，正好对应窗口权重
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 沿通道维度复制channel遍，每个通道对应一个权重(这里所有通道权重相同，均服从正态分布)，并且变为连续存储的数据
    window = (_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    # 返回窗口权重数据
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # 计算每个滑动窗口的均值
    # 卷积运算正好是窗口数据按权重求和再取均值，因此可以利用二维卷积运算来计算窗口中数据的均值
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    # 均值取平方，即E^2(X)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    # 计算E(X)E(Y)，用于后续计算协方差
    mu1_mu2 = mu1 * mu2
    # 依次计算img1与img2的方差
    # 这里计算方差利用公式D(X)=E(X^2)-E^2(X)，其中E^2(X)表示均值的平方，即上述公式中的mu1_sq、mu2_sq、mu1_mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    # 计算img1、img2之间的协方差
    # 利用公式Conv(X,Y)=E(XY)-E(X)E(Y)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    # 利用上述得到的指标，传入公式计算ssim值，此时会得到一张图，最后再求均值即可
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    if (img1.size(2) > 80):
        img1 = (img1-torch.min(img1))/(torch.max(img1)-torch.min(img1))
        img2 = (img2-torch.min(img2))/(torch.max(img2)-torch.min(img2))
    # 将输入的图像数据限制为0-1之间(一般数据就是位于0-1之间，防止出现异常值)
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    # 得到图片通道数
    (_, channel, _, _) = img1.size()
    # 得到窗口的权重数据，离窗口中心越远，权重越小。权重服从高斯分布(正态分布)
    window = create_window(window_size, channel)
    # 如果图片数据储存在cuda上(即利用显卡训练)，则将窗口权重数据也传入cuda中
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    # 统一数据类型
    window = window.type_as(img1)
    # 调用_ssim，计算ssim值
    return _ssim(img1, img2, window, window_size, channel, size_average)

if __name__ == '__main__':
    img1=torch.randn(50,1,40,40)
    img2=torch.randn(50,1,40,40)
    result = ssim(img1,img2)
    print('312')