import torch
from .ssim import ssim
from .utils import extract_contours_patch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..param_config import *

def line(x):
    if x<=120:
        y = 1
    else:
        y = 0.2

    return y

def cross_entropy(pred, label):
    batch_size = pred.size(0)
    loss = 0
    for i in range(batch_size):
        loss = loss + F.binary_cross_entropy(pred[i][0], label[i][0])
    return loss

def loss_function(output, labels, epoch, para = 15):
    if (img1.size(2) > 80):
        loss = F.mse_loss(output, labels, reduction='sum')/(img1.size(2)*img1.size(3)*train_batch_size)
        return loss
    ssim_result = ssim(output, labels)
    loss_1 = F.mse_loss(output, labels, reduction='sum') * (1 - ssim_result)

    output1 = output.cpu().detach()
    labels1 = labels.cpu().detach()

    edge_1 = extract_contours_patch(output1)
    edge_2 = extract_contours_patch(labels1)
    loss_2 = F.mse_loss(edge_1, edge_2, reduction='sum')
    # loos_2 = F.cross_entropy(edge_1, edge_2, reduction='sum')
    # loss_2 = cross_entropy(edge_1, edge_2)
    loss = F.sigmoid(torch.tensor(12.5-(epoch/200)*para)) * loss_1 + (1 - F.sigmoid(torch.tensor(12.5-(epoch/200)*para))) * loss_2
    # loss = line(epoch) * loos_1 + (1-line(epoch)) * loos_2

    return loss

class cross:
    def __init__(self, entropy_weight=[1, 1]):
        ew = torch.from_numpy(np.array(entropy_weight).astype(np.float32)).cuda()
        self.criterion2 = nn.CrossEntropyLoss(weight=ew)
    def __call__(self, outputs2, targets2):
        cross = self.criterion2(outputs2, torch.squeeze(targets2).long())
        return cross



if __name__ == '__main__':
    img1 = torch.rand(5,1,260,301)
    img2 = torch.rand(5,1,260,301)
    ssim_result = ssim(img1, img2)



    # epoch = 120
    # para = 10
    # a= F.sigmoid(torch.tensor(10 - (epoch / 200) * para))
    print('213')
