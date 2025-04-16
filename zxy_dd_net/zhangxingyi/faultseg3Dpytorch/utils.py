import torch
#import keras
from torch.utils.data import Dataset
import os
import numpy as np
import random


def datanormalize(x):
    gm = np.mean(x)
    gs = np.std(x)
    g = x-gm
    g = g/gs
    return g

class MyDataset(Dataset):
    def __init__(self, dpth, fpth ,dimension, chann):
        self.dpth = dpth
        self.fpth = fpth
        dpth_list = os.listdir(self.dpth)
        self.dfile = [self.dpth + file for file in dpth_list]
        fpth_list = os.listdir(self.fpth)
        self.ffile = [self.fpth + file for file in fpth_list]
        self.dim = dimension
        self.chann = chann
        self.num_file = len(dpth_list)

    def __getitem__(self, item):
        a = 2 #data augumentation
        A = np.zeros((a,128,128,128),dtype=np.single)
        Y = np.zeros((a,128,128,128),dtype=np.single)
        
        gx = np.fromfile(self.dfile[item],dtype=np.single)
        fx = np.fromfile(self.ffile[item],dtype=np.single)

        gx = np.reshape(gx,(128,128,128))
        fx = np.reshape(fx,(128,128,128))
        gx = datanormalize(gx)
        fx = 2*np.clip(fx,0,1)
        gx = np.transpose(gx)
        fx = np.transpose(fx)

        #in seismic processing, the dimensions of a seismic array is often arranged as
        #a[n3][n2][n1] where n1 represnts the vertical dimenstion. This is why we need 
        #to transpose the array here in python 
        A[0,:,:,:] = gx
        Y[0,:,:,:] = fx
        r = random.randint(0,3)
        gx = np.rot90(gx,r,(2,1))
        fx = np.rot90(fx,r,(2,1))
        A[1,:,:,:] = gx
        Y[1,:,:,:] = fx
        return A,Y

    def __len__(self):
        return self.num_file


# if __name__ == '__main__':
#     dim = (128,128,128)
#     chann = 1
#     datas = MyDataset('G:/datas/Train/seis/','G:/datas/Train/channel/',dim,chann)
#     (a,b) = datas[4]
   