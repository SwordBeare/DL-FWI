import sys
import numpy as np
#from open_fwi_data import UNet
from torch.utils.data import DataLoader
from att import UNet
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def dataProcess(sx):
    std = np.std(sx)
    mea = np.mean(sx)
    sx = (sx-mea)/std
    return sx

def predict(model,inputs):
    inputs = inputs[np.newaxis,np.newaxis,:,:,:] #np.newaxis的作用是增加一个维度
    inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
    outputs = model(inputs).cpu().detach().numpy()
    return outputs+1

def predictGPU(model,inputs):
    inputs = inputs[np.newaxis,np.newaxis,:,:,:]
    print(inputs.shape)
    inputs = torch.from_numpy(inputs).type(torch.FloatTensor).to(cudaName)
    outputs = model(inputs).cpu().detach().numpy()
    return outputs+1


filename = './data/train/sx/101.dat'
label ='./data/train/ft/101.dat'
n1,n2,n3 = 128, 128, 128
v3d = np.fromfile(filename,dtype=np.single).reshape((n1,n2,n3))

v3d=dataProcess(v3d)
v3d = np.transpose(v3d)
#v3d=np.reshape(v3d,(128,384,512))


v3dlabel=np.fromfile(label,dtype=np.single).reshape((n1,n2,n3))
v3dlabel= np.transpose(v3dlabel)
#plt.imshow(v3d[:,10,:],cmap='gray')
#plt.show()



'''
gpu_num = 4

cudaName = torch.device("cuda:%d" % gpu_num if torch.cuda.is_available() else "cpu")
'''
cudaName = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpth = 'valid'
is_loaded = True

res = np.zeros(v3d.shape)
#v3d = v3d[:128,129:257,129:257]
#v3d = v3d[:128,:128,:128]
#open_fwi_data = UNet(in_dim=1, out_dim=1, num_filters=32).to(cudaName)
model = UNet(in_channels=1, out_channels=1).to(cudaName)
#loaded_file = './check/model_min_%s.pth' % cpth
loaded_file = './check/model006.pth'

if is_loaded:
    #open_fwi_data.load_state_dict(torch.load(loaded_file, map_location={'cuda:3': 'cuda:%d' % gpu_num})['net'])
    m = torch.load(loaded_file)['net']
    try:  # 尝试进行网络读取
        model.load_state_dict(m)
    except RuntimeError:  # 如果出现错误, 有可能是因为模型是通过多卡训练得到的, 保存key值存在差异, 可通过修改弥补
        print("数据是多卡训练得到, 因此重新读取")
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in m.items():
            name = k[7:]  # 前7个正好去掉了module.
            new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值

        model.load_state_dict(new_state_dict)

    model = model.to(cudaName)

    res = predictGPU(model, v3d)[0, 0, :, :, :]
    #res=np.transpose(res)

# if torch.cuda.is_available():
#     open_fwi_data = torch.nn.DataParallel(open_fwi_data, device_ids=[0,1,2]).cuda()    # 申请多GPU



#plt.imshow(res[:, 10, :], cmap='jet') #jet 蓝-青-黄-红
m=10

fig = plt.figure(figsize=(12,12))
plt.subplot(1, 3, 1)
imgplot1 = plt.imshow(v3d[:,29,:], cmap='gray')
plt.subplot(1, 3, 2)
imgplot2 = plt.imshow(res[:,29,:],cmap=plt.cm.bone,interpolation='nearest',aspect=1) #jet 蓝-青-黄-红
plt.subplot(1, 3, 3)
imgplot3 = plt.imshow(v3dlabel[:,29,:],cmap=plt.cm.bone,interpolation='nearest',aspect=1)
plt.show()
