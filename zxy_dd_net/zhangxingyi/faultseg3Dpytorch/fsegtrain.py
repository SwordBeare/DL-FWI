#from open_fwi_data import UNet
from att import UNet
from utils import MyDataset
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from Dice import DiceLoss


# cuda:3 --- 10.26 用leo环境
cudaName = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log(fname, s):
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()

def imgReshape(img):
    newNum = img.shape[0]*img.shape[1]
    Nchann = 1
    Ndim = img.shape[2]
    return img.reshape((newNum,Nchann,128,128,Ndim))

def labelReshape(img):
    newNum = img.shape[0]*img.shape[1]
    Nchann = 1
    Ndim = img.shape[2]
    return img.reshape((newNum,Nchann,128,128,Ndim))

def train(model, criterion, optimizer, epochs):
    writer1 = SummaryWriter(log_dir='log/train',comment='Train_loss')
    writer2 = SummaryWriter(log_dir='log/valid',comment='Valid_loss')
    MinTrainLoss = 999
    MinValidLoss = 999
    MinMixedLoss = 999
    train_loss = []
    valid_loss = []

    for epoch in range(epochs):
        total_train_loss = []
        model.train()
        for i, (img,labels) in enumerate(tqdm(train_loader),0):

            img = imgReshape(img)
            labels = labelReshape(labels)
            inputs = img.type(torch.FloatTensor)
            labels = labels.float()

            if CUDA:
                inputs,labels = inputs.to(cudaName),labels.to(cudaName)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss.append(loss.item())
            running_loss = loss.item()
            if (i+1)%100==0:
                print('Epoch '+str(epoch+1)+' : train_LOSS ='+str(running_loss))
        train_loss.append(np.mean(total_train_loss))

        total_valid_loss = []
        model.eval()
        for i, (img,labels) in enumerate(valid_loader,0):
            if (i+1)%100==0:
                print("Epoch %d, batch %d, valid"%(epoch+1,i+1))
            img = imgReshape(img)
            labels = labelReshape(labels)
            inputs = img.type(torch.FloatTensor)
            labels = labels.float()
            
            if CUDA:
                inputs,labels = inputs.to(cudaName),labels.to(cudaName)
            with torch.no_grad():
                outputs = model(inputs)
            
            loss = criterion(outputs,labels)
            total_valid_loss.append(loss.item())
            running_loss = loss.item()
            if (i+1)%100==0:
                print('Epoch '+str(epoch+1)+' : valid_LOSS ='+str(running_loss))
        valid_loss.append(np.mean(total_valid_loss))
        
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}').format((epoch + 1), epochs,train_loss[-1],valid_loss[-1])

        print(log_string)
        log('./DeepLab.log',log_string)


        writer1.add_scalar('train_loss',train_loss[-1],epoch)
        writer2.add_scalar('valid_loss',valid_loss[-1],epoch)
        
        if train_loss[-1]<MinTrainLoss:
            # file_path = './check/open_fwi_data%03d.pth'%(epoch+1)
            print('The latest min train: %d %f'%(epoch+1,train_loss[-1]))
            file_path = './check/model_min_train.pth'
            state = {'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
            torch.save(state,file_path)
            MinTrainLoss = train_loss[-1]
        if valid_loss[-1]<MinValidLoss:
            print('The latest min valid: %d %f'%(epoch+1,valid_loss[-1]))
            file_path = './check/model_min_valid.pth'
            state = {'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
            torch.save(state,file_path)
            MinValidLoss = valid_loss[-1]

        mixed_loss = train_loss[-1]+valid_loss[-1]
        if mixed_loss<MinMixedLoss:
            print('The latest min mixed: %d %f'%(epoch+1,mixed_loss))
            file_path = './check/model_min_mixed.pth'
            state = {'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
            torch.save(state,file_path)
            MinMixedLoss = mixed_loss

        if (epoch+1)%1==0:
             file_path = './check/open_fwi_data%03d.pth'%(epoch+1)
             state = {'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
             torch.save(state,file_path)

    writer1.close()
    writer2.close()


    plt.figure(figsize=(12,4))
    plt.title(" Loss During Training") 

    plt.plot(train_loss,label="train_loss")
    plt.plot(valid_loss,label="valid_loss")
    plt.ylabel('Loss',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)  
    plt.legend(['train', 'valid'], loc='center right',fontsize=20) 
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.show()
    plt.savefig('history for loss.png')
    plt.close()




if __name__ == '__main__':
    n1, n2, n3 = 128, 128, 128
    params = {'batch_size': 1,
              'dim': (n1, n2, n3),
              'n_channels': 1,
              'shuffle': True}

    prefix = './data/'

    trainfix = prefix+'train/'
    validfix = prefix+'valid/'


    tdpath = trainfix+'sx/'
    tfpath = trainfix+'ft/'
    

    vdpath = validfix+'sx/'
    vfpath = validfix+'ft/'

    print("Start loading")

    batchSiz = 1

    
    training_generator = MyDataset(dpth=tdpath, fpth=tfpath, dimension=(n1,n2,n3), chann=1)
    validate_generator = MyDataset(dpth=vdpath, fpth=vfpath, dimension=(n1,n2,n3), chann=1)
    

    train_loader = DataLoader(dataset=training_generator, batch_size=batchSiz, shuffle=False,drop_last=True)
    valid_loader = DataLoader(dataset=validate_generator, batch_size=batchSiz, shuffle=False,drop_last=True)
    print("Finish loading")

    num_classes = 1  # Number of classes. (= number of output channel)
    input_channels = 1  # Number of input channel
    resnet = 'resnet18_os16'  # Base resnet architecture ('resnet18_os16', 'resnet34_os16', 'resnet50_os16', 'resnet101_os16', 'resnet152_os16', 'resnet18_os8', 'resnet34_os18')
    last_activation = 'sigmoid'  # 'softmax', 'sigmoid' or None

    is_loaded = False
    loaded_file = './check0303/model020.pth'
    
    model = UNet(in_channels=1, out_channels=1)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()  # 申请多GPU
    if is_loaded :
        model.load_state_dict(torch.load(loaded_file)['net'])
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.to(cudaName)
    else:
        model = model
    
    #criterion = DiceLoss()
    #weight_CE = torch.FloatTensor([2,48]).to(cudaName)
    #criterion = torch.nn.CrossEntropyLoss()#(weight=weight_CE)
    criterion = torch.nn.BCEWithLogitsLoss()
    #torch.backends.cudnn.enabled = True
    #criterion = torch.nn.BCELoss()
    #criterion = torch.nn.MSELoss()

    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if is_loaded :
        optimizer.load_state_dict(torch.load(loaded_file)['optimizer'])
    train(model,criterion,optimizer,epochs=10)

    
