from func.utils import *
data = data11[0,0,:,:]
min = data.min().detach().cpu()
max = data.max().detach().cpu()
pain_openfwi_velocity_model((data.detach().cpu()-min)/(max-min),min,max)

size = len(data11[0,:,0,0])
data = data11[0,int(size/2-5),:,:]
min = data.min().detach().cpu()
max = data.max().detach().cpu()
pain_openfwi_velocity_model((data.detach().cpu()-min)/(max-min),min,max)

size = len(data11[0,:,0,0])
data = data11[0,int(size/2+5),:,:]
min = data.min().detach().cpu()
max = data.max().detach().cpu()
pain_openfwi_velocity_model((data.detach().cpu()-min)/(max-min),min,max)

data = data11[0,-1,:,:]
min = data.min().detach().cpu()
max = data.max().detach().cpu()
pain_openfwi_velocity_model((data.detach().cpu()-min)/(max-min),min,max)




data = data31[0,0,:,:]
min = data.min().detach().cpu()
max = data.max().detach().cpu()
pain_openfwi_velocity_model((data.detach().cpu()-min)/(max-min),min,max)

size = len(data31[0,:,0,0])
data = data31[0,int(size/2-5),:,:]
min = data.min().detach().cpu()
max = data.max().detach().cpu()
pain_openfwi_velocity_model((data.detach().cpu()-min)/(max-min),min,max)

size = len(data31[0,:,0,0])
data = data31[0,int(size/2+5),:,:]
min = data.min().detach().cpu()
max = data.max().detach().cpu()
pain_openfwi_velocity_model((data.detach().cpu()-min)/(max-min),min,max)

data = data31[0,-1,:,:]
min = data.min().detach().cpu()
max = data.max().detach().cpu()
pain_openfwi_velocity_model((data.detach().cpu()-min)/(max-min),min,max)



