# -*- coding: utf-8 -*-
"""
Parameters setting

Created on Sep 2024

@author: Jian An (2569222191@qq.com)

"""

####################################################
####             MAIN PARAMETERS                ####
####################################################

# Existing datasets: SEGSalt|SEGSimulation|FlatVelA|CurveFaultA|FlatFaultA|CurveVelA)
# dataset_name = 'SEGSimulation'
dataset_name = 'CurveVelB'
learning_rate = 0.001                                   # Learning rate
classes = 1                                             # Number of output channels
display_step = 5                                        # Number of training sessions required to print a "loss"
gpus = [0]
# key_word = 'all_mse_mse_16'
key_word = 'para_12.5_15'

# add_gasuss_noise
is_gasuss_noise = False
####################################################
####            DATASET PARAMETERS              ####
####################################################
# openfiw:50 seg:10
if dataset_name == 'SEGSimulation':
    data_dim = [400, 301]                               # Dimension of original one-shot seismic data
    model_dim = [201, 301]                              # Dimension of one velocity open_fwi_data
    inchannels = 29                                     # Number of input channels
    train_size = 1600                                   # Number of training sets
    test_size = 100                                     # Number of testing sets

    firststage_epochs = 200
    secondstage_epochs = 0
    thirdstage_epochs = 0
    loss_weight = [1, 1e6]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 10                               # Number of batches fed in network in one training epoch.
    test_batch_size = 10

elif dataset_name == 'SEGSalt':
    data_dim = [400, 301]
    model_dim = [201, 301]
    inchannels = 29
    train_size = 130
    test_size = 10

    firststage_epochs = 200
    secondstage_epochs = 0
    thirdstage_epochs = 0                              # SEGSalt for transfer learning and does not require curriculum tasks
    loss_weight = [1, 1e6]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 13
    test_batch_size = 10

elif dataset_name == 'FlatVelA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    test_size = 6000

    # Course learning parameters
    firststage_epochs = 200
    secondstage_epochs = 0
    thirdstage_epochs = 0
    loss_weight = [1, 0.01]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 50
    test_batch_size = 5

elif dataset_name == 'CurveVelA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    test_size = 6000

    firststage_epochs = 200
    secondstage_epochs = 0
    thirdstage_epochs = 0
    loss_weight = [1, 0.1]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 50
    test_batch_size = 5

elif dataset_name == 'FlatFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 4800
    test_size = 6000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 100
    loss_weight = [1, 0.01]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 50
    test_batch_size = 5

elif dataset_name == 'CurveFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 5000
    test_size = 6000

    firststage_epochs = 100
    secondstage_epochs = 0
    thirdstage_epochs = 0
    loss_weight = [1, 0.1]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 50
    test_batch_size = 5
elif dataset_name == 'FlatVelB':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    test_size = 6000

    # Course learning parameters
    firststage_epochs = 200
    secondstage_epochs = 0
    thirdstage_epochs = 0
    loss_weight = [1, 0.01]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 50
    test_batch_size = 5
elif dataset_name == 'CurveVelB':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    test_size = 6000

    firststage_epochs = 200
    secondstage_epochs = 0
    thirdstage_epochs = 0
    loss_weight = [1, 0.1]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 50
    test_batch_size = 5
else:
    print('The selected dataset is invalid')
    exit(0)
