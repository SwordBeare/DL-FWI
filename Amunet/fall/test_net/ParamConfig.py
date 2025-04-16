# -*- coding: utf-8 -*-


####################################################
####             MAIN PARAMETERS1               ####
####################################################

SimulateData = True  # If False denotes training the CNN with SEGSaltData
OpenFWIData= False # If False denotes training the CNN with OpenFWIData
ReUse = False  # If False always re-train a network
dh = 10  # Space interval
PainStart = 10 # Start index of training loss
open_data_type = 'FlatVelA'
# open_data_type = 'FlatFaultA'
# open_data_type = 'CurveVelA'
# open_data_type = 'CurveFaultA'
FineTuning = True
UpperBound1 = 1e-6
UpperBound2 = 1e-2
EEpochs = 200
####################################################
####             NETWORK PARAMETERS             ####
####################################################
if SimulateData:
    Epochs= 200      # Number of epoch
    # Epochs= 10      # Number of epoch
    TrainSize = 1600  # Number of training set
    # TrainSize = 10  # Number of training set
    TestSize = 100  # Number of testing set
    TestBatchSize = 10
    BatchSize = 10  # SEG
    Inchannels = 29  # Number of input channels, i.e. the number of shots
    DataDim = [400, 301]  # Dimension of original one-shot seismic data
    data_dsp_blk = (1, 1)  # Downsampling ratio of input
    ModelDim = [201, 301]  # Dimension of one velocity model
    label_dsp_blk = (1, 1)  # Downsampling ratio of output
    DownSamplingList = [32, 64, 128, 256, 512]
    DecoderList = [1024, 512, 256, 128, 64, 32]

elif OpenFWIData:
    Epochs = 200
    # TrainSize = 5000  # Number of training set
    # TrainSize = 1000  # Number of training set
    TrainSize = 24000  # Number of training set
    # TestSize = 1000  # Number of testing set
    TestSize = 6000  # Number of testing set
    TestBatchSize = 50
    BatchSize = 50  # OpenFWI
    Inchannels = 5
    DataDim = [1000, 70]
    data_dsp_blk = (1, 1)
    ModelDim = [70, 70]
    label_dsp_blk = (1, 1)
    DownSamplingList = [64, 128, 256]
    DecoderList = [512, 256, 128, 64]

else:
    Epochs = 50
    TrainSize = 130
    # TrainSize = 20
    # TrainSize = 20
    TestSize = 10
    TestBatchSize = 1
    BatchSize = 10  # SEG
    Inchannels = 29
    DataDim = [400, 301]
    data_dsp_blk = (1, 1)
    ModelDim = [201, 301]
    label_dsp_blk = (1, 1)
    DownSamplingList = [32, 64, 128, 256, 512]
    DecoderList = [1024, 512, 256, 128, 64, 32]


####################################################
####             MAIN PARAMETERS2               ####
####################################################
LearnRate = 1e-3  # Learning rate
Nclasses = 1  # Number of output channels 输出的通道数
SaveEpoch = 20
DisplayStep = 20  # Number of steps till outputting stats
PatchSize = [2, 2]
HiddenSize = PatchSize[0] * PatchSize[1] * DownSamplingList[-1]

####################################################
####             TESTING INSTANCES              ####
####################################################

# ********************************Test for SEGSimulate ******
# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test01/'
# SaveEpoch = 20
# TrainSize = 1600
# Epochs = 300

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test02/'
# SaveEpoch = 20
# TrainSize = 1600
# Epochs = 200
# PainStart = 100

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test03/'
# SaveEpoch = 10
# TrainSize = 1600
# Epochs = 200
# PainStart = 100

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test04/'
# SaveEpoch = 10
# TrainSize = 1600
# Epochs = 200
# PainStart = 100

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test05/'
# SaveEpoch = 10
# TrainSize = 1600
# Epochs = 200
# PainStart = 100
# ********************************Test for SEGSal ******

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test01/'
# Epochs = 500
# PainStart = 200
#
# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test02/'
# Epochs = 200
# # EEpochs = 200
# LearnRate = 1e-3
# LearnRate2 = 1e-3


MLPDim = 4 * 128
NumHeads = 8
NumTransformerLayer = 4
Test_order = 'Test03/'
Epochs = 200
# EEpochs = 200
LearnRate = 1e-3
LearnRate2 = 1e-3
# ********************************Test for OPENFWI ******
# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test01/'
# Epochs = 200
# PainStart = 100
# SaveEpoch = 20
# open_data_type = "CurveVelA"
# TrainSize = 5000


# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test02/'
# Epochs = 200
# PainStart = 100
# SaveEpoch = 20
# open_data_type = "CurveVelA"
# TrainSize = 24000

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test03/'
# Epochs = 200
# PainStart = 100
# SaveEpoch = 20
# open_data_type = "CurveVelA"
# TrainSize = 5000

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test04/'
# Epochs = 200
# PainStart = 100
# SaveEpoch = 20
# open_data_type = "CurveVelA"
# TrainSize = 5000

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test05/'
# Epochs = 300
# PainStart = 100
# SaveEpoch = 20
# open_data_type = "CurveVelA"
# TrainSize = 24000

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test01/'
# Epochs = 300
# PainStart = 100
# SaveEpoch = 20
# open_data_type = "FlatVelA"
# TrainSize = 24000

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test01/'
# Epochs = 300
# PainStart = 100
# SaveEpoch = 20
# open_data_type = "CurveVelB"
# TrainSize = 24000

# MLPDim = 4 * 128
# NumHeads = 8
# NumTransformerLayer = 4
# Test_order = 'Test01/'
# Epochs = 300
# PainStart = 100
# SaveEpoch = 20
# open_data_type = "FlatVelB"
# TrainSize = 24000