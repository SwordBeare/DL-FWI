# -*- coding: utf-8 -*-
"""
Curriculum Learning

Created on Sep 2024

@author: Jian An (2569222191@qq.com)

"""
from path_config import *
from param_config import *
from func.utils import model_reader, add_gasuss_noise, magnify_amplitude_fornumpy, extract_contours_patch,plotcougter
from func.datasets_reader import batch_read_matfile, batch_read_npyfile
from func.ssim import ssim
from func.loss import loss_function

from net.InversionNet import InversionNet

from net.FCNVMB import FCNVMB

from net.DC_unet import DC_unet
from net.D_unet import D_unet
from net.DC_unet70 import DC_unet70
from net.D_unet70 import D_unet70
from net.C_unet70 import C_unet70

from net.DDNet70 import DDNet70Model
from net.DDNet import DDNetModel

from func.utils import *

from math import ceil
import time
import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn.functional as F


def determine_network(external_model_src="", model_type='FCNVMB'):
    '''
    Request a network object and import an external network, or create an initialized network

    :param external_model_src:  External pkl file path
    :param model_type:          The main open_fwi_data used, this open_fwi_data is differentiated based on different papers.
                                The available key open_fwi_data keywords are
                                [DDNet | DDNet70 | InversionNet | FCNVMB | SDNet | SDNet70]
    :return:                    A triplet: open_fwi_data object, GPU environment object and optimizer
    '''

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    # Network initialization
    if model_type == "InversionNet":
        net_model = InversionNet()
    elif model_type == "FCNVMB":
        net_model = FCNVMB(n_classes=classes,
                           in_channels=inchannels,
                           is_deconv=True,
                           is_batchnorm=True)
    elif model_type == 'DC_unet':
        net_model = DC_unet(n_classes=classes,
                            in_channels=inchannels,
                            is_deconv=True,
                            is_batchnorm=True)
    elif model_type == 'D_unet':
        net_model = D_unet(n_classes=classes,
                            in_channels=inchannels,
                            is_deconv=True,
                            is_batchnorm=True)
    elif model_type == "DC_unet70":
        net_model = DC_unet70(n_classes=classes,
                              in_channels=inchannels,
                              is_deconv=True,
                              is_batchnorm=True)
    elif model_type == "D_unet70":
        net_model = D_unet70(n_classes=classes,
                             in_channels=inchannels,
                             is_deconv=True,
                             is_batchnorm=True)
    elif model_type == "C_unet70":
        net_model = C_unet70(n_classes=classes,
                             in_channels=inchannels,
                             is_deconv=True,
                             is_batchnorm=True)
    elif model_type == "DDNet":
        net_model = DDNetModel(n_classes=classes,
                             in_channels=inchannels,
                             is_deconv=True,
                             is_batchnorm=True)

    elif model_type == "DDNet70":
        net_model = DDNet70Model(n_classes=classes,
                             in_channels=inchannels,
                             is_deconv=True,
                             is_batchnorm=True)
    else:
        print(
            'The "model_type" parameter selected in the determine_network(...)'
            ' is the undefined network open_fwi_data keyword! Please check!')
        exit(0)

    # Inherit the previous network structure
    if external_model_src != "":
        net_model = model_reader(net=net_model, device=device, save_src=external_model_src)

    # Allocate GPUs and set optimizers
    if torch.cuda.is_available():
        net_model = torch.nn.DataParallel(net_model.cuda(), device_ids=gpus)

    optimizer = torch.optim.Adam(net_model.parameters(), lr=learning_rate)

    return net_model, device, optimizer


def load_dataset(stage=3):
    '''
    Load the training data according to the parameters in "param_config"

    :return:
    '''

    print("---------------------------------")
    print("· Loading the datasets...")

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        data_set, label_sets = batch_read_matfile(data_dir, 1, train_size, "train")
    else:
        # openFWI 需要归一化 但seg不需要
        data_set, label_sets = batch_read_npyfile(data_dir, 1, ceil(train_size / 500), "train")
        for i in range(data_set.shape[0]):
            vm = label_sets[i][0]
            # standardization:
            label_sets[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        middle_shot_id = 15
        first_p = 9
        second_p = 18
    else:
        middle_shot_id = 2
        first_p = 2
        second_p = 4

    if stage == 1:
        # Change the data to:
        # 0---first_p is the increased noise data,
        # first_p---second_p is the amplitude expansion data,
        # and second_p---inchannels is the middle channel data
        for eachData in range(train_size):
            middle_shot = data_set[eachData, middle_shot_id, :, :].copy()
            middle_shot_with_noise = add_gasuss_noise(middle_shot.copy())
            middle_shot_magnified = magnify_amplitude_fornumpy(middle_shot.copy())
            for j in range(second_p, inchannels):
                data_set[eachData, j, :, :] = middle_shot
            for j in range(first_p, second_p):
                data_set[eachData, j, :, :] = middle_shot_magnified
            for j in range(0, first_p):
                data_set[eachData, j, :, :] = middle_shot_with_noise
    elif stage == 2:
        # Change all data to intermediate data
        for eachBatch in range(train_size):
            middle_shot = data_set[eachBatch, middle_shot_id, :, :].copy()
            for eachChannel in range(inchannels):
                data_set[eachBatch, eachChannel, :, :] = middle_shot
    else:
        pass

    # Training set
    seis_and_vm = data_utils.TensorDataset(
        torch.from_numpy(data_set).float(),
        torch.from_numpy(label_sets).float())
    seis_and_vm_loader = data_utils.DataLoader(
        seis_and_vm,
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True)

    print("· Number of seismic gathers included in the training set: {}".format(train_size))
    print("· Dimensions of seismic data: ({},{},{},{})".format(train_size, inchannels, data_dim[0], data_dim[1]))
    print(
        "· Dimensions of velocity open_fwi_data: ({},{},{},{})".format(train_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    return seis_and_vm_loader, data_set, label_sets


def train_for_one_stage(cur_epochs, model, training_loader, optimizer, save_times=1,
                        model_type="DC_unet"):
    '''
    Training for designated epochs

    :param cur_epochs:      Designated epochs
    :param model:           Network open_fwi_data objects to be used for training
    :param training_loader: Trainin dataset loader to be fed into the network
    :param optimizer:       Optimizer
    :param save_times:       The number of times the trained modes is saved in training

    :param key_word:        After the training, the keywords will be saved to the open_fwi_data
    :param stage_keyword:   The selected difficulty keyword (set "no settings" to ignore CL)
    :param model_type:      The main open_fwi_data used, this open_fwi_data is differentiated based on different papers.
                            The available key open_fwi_data keywords are [DDNet | DDNet70 | InversionNet | FCNVMB]
    :return:                Model save path
    '''

    loss_of_stage = []
    last_model_save_path = ""
    step = int(train_size / train_batch_size)  # Total number of batches to train
    save_epoch = cur_epochs // save_times
    training_time = 0

    model_save_name = "{}_{}_TrSize{}_AllEpo{}".format(dataset_name, key_word, train_size, cur_epochs)

    for epoch in range(cur_epochs):
        # Training for the current epoch
        loss_of_epoch = 0.0
        cur_node_time = time.time()
        ############
        # training #
        ############
        for i, (images, labels) in enumerate(training_loader):
            # Number of iteration trainings
            iteration = epoch * step + i + 1

            # a = images[4][15].numpy()
            # pain_seg_seismic_data(a)
            # pain_openfwi_seismic_data(a)

            # Set to training mode
            model.train()

            # Load to GPU
            if torch.cuda.is_available():
                # It does not wait for the data to be fully copied before performing subsequent operations.
                edge = extract_contours_patch(labels)
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                edge = edge.cuda(non_blocking=True)
                # contours_labels = contours_labels.cuda(non_blocking=True)

            # Gradient cache clearing
            optimizer.zero_grad()
            # criterion = LossDDNet(weights=loss_weight)

            if model_type in ['InversionNet']:
                output = model(images)
                loss = F.mse_loss(output, labels, reduction='sum') / (model_dim[0] * model_dim[1] * train_batch_size)
            elif model_type in ['FCNVMB']:
                output = model(images, model_dim)
                loss = F.mse_loss(output, labels, reduction='sum') / (model_dim[0] * model_dim[1] * train_batch_size)
            elif model_type in ['DC_unet', 'DC_unet70', 'D_unet70','C_unet70','D_unet']:
                output = model(images)
                # loss = F.mse_loss(output, labels, reduction='sum') / (model_dim[0] * model_dim[1] * train_batch_size)
                loss = loss_function(output, labels, epoch) / (model_dim[0] * model_dim[1] * train_batch_size)

            else:
                print(
                    'The "model_type" parameter selected in the train_for_one_stage(...)'
                    ' is the undefined network open_fwi_data keyword! Please check!')
                exit(0)

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            # Loss backward propagation
            # The loss value of a single pixel
            loss.backward()

            # Optimize
            optimizer.step()

            loss_of_epoch += loss.item()

            if iteration % display_step == 0:
                print('[{}] Epochs: {}/{}, Iteration: {}/{}, index: {}/{}--- Training Loss:{:.6f}'
                      .format(key_word, epoch + 1, cur_epochs, iteration, step * cur_epochs, i + 1,
                              len(training_loader),
                              loss.item()))

        ################################
        # The end of the current epoch #
        ################################
        if (epoch + 1) % 1 == 0:
            # Calculate the average loss of the current epoch
            print('[{}] Epochs: {:d} finished ! Training loss: {:.5f}'
                  .format(key_word, epoch + 1, loss_of_epoch / i))

            # Include the average loss in the array belonging to the current stage
            loss_of_stage.append(loss_of_epoch / i)

            # Statistics of the time spent in a epoch
            time_elapsed = time.time() - cur_node_time
            print('[{}] Epochs consuming time: {:.0f}m {:.0f}s'
                  .format(key_word, time_elapsed // 60, time_elapsed % 60))
            training_time += time_elapsed
        #########################################################################
        # When it reaches the point where intermediate results can be stored... #
        #########################################################################
        if (epoch + 1) % save_epoch == 0:
            last_model_save_path = models_dir + model_save_name + '_CurEpo' + str(epoch + 1) + "_" + model_type + '.pkl'
            last_model_save_path = r'' + last_model_save_path
            torch.save(model.state_dict(), last_model_save_path)
            print('[' + key_word + '] Trained open_fwi_data saved: %d percent completed' % int(
                (epoch + 1) * 100 / cur_epochs))
        last_result_Loss_save_path = r'' + results_dir + "[Loss]" + model_save_name + "_" + model_type + ".npy"
        np.save(last_result_Loss_save_path, np.array(loss_of_stage))

    return last_model_save_path, training_time


def load_dataset2():
    seis_path = r'E:\DATA\trainData_1600groups_29channels_400x301_georecData.npy'
    velo_path = r'E:\DATA\trainData_1600groups_1channel_201x301_vmodelData.npy'
    data_set = np.load(seis_path).reshape(1600, 29, 400, 301)
    label_sets = np.load(velo_path)[0].reshape(1600, 1, 201, 301)

    seis_and_vm = data_utils.TensorDataset(
        torch.from_numpy(data_set).float(),
        torch.from_numpy(label_sets).float())
    seis_and_vm_loader = data_utils.DataLoader(
        seis_and_vm,
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True
    )
    return seis_and_vm_loader, 1, 2


if __name__ == "__main__":
    path = ''
    # path = r'D:\Allresult\models\SEGSimulationmodel\SEGSimulation_parm15_TrSize1600_AllEpo200_CurEpo100_FCNVMB.pkl'
    net_model, device, optimizer = determine_network(path, model_type='DC_unet70')

    # for name, param in net_model.named_parameters():
    #     if 'up1' not in name and 'final' not in name:
    #         param.requires_grad = False

    train_loader, _, _ = load_dataset()
    last_model_save_path, training_time = train_for_one_stage(epochs, net_model,
                                                              train_loader, optimizer,
                                                              save_times=20,
                                                              model_type='DC_unet70')
    print('-----------------------')







