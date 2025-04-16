# -*- coding: utf-8 -*-
"""
Direct method for reading datasets

Created on Sep 2024

@author: Jian An (2569222191@qq.com)

"""
from Amunet.param_config import *
import scipy.io
import scipy
import numpy as np


def batch_read_matfile(dataset_dir,
                       start,
                       batch_length,
                       train_or_test="train",
                       data_channels=29):
    '''
    Batch read seismic gathers and velocity models for .mat file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :param data_channels:           The total number of channels read into the data itself
    :return:                        a quadruple: (seismic data, [velocity open_fwi_data, contour of velocity open_fwi_data])
                                    Among them, the dimensions of seismic data, velocity open_fwi_data and contour of velocity open_fwi_data are all (number of read data, channel, width x height)
    '''

    data_set = np.zeros([batch_length, data_channels, data_dim[0], data_dim[1]])
    label_set = np.zeros([batch_length, classes, model_dim[0], model_dim[1]])

    for indx, i in enumerate(range(start, start + batch_length)):

        # Load Seismic Data
        filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, i)
        if i%10==0:
            print("Reading: {}".format(filename_seis))
        sei_data = scipy.io.loadmat(filename_seis)["data"]
        # (400, 301, 29) -> (29, 400, 301)
        sei_data = sei_data.swapaxes(0, 2)
        sei_data = sei_data.swapaxes(1, 2)
        for ch in range(inchannels):
            data_set[indx, ch, ...] = sei_data[ch, ...]

        # Load Velocity Model
        filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, i)
        if i % 10 == 0:
            print("Reading: {}".format(filename_label))
        vm_data = scipy.io.loadmat(filename_label)["data"]
        label_set[indx, 0, ...] = vm_data

    return data_set, label_set


def batch_read_npyfile(dataset_dir,
                       start,
                       batch_length,
                       train_or_test="train"):
    '''
    Batch read seismic gathers and velocity models for .npy file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")

    :return:                        a pair: (seismic data, [velocity open_fwi_data, contour of velocity open_fwi_data])
                                    Among them, the dimensions of seismic data, velocity open_fwi_data and contour of velocity
                                    open_fwi_data are all (number of read data * 500, channel, height, width)
    '''

    dataset = []
    labelset = []

    for i in range(start, start + batch_length):
        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))
        temp_data = np.load(filename_seis)

        dataset.append(temp_data)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity open_fwi_data path in the dataset
        filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_label))
        temp_data = np.load(filename_label)

        labelset.append(temp_data)

    dataset = np.vstack(dataset)
    labelset = np.vstack(labelset)

    return dataset, labelset


def single_read_matfile(dataset_dir,
                        seismic_data_size,
                        velocity_model_size,
                        readID,
                        train_or_test="train",
                        data_channels=29):
    '''
    Single read seismic gathers and velocity models for .mat file

    :param dataset_dir:             Path to the dataset
    :param seismic_data_size:       Size of the seimic data
    :param velocity_model_size:     Size of the velocity open_fwi_data
    :param readID:                  The ID number of the selected data
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :param data_channels:           The total number of channels read into the data itself
    :return:                        a triplet: (seismic data, velocity open_fwi_data, contour of velocity open_fwi_data)
                                    Among them, the dimensions of seismic data, velocity open_fwi_data and contour of velocity open_fwi_data are
                                    (channel, width, height), (width, height) and (width, height) respectively
    '''
    filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, readID)
    print("Reading: {}".format(filename_seis))
    filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, readID)
    print("Reading: {}".format(filename_label))

    se_data = scipy.io.loadmat(filename_seis)
    se_data = np.float32(se_data["data"].reshape([seismic_data_size[0], seismic_data_size[1], data_channels]))
    vm_data = scipy.io.loadmat(filename_label)
    vm_data = np.float32(vm_data["data"].reshape(velocity_model_size[0], velocity_model_size[1]))

    # (400, 301, 29) -> (29, 400, 301)
    se_data = se_data.swapaxes(0, 2)
    se_data = se_data.swapaxes(1, 2)

    return se_data, vm_data


def single_read_npyfile(dataset_dir,
                        readIDs,
                        train_or_test="train"):
    '''
    Single read seismic gathers and velocity models for .npy file

    :param dataset_dir:             Path to the dataset
    :param readID:                  The IDs number of the selected data
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        seismic data, velocity open_fwi_data, contour of velocity open_fwi_data
    '''

    # Determine the seismic data path in the dataset
    filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_seis))
    # Determine the velocity open_fwi_data path in the dataset
    filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_label))

    se_data = np.load(filename_seis)[readIDs[1]]
    vm_data = np.load(filename_label)[readIDs[1]][0]

    return se_data, vm_data


if __name__ == '__main__':
    dataset_dir = r"/pytorch/pytor_selflearn/anjian/data/data_all/FlatVelA/"
    data, label = batch_read_npyfile(dataset_dir, 1, 2)

    path = r'/pytorch/pytor_selflearn/anjian/data/data_label/FCNVMB-data/data/train_data/SEGSaltData/vmodel_train/svmodel1.mat'
    se_data = scipy.io.loadmat(path)

    dataset_dir = r"/pytorch/pytor_selflearn/anjian/data/data_all/FlatVelA/"
    readIDs = [1, 200]
    data, label = single_read_npyfile(dataset_dir, readIDs)
    print('------')
