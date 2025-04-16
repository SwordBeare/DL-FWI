# -*- coding: utf-8 -*-
"""
Build helper method

Created on Sep 2024

@author: Jian An (2569222191@qq.com)

"""

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter
from torch.autograd import Variable

import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import scipy.io
import scipy

import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.pylab as plt

font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}
font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}


def pain_seg_seismic_data(para_seismic_data, is_colorbar=1):
    '''
    Plotting seismic data images of SEG salt datasets

    :param para_seismic_data:   Seismic data (400 x 301) (numpy)
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    '''

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(6.5, 8), dpi=120)
    else:
        fig, ax = plt.subplots(figsize=(6.2, 8), dpi=120)

    im = ax.imshow(para_seismic_data, extent=[0, 300, 400, 0], cmap=plt.cm.seismic, vmin=-(0.5-0.05*0.5), vmax=0.5)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)

    ax.set_xticks(np.linspace(0, 300, 5))
    ax.set_yticks(np.linspace(0, 400, 5))
    ax.set_xticklabels(labels=[0, 0.75, 1.5, 2.25, 3.0], size=21)
    ax.set_yticklabels(labels=[0.0, 0.50, 1.00, 1.50, 2.00], size=21)

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.99)
    else:
        plt.rcParams['font.size'] = 14  # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.32)
        cbar = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
        from matplotlib import ticker
        cbar.locator = ticker.MaxNLocator(nbins=6)

        plt.subplots_adjust(bottom=0.1, top=0.98, left=0.15, right=0.99)

    plt.show(block=True)


def pain_openfwi_seismic_data(para_seismic_data, is_colorbar=1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_seismic_data:   Seismic data (1000 x 70) (numpy)
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    '''

    # The size of 1000 x 70 is not easy to display, we compressed it to a similar size of 400 x 301 as the SEG dataset.
    data = cv2.resize(para_seismic_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)  #

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(6.5, 8), dpi=120)
    else:
        fig, ax = plt.subplots(figsize=(6.1, 8), dpi=120)

    im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-19, vmax=20)
    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)
    ax.set_xticks(np.linspace(0, 0.7, 5))
    ax.set_yticks(np.linspace(0, 1.0, 5))
    ax.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.5, top=0.95, left=0.20, right=0.99)
    else:
        plt.rcParams['font.size'] = 14  # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.3)
        cbar = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
        from matplotlib import ticker
        cbar.locator = ticker.MaxNLocator(nbins=7)

        plt.subplots_adjust(bottom=0.1, top=0.98, left=0.15, right=0.99)

    plt.show(block=True)

def plotcomparison(gt, pre):
    """
    plot current inverted model and ground truth
    """

    fig, axs = plt.subplots(1,2,figsize=(12, 5))

    im1 = axs[0].imshow(pre, vmin=np.min(pre), vmax=np.max(pre))
    fig.colorbar(im1,ax = axs[0])
    axs[0].set_title('predicted model')

    im2 = axs[1].imshow(gt, vmin=np.min(gt), vmax=np.max(gt))
    fig.colorbar(im2,ax = axs[1])
    axs[1].set_title('true model')
    plt.show(block=True)

def plotcomparison_singel(pre,vmax,vmin):
    """
    plot current inverted model and ground truth
    """
    pre = pre*(vmax-vmin)+vmin
    fig, axs = plt.subplots(1,1,figsize=(6, 5))

    im1 = axs.imshow(pre, vmin=np.min(pre), vmax=np.max(pre))
    cbar_ax = fig.add_axes([0.15, 0.85, 0.7, 0.02])  # [left, bottom, width, height]
    fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    # fig.colorbar(im1,ax = axs)
    axs.set_xlabel('Position(km)',fontsize=12)
    axs.set_ylabel('Depth(km)',fontsize=12)

    plt.show(block=True)

def plot_diceng1( predicted_vmod1, predicted_vmod2,predicted_vmod3, velocity_model,LOCATION):
    depth1 = np.linspace(0, 2000, 201)
    # 绘制地层速度图
    plt.figure(figsize=(5.5, 6))
    plt.plot(predicted_vmod1, depth1, label='FCNVMB', color='y')
    plt.plot(predicted_vmod2, depth1, label='DD-Net', color='b')
    plt.plot(predicted_vmod3, depth1, label='DC-Unet', color='r')
    plt.plot(velocity_model, depth1, label='Ground true', color='k')
    plt.gca().invert_yaxis()  # 反转y轴，使得深度从上到下
    plt.title('x = '+str(LOCATION)+'0 m',fontweight='bold')
    plt.xlabel('Velocity (m/s)',fontweight='bold')
    plt.ylabel('Depth (m)',fontweight='bold')
    # plt.grid()
    plt.legend()
    plt.show(block=True)

def plot_diceng2( predicted_vmod1, predicted_vmod2,predicted_vmod3, velocity_model,LOCATION):
    depth1 = np.linspace(0, 700, 70)
    # 绘制地层速度图
    plt.figure(figsize=(5, 6))
    plt.plot(predicted_vmod1, depth1, label='InversionNet', color='y')
    plt.plot(predicted_vmod2, depth1, label='DD-Net70', color='b')
    plt.plot(predicted_vmod3, depth1, label='DC-Unet70', color='r')
    plt.plot(velocity_model, depth1, label='Ground true', color='k')
    plt.gca().invert_yaxis()  # 反转y轴，使得深度从上到下
    plt.title('x = '+str(LOCATION)+'0 m',fontweight='bold')
    plt.xlabel('Velocity (m/s)',fontweight='bold')
    plt.ylabel('Depth (m)',fontweight='bold')
    # plt.grid()
    plt.legend()
    plt.show(block=True)
def plotcomparison_Compare(InversionNet,DDNet,DC_unet,True_model):
    """
    plot current inverted model and ground truth
    """


    if(True_model.shape[0]<71):

        max = np.max(True_model)
        min = np.min(True_model)
        Inversion_1 = True_model - (InversionNet * (max - min) + min)
        DDNet_1 = True_model - (DDNet * (max - min) + min)
        DC_unet_1 = True_model - (DC_unet * (max - min) + min)
        t = True_model - True_model

        pain_openfwi_velocity_model1(Inversion_1,-100,100)
        pain_openfwi_velocity_model1(DDNet_1,-100,100)
        pain_openfwi_velocity_model1(DC_unet_1,-100,100)
        pain_openfwi_velocity_model1(t,-100,100)

    else:
        Inversion_1 = True_model - InversionNet
        DDNet_1 = True_model - DDNet
        DC_unet_1 = True_model - DC_unet

        # pain_seg_velocity_model1(Inversion_1,-100,100)
        # pain_seg_velocity_model1(DDNet_1,-100,100)
        # pain_seg_velocity_model1(DC_unet_1,-100,100)
        pain_seg_velocity_model1(Inversion_1,np.min(Inversion_1),np.max(Inversion_1))
        pain_seg_velocity_model1(DDNet_1,np.min(DDNet_1),np.max(DDNet_1))
        pain_seg_velocity_model1(DC_unet_1,np.min(DC_unet_1),np.max(DC_unet_1))



    fig, axs = plt.subplots(1,4,figsize=(24, 5))

    im1 = axs[0].imshow(InversionNet, vmin=np.min(InversionNet), vmax=np.max(InversionNet))
    fig.colorbar(im1,ax = axs[0])
    axs[0].set_title('InversionNet')

    im2 = axs[1].imshow(DDNet, vmin=np.min(DDNet), vmax=np.max(DDNet))
    fig.colorbar(im2,ax = axs[1])
    axs[1].set_title('DDNet')

    im3 = axs[2].imshow(DC_unet, vmin=np.min(DC_unet), vmax=np.max(DC_unet))
    fig.colorbar(im3,ax = axs[2])
    axs[2].set_title('DC_unet')

    im4 = axs[3].imshow(True_model, vmin=np.min(True_model), vmax=np.max(True_model))
    fig.colorbar(im4, ax=axs[3])
    axs[3].set_title('True model')

    plt.show(block=True)








def pain_openfwi_velocity_model(para_velocity_model, min_velocity, max_velocity, is_colorbar=1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity open_fwi_data (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity open_fwi_data
    :param max_velocity:        Lower limit of velocity in the velocity open_fwi_data
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)
    para_velocity_model = para_velocity_model*(max_velocity-min_velocity)+min_velocity


    if is_colorbar == 0:
        im = ax.imshow(para_velocity_model)
        ax.set_xlabel('Position (km)', font18)
        ax.set_ylabel('Depth (km)', font18)
        plt.subplots_adjust(bottom=0.10, top=0.95, left=0.14, right=0.95)
    else:

        im = ax.imshow(para_velocity_model, extent=[0, 0.7, 0.7, 0])
        ax.set_xlabel('Position (km)', font18)
        ax.set_ylabel('Depth (km)', font18)
        ax.set_xticks(np.linspace(0, 0.7, 8))
        ax.set_yticks(np.linspace(0, 0.7, 8))
        ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
        ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
        plt.rcParams['font.size'] = 14  # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.35)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(min_velocity+100, max_velocity-100, 7), format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.14, right=0.95)

    plt.show(block=True)

def pain_openfwi_velocity_model1(para_velocity_model, min_velocity, max_velocity, is_colorbar = 1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)

    im = ax.imshow(para_velocity_model, extent=[0, 0.7, 0.7, 0], vmin=min_velocity, vmax=max_velocity)

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.95)
    else:
        plt.rcParams['font.size'] = 14      # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.35)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(min_velocity, max_velocity, 7), format = mpl.ticker.StrMethodFormatter('{x:.0f}'))
        plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)

    plt.show()

def pain_seg_velocity_model(para_velocity_model, min_velocity, max_velocity, is_colorbar=1):
    '''

    :param para_velocity_model: Velocity open_fwi_data (200 x 301) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity open_fwi_data
    :param max_velocity:        Lower limit of velocity in the velocity open_fwi_data
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''
    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(5.8, 4.3), dpi=150)
    para_velocity_model = para_velocity_model * (max_velocity - min_velocity) + min_velocity
    im = ax.imshow(para_velocity_model, extent=[0, 3, 2, 0])

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.tick_params(labelsize=14)

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.15, top=0.95, left=0.11, right=0.99)
    else:
        plt.rcParams['font.size'] = 14  # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.32)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(min_velocity, max_velocity, 7))
        plt.subplots_adjust(bottom=0.14, top=0.95, left=0.08, right=0.99)

    plt.show(block=True)

def pain_seg_velocity_model1(para_velocity_model, min_velocity, max_velocity, is_colorbar=1):
    '''

    :param para_velocity_model: Velocity open_fwi_data (200 x 301) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity open_fwi_data
    :param max_velocity:        Lower limit of velocity in the velocity open_fwi_data
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''
    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(5.8, 4.3), dpi=150)
    im = ax.imshow(para_velocity_model, extent=[0, 3, 2, 0])

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.tick_params(labelsize=14)

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.15, top=0.95, left=0.11, right=0.99)
    else:
        plt.rcParams['font.size'] = 14  # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.32)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(min_velocity, max_velocity, 7))
        plt.subplots_adjust(bottom=0.14, top=0.95, left=0.08, right=0.99)

    plt.show(block=True)
def add_gasuss_noise(image, mu=0, sigma=0.05):
    '''
    Add Gaussian noise

    :param image:       Input image
    :param mu:          Noise mean value
    :param sigma:       Noise variance
    :return:
    '''

    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise

    return gauss_noise


def agc_on_one_trace(data, window, length, min):
    '''
    Amplify the amplitude of a trace by automatic gain control

    :param data:    A trace of wave information
    :param window:  Window information
    :param length:  The width of the waveform, i.e. the length of a trace
    :param min:     Minimum value
    :return:
    '''

    window = math.floor(window / 2)
    if window < min:
        window = min
    w = np.ones(length)
    sumData = np.sum(np.abs(data[0: window * 2]))
    ave = sumData / (window * 2)
    if ave > 0.0001:
        for i in range(window):
            w[i] = 1.0 / ave
    ave = 0
    left = length - window * 2 - 1
    ave = np.sum(np.abs(data[left: length])) / (window * 2)
    if ave > 0.0001:
        for i in range(left, length):
            w[i] = 1.0 / ave

    for i in range(window - 1, 5, length - window):
        ave = sumData
        if i > window:
            for j in range(i - window - 5, i - window):
                ave = ave - abs(data[j])
            for k in range(i + window - 5, i + window):
                ave = ave + abs(data[k])
        sumData = ave
        ave = ave / (window * 2)
        if ave > 0.0001:
            w[i] = 1.0 / ave
            for j in range(0, 4):
                w[i + j] = w[i]
    data1 = data * w
    return data1, w


def magnify_amplitude_fortensor(para_image):
    '''
    Amplify the amplitude of the seismic data

    :param para_image:  Seismic data (tensor)
    :return:            Seismic data (tensor)
    '''
    image = para_image.numpy()
    width = image.shape[1]
    height = image.shape[0]

    mean_expand_ratio = 0
    for trace in range(width):
        wave_of_trace = image[:, trace]
        magnified_wave_of_trace, w = agc_on_one_trace(data=wave_of_trace, window=width, length=height, min=1)
        mean_expand_ratio += max(magnified_wave_of_trace) / max(wave_of_trace)
        image[:, trace] = magnified_wave_of_trace
    mean_expand_ratio /= width
    image /= mean_expand_ratio

    return torch.from_numpy(image)


def magnify_amplitude_fornumpy(para_image):
    '''
    Amplify the amplitude of the seismic data

    :param para_image:  Seismic data (numpy)
    :return:            Seismic data (numpy)
    '''
    image = para_image
    width = image.shape[1]
    height = image.shape[0]

    mean_expand_ratio = 0
    for trace in range(width):
        wave_of_trace = image[:, trace]
        magnified_wave_of_trace, w = agc_on_one_trace(data=wave_of_trace, window=width, length=height, min=1)
        mean_expand_ratio += max(magnified_wave_of_trace) / max(wave_of_trace)
        image[:, trace] = magnified_wave_of_trace
    mean_expand_ratio /= width
    image /= mean_expand_ratio

    return image


def extract_contours(para_image):
    '''
    Use Canny to extract contour features

    :param image:       Velocity open_fwi_data (numpy)
    :return:            Binary contour structure of the velocity open_fwi_data (numpy)
    '''

    image = para_image

    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image_to_255 = norm_image * 255
    norm_image_to_255 = norm_image_to_255.astype(np.uint8)
    canny = cv2.Canny(norm_image_to_255, 10, 15)
    bool_canny = np.clip(canny, 0, 1)
    return bool_canny

def extract_contours_patch(para_image):
    '''
    输入输出都是tensor数据
    :param para_image: tensor cpu
    :return: tensor cpu
    '''
    edge = para_image.numpy()
    batch = edge.shape[0]
    dim = edge.shape[2],edge.shape[3]

    # result = []
    result = np.zeros([batch, 1, dim[0], dim[1]])
    for i in range(batch):
        temp_edge = extract_contours(edge[i,0])
        result[i,0,...] = temp_edge
    result = torch.from_numpy(result)
    return result

def model_reader(net, device, save_src):
    '''
    Read the .pkl open_fwi_data into the program

    :param net:         Variables used to store the .pkl open_fwi_data (need to open up the space in advance)
    :param device:      Equipment environment
    :param save_src:    The path where the open_fwi_data to be read is located
    :return:            Returns the variable "net", but points to what is read into the open_fwi_data
    '''

    print("The external .pkl open_fwi_data is about to be imported")
    print("Read file: {}".format(save_src))
    model = torch.load(save_src)
    try:  # Attempt to do a network read
        net.load_state_dict(model)
    except RuntimeError:
        print("This open_fwi_data is obtained by multi-GPU training...")
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in model.items():
            name = k[7:]  # 7 is the length of the module
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)

    net = net.to(device)
    return net


def save_results(loss, epochs, save_path, xtitle, ytitle, title, is_show=False):
    '''
    Save the loss

    :param loss:        An array storing the loss value of each epoch
    :param epochs:      How many epochs does the current loss curve contain
    :param save_path:   Save path
    :param xtitle:      Description of the horizontal axis
    :param ytitle:      Description of the vertical axis
    :param title:       Title
    :param is_show:     Whether to display after saving
    '''

    fig, ax = plt.subplots()
    plt.plot(loss[1:], linewidth=2)
    ax.set_xlabel(xtitle, font18)
    ax.set_ylabel(ytitle, font18)
    ax.set_title(title, font21)
    ax.set_xticks([i for i in range(0, epochs + 1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, epochs + 1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(save_path + title, transparent=True)
    data = {}
    data['loss'] = loss
    scipy.io.savemat(save_path + '{}.mat'.format(title), data)
    if is_show == True:
        plt.show()
    plt.close()


def save_numpy(para_data, src_path, src_name):
    '''
    Save numpy data in .npy format.

    :param para_data:   The name of file
    :param src_path:    Save path
    :param src_name:    What name to save
    :return:
    '''
    print("Saving: {}".format(src_path + src_name))
    np.save(src_path + src_name, para_data)


def read_numpy(src_name, src_path):
    '''
    Read .npy files

    :param src_path:    Read path
    :param src_name:    The name of the file to read
    :return:
    '''
    print("Reading: {}".format(src_path + src_name))
    data = np.load(src_path + src_name, allow_pickle=True)
    return data


def run_mse(prediction, target):
    '''
    Evaluation metric: MSE

    :param prediction:  The velocity open_fwi_data predicted by the network
    :param target:      The ground truth
    :return:
    '''
    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    criterion = nn.MSELoss(reduction='mean')
    result = criterion(prediction, target)
    return result.item()


def run_mae(prediction, target):
    '''
    Evaluation metric: MAE

    :param prediction:  The velocity open_fwi_data predicted by the network
    :param target:      The ground truth
    :return:
    '''

    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    criterion = nn.L1Loss(reduction='mean')
    result = criterion(prediction, target)
    return result.item()


def run_R2(y_true, y_pred):
    '''

    :param y_true: np
    :param y_pred: np
    :return:
    '''
    y_mean = np.mean(y_true)

    sst = np.sum((y_true - y_mean) ** 2)

    ssr = np.sum((y_true - y_pred) ** 2)

    r2 = 1 - (ssr / sst)
    return r2


def _uqi_single(GT, P, ws):
    '''
    a component of UQI metric

    :param GT:          The ground truth
    :param P:           The velocity open_fwi_data predicted by the network
    :param ws:          Window size
    :return:
    '''
    N = ws ** 2

    GT_sq = GT * GT
    P_sq = P * P
    GT_P = GT * P

    GT_sum = uniform_filter(GT, ws)
    P_sum = uniform_filter(P, ws)
    GT_sq_sum = uniform_filter(GT_sq, ws)
    P_sq_sum = uniform_filter(P_sq, ws)
    GT_P_sum = uniform_filter(GT_P, ws)

    GT_P_sum_mul = GT_sum * P_sum
    GT_P_sum_sq_sum_mul = GT_sum * GT_sum + P_sum * P_sum
    numerator = 4 * (N * GT_P_sum - GT_P_sum_mul) * GT_P_sum_mul
    denominator1 = N * (GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
    denominator = denominator1 * GT_P_sum_sq_sum_mul

    q_map = np.ones(denominator.shape)
    index = np.logical_and((denominator1 == 0), (GT_P_sum_sq_sum_mul != 0))
    q_map[index] = 2 * GT_P_sum_mul[index] / GT_P_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_map[index] = numerator[index] / denominator[index]

    s = int(np.round(ws / 2))
    return np.mean(q_map[s:-s, s:-s])


def run_uqi(GT, P, ws=8):
    '''
    Evaluation metric: UQI

    :param P:       The velocity open_fwi_data predicted by the network
    :param GT:      The ground truth
    :param ws:      Size of window
    :return:
    '''
    if len(GT.shape) == 2:
        GT = GT[:, :, np.newaxis]
        P = P[:, :, np.newaxis]

    GT = GT.astype(np.float64)
    P = P.astype(np.float64)
    return np.mean([_uqi_single(GT[:, :, i], P[:, :, i], ws) for i in range(GT.shape[2])])


def run_lpips(GT, P, lp):
    '''
    Evaluation metric: LPIPS

    :param GT:      The ground truth
    :param P:       The velocity open_fwi_data predicted by the network
    :param lp:      LPIPS related objects
    :return:
    '''
    GT_tensor = torch.from_numpy(GT)
    P_tensor = torch.from_numpy(P)
    return lp.forward(GT_tensor, P_tensor).item()
def plotcougter(gt):
    """
    plot current inverted model and ground truth
    """

    fig, axs = plt.subplots(1,1,figsize=(6, 5))

    axs.imshow(gt, extent=[0, 0.7, 0.7, 0])
    # axs.imshow(gt, vmin=np.min(gt), vmax=np.max(gt))

    axs.set_xlabel('Position (km)', font18)
    axs.set_ylabel('Depth (km)', font18)
    axs.set_xticks(np.linspace(0, 0.7, 8))
    axs.set_yticks(np.linspace(0, 0.7, 8))
    axs.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    axs.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)

    plt.subplots_adjust(bottom=0.13, top=0.95, left=0.11, right=0.95)



    plt.show(block=True)

def plotcougter_counet(gt):
    """
    plot current inverted model and ground truth
    """

    fig, axs = plt.subplots(1,1,figsize=(6, 5))

    axs.imshow(gt, cmap='gray',extent=[0, 0.7, 0.7, 0])
    # axs.imshow(gt, vmin=np.min(gt), vmax=np.max(gt))

    axs.set_xlabel('Position (km)', font18)
    axs.set_ylabel('Depth (km)', font18)
    axs.set_xticks(np.linspace(0, 0.7, 8))
    axs.set_yticks(np.linspace(0, 0.7, 8))
    axs.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    axs.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)

    plt.subplots_adjust(bottom=0.13, top=0.95, left=0.11, right=0.95)



    plt.show(block=True)
    # from func.utils import plotcougter,plotcougter_counet
    # edge = extract_contours(velocity_model)
    # plotcougter_counet(edge)
    # plotcougter(velocity_model)
if __name__ == '__main__':

    img = np.random.rand(200,301)
    pain_seg_velocity_model(img,np.min(img),np.max(img))
    print('213')