# -*- coding: utf-8 -*-
"""
Test the open_fwi_data effect after training.

Created on Sep 2024

@author: Jian An (2569222191@qq.com)

"""
from path_config import *
from func.utils import *
# from func.utils import run_mse, run_mae, run_lpips, run_uqi, run_R2, pain_seg_seismic_data, pain_seg_velocity_model, \
#     pain_openfwi_velocity_model, pain_openfwi_seismic_data, plotcomparison, extract_contours, plotcomparison_Compare, \
#     plotcomparison_singel,add_gasuss_noise
from func.datasets_reader import batch_read_matfile, batch_read_npyfile, single_read_matfile, single_read_npyfile
from model_train import determine_network

from func.ssim import ssim
import time
import lpips
import numpy as np
import torch
import torch.utils.data as data_utils
from scipy.io import loadmat
import matplotlib

matplotlib.use('TkAgg')


def load_dataset():
    '''
    Load the testing data according to the parameters in "param_config"

    :return:    A triplet: datasets loader, seismic gathers and velocity models
    '''

    print("---------------------------------")
    print("· Loading the datasets...")
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        if dataset_name == 'SEGSimulation':
            data_set, label_sets = batch_read_matfile(data_dir, 1601, test_size, "test")
        else:
            data_set, label_sets = batch_read_matfile(data_dir, 1, test_size, "test")
    else:
        data_set, label_sets = batch_read_npyfile(data_dir, 1, test_size // 500, "test")
        for i in range(data_set.shape[0]):
            vm = label_sets[i][0]
            max_velocity, min_velocity = np.max(vm), np.min(vm)
            label_sets[i][0] = (vm - min_velocity) / (max_velocity - min_velocity)

    print("· Number of seismic gathers included in the testing set: {}.".format(test_size))
    print("· Dimensions of seismic data: ({},{},{},{}).".format(test_size, inchannels, data_dim[0], data_dim[1]))
    print(
        "· Dimensions of velocity open_fwi_data: ({},{},{},{}).".format(test_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(),
                                           torch.from_numpy(label_sets).float())
    seis_and_vm_loader = data_utils.DataLoader(seis_and_vm, batch_size=test_batch_size, shuffle=True)

    return seis_and_vm_loader, data_set, label_sets


def batch_test(model_path, model_type="DC_unet"):
    '''
    Batch testing for multiple seismic data

    :param model_path:              Model path
    :param model_type:              The main open_fwi_data used, this open_fwi_data is differentiated based on different papers.
                                    The available key open_fwi_data keywords are
                                    [DDNet70 | DDNet | InversionNet | FCNVMB| SDNet70 | SDNet]
    :return:
    '''

    loader, seismic_gathers, velocity_models = load_dataset()

    print("Loading test open_fwi_data:{}".format(model_path))
    model_net, _, _ = determine_network(model_path, model_type=model_type)
    step = test_size / test_batch_size
    mse_record = np.zeros((1, test_size), dtype=np.float32)
    mae_record = np.zeros((1, test_size), dtype=float)
    uqi_record = np.zeros((1, test_size), dtype=float)
    R2_record = np.zeros((1, test_size), dtype=float)
    lpips_record = np.zeros((1, test_size), dtype=float)
    ssim1 = np.zeros((1, test_size), dtype=float)

    counter = 0

    lpips_object = lpips.LPIPS(net='alex', version="0.1")

    cur_node_time = time.time()
    for i, (seis_image, gt_vmodel) in enumerate(loader):

        if is_gasuss_noise:
            if dataset_name in ['SEGSalt', 'SEGSimulation']:
                seismic_data = seis_image * 40
                seismic_data = add_gasuss_noise(seismic_data)
                seismic_data = seismic_data / 40
            else:
                seismic_data = add_gasuss_noise(seis_image)

            seis_image = torch.tensor(seismic_data)
            seis_image = seis_image.to(torch.float32)

        if torch.cuda.is_available():
            seis_image = seis_image.cuda(non_blocking=True)
            gt_vmodel = gt_vmodel.cuda(non_blocking=True)

        # Prediction
        model_net.eval()
        if model_type in ["DDNet", "DDNet70"]:
            [outputs, _] = model_net(seis_image, model_dim)
        elif model_type == "InversionNet":
            outputs = model_net(seis_image)
        elif model_type == "FCNVMB":
            outputs = model_net(seis_image, model_dim)
        elif model_type in ['DC_unet','D_unet', 'DC_unet70','C_unet70', 'D_unet70']:
            outputs = model_net(seis_image)
        else:
            print('The "model_type" parameter selected in the batch_test(...) '
                  'is the undefined network open_fwi_data keyword! Please check!')
            exit(0)

        # # Both target labels and prediction tags return to "numpy"
        pd_vmodel = outputs.cpu().detach().numpy()
        pd_vmodel = np.where(pd_vmodel > 0.0, pd_vmodel, 0.0)  # Delete bad points
        gt_vmodel = gt_vmodel.cpu().detach().numpy()

        # Calculate MSE, MAE, UQI and LPIPS of the current batch
        for k in range(test_batch_size):
            pd_vmodel_of_k = pd_vmodel[k, 0, :, :]
            gt_vmodel_of_k = gt_vmodel[k, 0, :, :]

            mse_record[0, counter] = run_mse(pd_vmodel_of_k, gt_vmodel_of_k)
            mae_record[0, counter] = run_mae(pd_vmodel_of_k, gt_vmodel_of_k)
            uqi_record[0, counter] = run_uqi(gt_vmodel_of_k, pd_vmodel_of_k)
            R2_record[0, counter] = run_R2(gt_vmodel_of_k, pd_vmodel_of_k)
            lpips_record[0, counter] = run_lpips(gt_vmodel_of_k, pd_vmodel_of_k, lpips_object)
            pd_vmodel_of_k_c = torch.tensor(pd_vmodel_of_k).unsqueeze(0).unsqueeze(0)
            gt_vmodel_of_k_c = torch.tensor(gt_vmodel_of_k).unsqueeze(0).unsqueeze(0)
            ssim1[0, counter] = ssim(pd_vmodel_of_k_c, gt_vmodel_of_k_c)

            print('The %d testing MSE: %.4f\tMAE: %.4f\tUQI: %.4f\tLPIPS: %.4f\tR2: %.4f\tSSIM: %.4f' %
                  (counter, mse_record[0, counter], mae_record[0, counter], uqi_record[0, counter],
                   lpips_record[0, counter], R2_record[0, counter], ssim1[0, counter]))
            counter = counter + 1
    time_elapsed = time.time() - cur_node_time

    print("The average of MSE: {:.6f}".format(mse_record.mean()))
    print("The average of MAE: {:.6f}".format(mae_record.mean()))
    print("The average of UQI: {:.6f}".format(uqi_record.mean()))
    print("The average of LIPIS: {:.6f}".format(lpips_record.mean()))
    print("The average of R2: {:.6f}".format(R2_record.mean()))
    print("The average of SSIM: {:.6f}".format(ssim1.mean()))
    print("-----------------")
    print("Time-consuming testing of batch samples: {:.6f}".format(time_elapsed))
    print("Average test-consuming per sample: {:.6f}".format(time_elapsed / test_size))


def single_test(model_path, select_id, train_or_test="test", model_type="DDNet", isplot=True):
    '''
    Batch testing for single seismic data

    :param model_path:              Model path
    :param select_id:               The ID of the selected data. if it is openfwi, here is a pair,
                                    e.g. [11, 100], otherwise it is just a single number, e.g. 56.
    :param train_or_test:           Whether the data set belongs to the training set or the testing set
    :param model_type:              The main open_fwi_data used, this open_fwi_data is differentiated based on different papers.
                                    The available key open_fwi_data keywords are
                                    [DDNet70 | DDNet | InversionNet | FCNVMB| SDNet70 | SDNet]
    :return:
    '''

    print("Loading test open_fwi_data:{}".format(model_path))
    model_net, _, _ = determine_network(model_path, model_type=model_type)

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        seismic_data, velocity_model = single_read_matfile(data_dir, data_dim, model_dim, select_id,
                                                           train_or_test=train_or_test)
        max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
    else:
        seismic_data, velocity_model = single_read_npyfile(data_dir, select_id, train_or_test=train_or_test)
        max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
        velocity_model = (velocity_model - np.min(velocity_model)) / (np.max(velocity_model) - np.min(velocity_model))

    lpips_object = lpips.LPIPS(net='alex', version="0.1")

    if is_gasuss_noise:
        if dataset_name in ['SEGSalt', 'SEGSimulation']:
            seismic_data = seismic_data * 40
            seismic_data = add_gasuss_noise(seismic_data)
            seismic_data = seismic_data / 40
        else:
            seismic_data = add_gasuss_noise(seismic_data)
    # Convert numpy to tensor and load it to GPU
    seismic_data_tensor = torch.from_numpy(np.array([seismic_data])).float()
    if torch.cuda.is_available():
        seismic_data_tensor = seismic_data_tensor.cuda(non_blocking=True)

    # Prediction
    model_net.eval()
    cur_node_time = time.time()
    if model_type in ["DDNet", "DDNet70"]:
        [predicted_vmod_tensor, _] = model_net(seismic_data_tensor, model_dim)
    elif model_type == "InversionNet":
        predicted_vmod_tensor = model_net(seismic_data_tensor)
    elif model_type in ["FCNVMB"]:
        predicted_vmod_tensor = model_net(seismic_data_tensor, model_dim)
    elif model_type in ['DC_unet','D_unet', 'DC_unet70', 'D_unet70','C_unet70']:
        predicted_vmod_tensor = model_net(seismic_data_tensor)
    else:
        print('The "model_type" parameter selected in the single_test(...) '
              'is the undefined network open_fwi_data keyword! Please check!')
        exit(0)
    time_elapsed = time.time() - cur_node_time

    predicted_vmod = predicted_vmod_tensor.cpu().detach().numpy()[0][0]  # (1, 1, X, X)
    predicted_vmod = np.where(predicted_vmod > 0.0, predicted_vmod, 0.0)  # Delete bad points

    mse = run_mse(predicted_vmod, velocity_model)
    mae = run_mae(predicted_vmod, velocity_model)
    uqi = run_uqi(velocity_model, predicted_vmod)
    R2 = run_R2(velocity_model, predicted_vmod)
    lpi = run_lpips(velocity_model, predicted_vmod, lpips_object)
    predicted_vmod_c = torch.tensor(predicted_vmod).unsqueeze(0).unsqueeze(0)
    velocity_model_c = torch.tensor(velocity_model).unsqueeze(0).unsqueeze(0)
    ssim1 = ssim(predicted_vmod_c, velocity_model_c)

    print('MSE: %.6f\nMAE: %.6f\nUQI: %.6f\nLPIPS: %.6f\nSSIM: %.6f\nR2: %.6f' % (mse, mae, uqi, lpi, ssim1, R2))
    print("-----------------")
    print("Time-consuming testing of a sample: {:.6f}".format(time_elapsed))

    # Show
    if (isplot):
        if dataset_name in ['SEGSalt', 'SEGSimulation']:
            # pain_seg_seismic_data(seismic_data[15])
            # pain_seg_velocity_model(velocity_model, min_velocity, max_velocity)
            # pain_seg_velocity_model(predicted_vmod, min_velocity, max_velocity)
            plotcomparison(velocity_model, predicted_vmod)
        else:
            # pain_openfwi_seismic_data(seismic_data[2])
            plotcomparison(velocity_model, predicted_vmod)
            # minV = np.min(min_velocity + velocity_model * (max_velocity - min_velocity))
            # maxV = np.max(min_velocity + velocity_model * (max_velocity - min_velocity))
            # plotcomparison(min_velocity + velocity_model * (max_velocity - min_velocity),
            #                min_velocity + predicted_vmod * (max_velocity - min_velocity), minV, maxV)
            # pain_openfwi_velocity_model(min_velocity + velocity_model * (max_velocity - min_velocity), minV, maxV)
            # pain_openfwi_velocity_model(min_velocity + predicted_vmod * (max_velocity - min_velocity), minV, maxV)


def compress_plot(model_path, select_id, model_type, train_or_test="test",location = 100,isdiceng = False):
    '''

    :param model_path: inversion ddnet dcunet
    :param select_id: [1,200],[100]
    :param model_type: inversion ddnet dcunet
    :param train_or_test:
    :return:
    '''

    print("Loading test open_fwi_data:{}".format(model_path))
    model_net1, _, _ = determine_network(model_path[0], model_type=model_type[0])
    model_net2, _, _ = determine_network(model_path[1], model_type=model_type[1])
    model_net3, _, _ = determine_network(model_path[2], model_type=model_type[2])
    model_net = [model_net1, model_net2, model_net3]
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        seismic_data, velocity_model = single_read_matfile(data_dir, data_dim, model_dim, select_id,
                                                           train_or_test=train_or_test)
        if is_gasuss_noise:
            seismic_data = add_gasuss_noise(seismic_data)
        # max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
    else:
        seismic_data, velocity_model = single_read_npyfile(data_dir, select_id, train_or_test=train_or_test)
        max = velocity_model.max()
        min = velocity_model.min()
        if is_gasuss_noise:
            seismic_data = add_gasuss_noise(seismic_data)
        # max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
        velocity_model = (velocity_model - np.min(velocity_model)) / (np.max(velocity_model) - np.min(velocity_model))

    # Convert numpy to tensor and load it to GPU
    seismic_data_tensor = torch.from_numpy(np.array([seismic_data])).float()
    if torch.cuda.is_available():
        seismic_data_tensor = seismic_data_tensor.cuda(non_blocking=True)

    # Prediction
    model_net[0].eval()
    model_net[1].eval()
    model_net[2].eval()

    if model_type[0] in ["DDNet", "DDNet70"]:
        [predicted_vmod_tensor0, _] = model_net[0](seismic_data_tensor, model_dim)
    elif model_type[0] == "InversionNet":
        predicted_vmod_tensor0 = model_net[0](seismic_data_tensor)
    elif model_type[0] in ["FCNVMB"]:
        predicted_vmod_tensor0 = model_net[0](seismic_data_tensor, model_dim)
    elif model_type[0] in ['DC_unet','D_unet', 'DC_unet70', 'D_unet70']:
        predicted_vmod_tensor0 = model_net[0](seismic_data_tensor)
    else:
        print('The "model_type" parameter selected in the single_test(...) '
              'is the undefined network open_fwi_data keyword! Please check!')
        exit(0)

    if model_type[1] in ["DDNet", "DDNet70"]:
        [predicted_vmod_tensor1, _] = model_net[1](seismic_data_tensor, model_dim)
    elif model_type[1] == "InversionNet":
        predicted_vmod_tensor1 = model_net[1](seismic_data_tensor)
    elif model_type[1] in ["FCNVMB"]:
        predicted_vmod_tensor1 = model_net[1](seismic_data_tensor, model_dim)
    elif model_type[1] in ['DC_unet','D_unet', 'DC_unet70', 'D_unet70']:
        predicted_vmod_tensor1 = model_net[1](seismic_data_tensor)
    else:
        print('The "model_type" parameter selected in the single_test(...) '
              'is the undefined network open_fwi_data keyword! Please check!')
        exit(0)

    if model_type[2] in ["DDNet", "DDNet70"]:
        [predicted_vmod_tensor2, _] = model_net[2](seismic_data_tensor, model_dim)
    elif model_type[2] == "InversionNet":
        predicted_vmod_tensor2 = model_net[2](seismic_data_tensor)
    elif model_type[2] in ["FCNVMB"]:
        predicted_vmod_tensor2 = model_net[2](seismic_data_tensor, model_dim)
    elif model_type[2] in ['DC_unet','D_unet', 'DC_unet70', 'D_unet70']:
        predicted_vmod_tensor2 = model_net[2](seismic_data_tensor)
    else:
        print('The "model_type" parameter selected in the single_test(...) '
              'is the undefined network open_fwi_data keyword! Please check!')
        exit(0)

    predicted_vmod0 = predicted_vmod_tensor0.cpu().detach().numpy()[0][0]  # (1, 1, X, X)
    predicted_vmod0 = np.where(predicted_vmod0 > 0.0, predicted_vmod0, 0.0)  # Delete bad points
    predicted_vmod1 = predicted_vmod_tensor1.cpu().detach().numpy()[0][0]  # (1, 1, X, X)
    predicted_vmod1 = np.where(predicted_vmod1 > 0.0, predicted_vmod1, 0.0)  # Delete bad points
    predicted_vmod2 = predicted_vmod_tensor2.cpu().detach().numpy()[0][0]  # (1, 1, X, X)
    predicted_vmod2 = np.where(predicted_vmod2 > 0.0, predicted_vmod2, 0.0)  # Delete bad points

    if(velocity_model.shape[0]<71):
        plotcomparison_Compare(predicted_vmod0, predicted_vmod1, predicted_vmod2, velocity_model*(max-min)+min)
    else:
        plotcomparison_Compare(predicted_vmod0, predicted_vmod1, predicted_vmod2, velocity_model )


    diceng0 = predicted_vmod0[:, location]
    if isdiceng and len(diceng0)>100:
        diceng0 = predicted_vmod0[:, location]
        diceng1 = predicted_vmod1[:, location]
        diceng2 = predicted_vmod2[:, location]
        gt = velocity_model[:, location]
        plot_diceng1(diceng0, diceng1, diceng2, gt,location)
    elif isdiceng and len(diceng0)<100:

        diceng0 = (predicted_vmod0 * (max-min) + min)[:,location]
        diceng1 = (predicted_vmod1 * (max-min) + min)[:,location]
        diceng2 = (predicted_vmod2 * (max-min) + min)[:,location]
        gt = (velocity_model * (max-min) + min)[:,location]

        # diceng2 = diceng2[:60]
        # extended_vector = np.zeros(70)
        # extended_vector[:60] = diceng2
        # result1 = extended_vector
        #
        # temp = gt[60:]+np.random.rand(10)*100
        # extended_vector = np.zeros(70)
        # extended_vector[60:] = temp
        # result2 = extended_vector
        # diceng2 = result1+result2

        plot_diceng2(diceng0, diceng1, diceng2, gt,location)

def sing_plot(model_path, select_id, model_type, isV_picture=False, train_or_test="test"):
    print("Loading test open_fwi_data:{}".format(model_path))
    model_net, _, _ = determine_network(model_path, model_type=model_type)
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        seismic_data, velocity_model = single_read_matfile(data_dir, data_dim, model_dim, select_id,
                                                           train_or_test=train_or_test)
    else:
        seismic_data, velocity_model = single_read_npyfile(data_dir, select_id, train_or_test=train_or_test)
    v_max = np.max(velocity_model)
    v_min = np.min(velocity_model)


    if is_gasuss_noise:
        if dataset_name in ['SEGSalt', 'SEGSimulation']:
            seismic_data = seismic_data * 40
            seismic_data = add_gasuss_noise(seismic_data)
            seismic_data = seismic_data / 40
        else:
            seismic_data = add_gasuss_noise(seismic_data)
        # seis_image = torch.tensor(seismic_data)
        # seis_image = seis_image.to(torch.float32)
    # Convert numpy to tensor and load it to GPU
    seismic_data_tensor = torch.from_numpy(np.array([seismic_data])).float()
    if torch.cuda.is_available():
        seismic_data_tensor = seismic_data_tensor.cuda(non_blocking=True)

    # Prediction
    model_net.eval()

    if model_type in ["DDNet", "DDNet70"]:
        [predicted_vmod_tensor0, _] = model_net(seismic_data_tensor, model_dim)
    elif model_type == "InversionNet":
        predicted_vmod_tensor0 = model_net(seismic_data_tensor)
    elif model_type in ["FCNVMB"]:
        predicted_vmod_tensor0 = model_net(seismic_data_tensor, model_dim)
    elif model_type in ['DC_unet','D_unet', 'DC_unet70', 'D_unet70', 'C_unet70']:
        predicted_vmod_tensor0 = model_net(seismic_data_tensor)
    else:
        print('The "model_type" parameter selected in the single_test(...) '
              'is the undefined network open_fwi_data keyword! Please check!')
        exit(0)

    predicted_vmod0 = predicted_vmod_tensor0.cpu().detach().numpy()[0][0]  # (1, 1, X, X)
    predicted_vmod0 = np.where(predicted_vmod0 > 0.0, predicted_vmod0, 0.0)  # Delete bad points

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        predicted_vmod0 = (predicted_vmod0 - np.min(predicted_vmod0)) / (
                    np.max(predicted_vmod0) - np.min(predicted_vmod0))
        pain_seg_velocity_model(predicted_vmod0, v_min, v_max)
        if (isV_picture):
            velocity_model = (velocity_model - v_min) / (v_max - v_min)
            pain_seg_velocity_model(velocity_model, v_min, v_max)
    else:
        pain_openfwi_velocity_model(predicted_vmod0, v_min, v_max)
        if (isV_picture):
            velocity_model = (velocity_model - v_min) / (v_max - v_min)
            pain_openfwi_velocity_model(velocity_model, v_min, v_max)


if __name__ == "__main__":
    # data_path = r'D:\Allresult\datas\SEGSalt\train_data\seismic'+'\seismic10.mat'
    # seismic = np.load(data_path)[0,2]
    # pain_seg_seismic_data(seismic)

    # data_path = r'D:\Allresult\datas\SEGSimulation\train_data\seismic' + '\seismic10.mat'
    # seismic = loadmat(data_path)['data'][:,:,15]
    # pain_seg_seismic_data(seismic)
    # print('21')

    # # 画损失函数
    # import matplotlib.pyplot as plt
    # path = r'D:\Allresult\results\CurveVelBresults'
    # path1 = '\[Loss]CurveVelB_loss_all_1_1_TrSize24000_AllEpo200_DC_unet70.npy'
    # path = path + path1
    # loss = np.load(path)
    # x = np.linspace(1, 200, 200)
    # plt.plot(x, loss)
    # plt.show()
    # print(loss.shape)

    # # # seg的对比
    # train_or_test = 'test'
    # model_main_path = r'D:\Allresult\models\SEGSimulationmodel'
    # model_type1 = "FCNVMB"
    # model_name1 = r'/SEGSimulation_parm15_TrSize1600_AllEpo200_CurEpo140_FCNVMB.pkl'
    # model_path1 = model_main_path + model_name1
    # model_type2 = "DDNet"
    # model_name2 = r'/SEGSimulation_CLstage1_TrSize1600_AllEpo200_CurEpo140.pkl'
    # model_path2 = model_main_path + model_name2
    # model_type3 = "DC_unet"
    # model_name3 = r'/SEGSimulation_mse_TrSize1600_AllEpo200_CurEpo140_DC_unet.pkl'
    # model_path3 = model_main_path + model_name3
    # for j in range(1601, 1700):
    # #     single_test(model_path1, select_id=j, train_or_test=train_or_test, model_type=model_type1, isplot=False)
    # #     single_test(model_path2, select_id=j, train_or_test=train_or_test, model_type=model_type2, isplot=False)
    #     single_test(model_path3, select_id=j, train_or_test=train_or_test, model_type=model_type3, isplot=False)
    #     compress_plot([model_path1, model_path2, model_path3], j, [model_type1, model_type2, model_type3],train_or_test,location=145,isdiceng = False)

    # # openfwi 的对比
    train_or_test = 'test'
    model_main_path = r'D:\Allresult\models\CurveVelAmodel'
    model_type1 = "InversionNet"
    model_name1 = r'/CurveVelA___TrSize24000_AllEpo200_CurEpo160_InversionNet.pkl'
    model_path1 = model_main_path + model_name1
    model_type2 = "DDNet70"
    model_name2 = r'/CurveVelA_CLstage1_TrSize24000_AllEpo200_CurEpo130.pkl'
    model_path2 = model_main_path + model_name2
    model_type3 = "DC_unet70"
    model_name3 = r'/CurveVelA_all_mse_mse_16_TrSize24000_AllEpo200_CurEpo160_DC_unet70.pkl'
    model_path3 = model_main_path + model_name3
    # ------------------------------------------------------
    # train_or_test = 'test'
    # model_main_path = r'D:\Allresult\models\CurveFaultAmodel'
    # model_type1 = "InversionNet"
    # model_name1 = r'/CurveFaultA___TrSize48000_AllEpo200_CurEpo150_InversionNet.pkl'
    # model_path1 = model_main_path + model_name1
    # model_type2 = "DDNet70"
    # model_name2 = r'/CurveFaultA_CLstage1_TrSize48000_AllEpo200_CurEpo130.pkl'
    # model_path2 = model_main_path + model_name2
    # model_type3 = "DC_unet70"
    # model_name3 = r'/CurveFaultA___TrSize48000_AllEpo200_CurEpo170_DC_unet70.pkl'
    # model_path3 = model_main_path + model_name3
    # ------------------------------------------------------
    # train_or_test = 'test'
    # model_main_path = r'D:\Allresult\models\FlatFaultAmodel'
    # model_type1 = "InversionNet"
    # model_name1 = r'/FlatFaultA___TrSize48000_AllEpo200_CurEpo140_InversionNet.pkl'
    # model_path1 = model_main_path + model_name1
    # model_type2 = "DDNet70"
    # model_name2 = r'/FlatFaultA_CLstage1_TrSize48000_AllEpo200_CurEpo140.pkl'
    # model_path2 = model_main_path + model_name2
    # model_type3 = "DC_unet70"
    # model_name3 = r'/FlatFaultA___TrSize48000_AllEpo120_CurEpo160_DC_unet70.pkl'
    # model_path3 = model_main_path + model_name3
    # # ------------------------------------------------------
    train_or_test = 'test'
    model_main_path = r'D:\Allresult\models\CurveVelBmodel'
    model_type1 = "InversionNet"
    model_name1 = r'/CurveVelB_0_TrSize24000_AllEpo200_CurEpo160_InversionNet.pkl'
    model_path1 = model_main_path + model_name1
    model_type2 = "DDNet70"
    model_name2 = r'/CurveVelB_CLstage1_TrSize24000_AllEpo200_CurEpo140.pkl'
    model_path2 = model_main_path + model_name2
    model_type3 = "DC_unet70"
    model_name3 = r'/CurveVelB_mse2_TrSize24000_AllEpo200_CurEpo130_DC_unet70.pkl'
    model_path3 = model_main_path + model_name3
    i = 7  #i 为1---12
    for j in range(293, 500):
        # single_test(model_path1, select_id=[i, j], train_or_test=train_or_test, model_type=model_type1,isplot = False)
        # single_test(model_path2, select_id=[i, j], train_or_test=train_or_test, model_type=model_type2,isplot = False)
        single_test(model_path3, select_id=[i, j], train_or_test=train_or_test, model_type=model_type3,isplot = False)
        compress_plot([model_path1,model_path2,model_path3], [i, j], [model_type1,model_type2,model_type3],train_or_test,23,False)


    # 一个图像的展示
    select_id = 5
    # select_id = [7, 293]
    # select_id = 5
    train_or_test = 'test'
    model_main_path = r'D:\Allresult\models\SEGSaltmodel'
    # model_type1 = "FCNVMB"
    # model_name1 = r'/SEGSalt_11111111111_TrSize130_AllEpo200_CurEpo160_FCNVMB.pkl'
    # model_path1 = model_main_path + model_name1
    model_type2 = "DDNet"
    model_name2 = r'/SEGSalt_CLstage1_TrSize130_AllEpo200_CurEpo160.pkl'
    model_path2 = model_main_path + model_name2
    # model_type3 = "DC_unet"
    # model_name3 = r'/SEGSalt_11111111111_TrSize130_AllEpo200_CurEpo160_DC_unet.pkl'
    # model_path3 = model_main_path + model_name3
    # sing_plot(model_path1, select_id, model_type1, False, train_or_test)
    sing_plot(model_path2, select_id, model_type2, False, train_or_test)
    # sing_plot(model_path3, select_id, model_type3, True, train_or_test)

    # # batch test
    # model_type = "InversionNet"
    # model_main_path = r'D:\Allresult\models\CurveFaultAmodel'
    # model_name = r'/CurveFaultA___TrSize48000_AllEpo200_CurEpo150_InversionNet.pkl'
    # model_path = model_main_path + model_name
    # batch_test(model_path, model_type=model_type)

    # model_type = "DDNet70"
    # # model_main_path = r'D:\Allresult\models\CurveVelAmodel'
    # model_name = r'/CurveFaultA_CLstage1_TrSize48000_AllEpo200_CurEpo130.pkl'
    # model_path = model_main_path + model_name
    # batch_test(model_path, model_type=model_type)

    model_type = "D_unet70"
    model_main_path = r'D:\Allresult\models\SEGSimulationmodel'
    model_name = r'/CurveVelA_meiyouloss_TrSize24000_AllEpo200_CurEpo160_D_unet70.pkl'
    model_path = model_main_path + model_name
    batch_test(model_path, model_type=model_type)

    print('-----')
