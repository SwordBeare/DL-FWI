# -*- coding: utf-8 -*-
"""
Path setting

Created on Sep 2024

@author: Jian An (2569222191@qq.com)

"""

from param_config import *                                                      # Get the dataset name
import os

###################################################
####                 PATHS                    #####
###################################################

main_dir        = r'D:\Allresult/'
data_dir        = main_dir + 'datas/'                                            # The path of dataset
# results_dir     = main_dir + 'results/'                                         # Output path of run results (not open_fwi_data information)
results_dir     = main_dir+'results/'
# models_dir      = main_dir + 'models/'                                          # The path where the open_fwi_data will be stored at the end of the run
models_dir      = main_dir+'models/'                                                # The path where the open_fwi_data will be stored at the end of the run

###################################################
####              DYNAMIC PATHS               #####
###################################################

temp_results_dir= results_dir + '{}results/'.format(dataset_name)                   # Generate results storage paths for specific dataset
temp_models_dir = models_dir  + '{}model/'.format(dataset_name)                 # Generate open_fwi_data   storage paths for specific dataset
data_dir        = data_dir    + '{}/'.format(dataset_name)                      # Generate data    storage paths for specific dataset

if os.path.exists(temp_results_dir) and os.path.exists(temp_models_dir):
    results_dir = temp_results_dir
    models_dir  = temp_models_dir
else:
    os.makedirs(temp_results_dir)
    os.makedirs(temp_models_dir)
    results_dir = temp_results_dir
    models_dir  = temp_models_dir
