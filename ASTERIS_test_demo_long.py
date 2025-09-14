# ASTERIS: Pushing Detection Limits of Astronomical Imaging via Self-supervised Spatiotemporal Denoising
# Author: Yuduo Guo, Hao Zhang, Mingyu Li
# Tsinghua University, Beijing, China

# ASTERIS is a deep learning framework for pushing the detection limit of astronomical 
# imaging, with a focus on spatiotemporal denoising across multi-exposure observations. 
# It is built upon and extends the architecture of 
# [Restormer](https://arxiv.org/abs/2111.09881) by introducing temporal modeling and 
# adaptive restoration tailored for scientific image sequences.
# We sincerely thank the original authors of Restormer for making their code and design 
# publicly available.

# ## ⚖️ License & Copyright
# All original contributions, modifications, and extensions made in this project, 
# including the ASTERIS model and training framework, are copyrighted © 2025 by Yuduo Guo.
# This repository is released under the MIT License, unless otherwise specified. 
# See the [LICENSE](./LICENSE) file for details.
# ---
# ## ✉️ Contact
# For questions or potential collaborations, please contact Yuduo Guo at `gyd@mail.tsinghua.edu.cn`.
# Copyright (c) 2025 Yuduo Guo. All rights reserved.
# Date: 2025-09-12
import os
from asteris.test import testing_class
from asteris.utils import list_subfolders_or_self,make_stack

restore_clip_part = True
# Datasets path for ASTERIS input
datasets_path = "./test_datasets/long/" 
# Dataets path for ASTERIS output
save_path = "./result/long/"
# Hdu_num of the FITS file, usually 1 for JWST NIRCam data
hdu_num = 1
# ASTERIS model 
denoise_model = 'ASTERIS4_nrclong' 
# the version of ASTERIS, 4 or 8
test_mode = 4
# Index of GPU you will use for computation (e.g. '0', '0,1', '0,1,2')  ``    
GPU = '0,1,2,3'
# Batch size for each GPU, the total batch size = batch_size * len(GPU)
batch_size = 1
# Width and height of 3D patches, has to be //8                    
patch_xy = 800
# the overlap factor between two adjacent patches.                          
overlap_factor = 0.1              
# if you use Windows system, set this to 0.0            
num_workers = 8 

##################################
###Fixed parameters for testing###
##################################
# Sigma threshold for z-score
sigma_thresh_val = 3.0       
# scale factor for z-normalisation
scale_factor = 4 
##################################       

test_dict = {
    'restore_clip_part':restore_clip_part,
    # dataset dependent parameters
    'patch_x': patch_xy,
    'patch_y': patch_xy,
    'patch_t': test_mode,
    'overlap_factor':overlap_factor,
    'test_datasize': test_mode,
    'datasets_path': datasets_path+'cache/images_for_test/',
    'pth_dir': './pth/',                 
    'denoise_model' : denoise_model,  
    'output_dir' : save_path,  
    'prefix': datasets_path.split('/') [-2],      
    # network related parameters 
    'fmap': 24,                          
    'GPU': GPU,
    'num_workers': num_workers,
    'batch_size':batch_size,
    'scale_factor': scale_factor,
    'sigma_thresh': sigma_thresh_val
}

cache_dir = os.path.join(datasets_path, "cache")
if not os.path.isdir(cache_dir):
    ###### Preprocess raw cal data for ASTERIS ######
    test_path_list = list_subfolders_or_self(datasets_path)
    for sub_path in test_path_list:
        make_stack(scale_factor, sub_path,datasets_path+'cache/',hdu_num, test_mode, make_val = True,  sigma_thresh = sigma_thresh_val)
    
tc = testing_class(test_dict)  
tc.run()



