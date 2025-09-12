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

from asteris.train import training_class
import swanlab

# This script is used to train the ASTERIS model on the JWST dataset.
project_name = 'ASTERIS_training'
# Path to the dataset
# The dataset shall be organised as pre-processed [with sigma-clipping], grayscale, 3D tiff arrays
# Train dataset path
datasets_path = "./train_datasets/"
# ASTERIS model '4' or '8'
train_mode = 8
# Whether to MASK the nan-value pixel for training: 0 for no mask, 1 for masking
mask_train = 0
# Start the tarining from the last checkpoint: 0 for starting a new training, 1 for continuing training
continue_train = 0          
# If continue_train = 1, set the path of the last checkpoint
checkpoint_path = ''
# Index of GPU used for computation (e.g. '0', '0,1', '0,1,2')
GPU = '0,1,2,3'              
# Recommended number of training epochs: from 10 to 20
n_epochs = 20                
# Recommended learning rate: from 1.0e-4 to 1.5e-4
learning_rate = 1.5e-4     
# Batch size for each GPU, the total batch size = batch_size * len(GPU)
batch_size = 3
# The width and height of 3D patches
patch_xy = 128       
# The overlap factor between two adjacent patches: from 0 to 1
overlap_factor = 0.1           
# Pth file and visualization result file path
pth_dir = "./pth/"  
# If you use Windows system, set this to 0.         
num_workers = 8                

train_dict = {
    # Dataset dependent parameters
    'patch_x': patch_xy,
    'patch_y': patch_xy,
    'patch_t': train_mode,
    'overlap_factor':overlap_factor,
    'datasets_path': datasets_path,
    'pth_dir': pth_dir,
    # Network related parameters
    'n_epochs': n_epochs,
    'batch_size': batch_size,
    'lr': learning_rate,                
    'b1': 0.9,   # Adam: beta1
    'b2': 0.999, # Adam: beta2
    'fmap': 24,  # The number of feature maps in the first layer
    'GPU': GPU,
    'num_workers': num_workers,
    'checkpoint_path': checkpoint_path,
    'continue_train': continue_train,
    'mask_train': mask_train
}

# Start a new Swanlab run to track this script
swanlab.init(
    project= project_name,
    config=train_dict, 
    workspace="ASTERIS_training"
)

tc = training_class(train_dict)
tc.run()