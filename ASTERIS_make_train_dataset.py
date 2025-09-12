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
from asteris.utils import make_train_datasets

# Fits image path (after reduction and astrometrical alignment)
reduction_image_path = './reduction_datasets/'
# Set to 1 for ranking input fits images by the relative MSE value
mse_select = 1
# Which extension is the science data in
hdu_num = 1
# Sigma threshold along temporal domain to remove outliers
z_axis_clip = 3.0
# Sigma threshold for z-score
clip_threshold = 3.0
# Output train dataset path
datasets_path = "./train_datasets/"
# Scale factor for z-normalisation
scale_factor = 4.0
make_train_datasets(scale_factor, mse_select, hdu_num, z_axis_clip, 
                    clip_threshold, reduction_image_path, datasets_path)
