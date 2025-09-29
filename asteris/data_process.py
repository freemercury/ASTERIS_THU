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
# Date: 2025-05-22

import numpy as np
import os
import tifffile as tiff
import random
import math
import torch
from torch.utils.data import Dataset
import warnings
from tqdm import tqdm


def random_transform_3d(input, target):
    """
    The function for data augmentation. Randomly select one method among five
    transformation methods (including rotation and flip) or do not use data
    augmentation.

    Args:
        input, target : the input and target patch before data augmentation
    Return:
        input, target : the input and target patch after data augmentation
    """
    p_trans = random.randrange(8)
    if p_trans == 0:  # no transformation
        input = input
        target = target
    elif p_trans == 1:  # left rotate 90
        input = np.rot90(input, k=1, axes=(1, 2))
        target = np.rot90(target, k=1, axes=(1, 2))
    elif p_trans == 2:  # left rotate 180
        input = np.rot90(input, k=2, axes=(1, 2))
        target = np.rot90(target, k=2, axes=(1, 2))
    elif p_trans == 3:  # left rotate 270
        input = np.rot90(input, k=3, axes=(1, 2))
        target = np.rot90(target, k=3, axes=(1, 2))
    elif p_trans == 4:  # horizontal flip
        input = input[:, :, ::-1]
        target = target[:, :, ::-1]
    elif p_trans == 5:  # horizontal flip & left rotate 90
        input = input[:, :, ::-1]
        input = np.rot90(input, k=1, axes=(1, 2))
        target = target[:, :, ::-1]
        target = np.rot90(target, k=1, axes=(1, 2))
    elif p_trans == 6:  # horizontal flip & left rotate 180
        input = input[:, :, ::-1]
        input = np.rot90(input, k=2, axes=(1, 2))
        target = target[:, :, ::-1]
        target = np.rot90(target, k=2, axes=(1, 2))
    elif p_trans == 7:  # horizontal flip & left rotate 270
        input = input[:, :, ::-1]
        input = np.rot90(input, k=3, axes=(1, 2))
        target = target[:, :, ::-1]
        target = np.rot90(target, k=3, axes=(1, 2))
    return input, target


class trainset(Dataset):
    """
    Train set generator for pytorch training
    Produces paired 3D tiles (input, target) 
    by interlacing frames from a 3D noisy stack.
    """

    def __init__(self, name_list, coordinate_list, noise_img_all, stack_index):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index

    def __getitem__(self, index):
        """
        Returns:
            input, target: torch.Tensors with shape (1, t/2, h, w) after slicing,
                           built from interlaced frames of the selected sub-stack.

        For small lateral size / short time stacks:
          1) Crop a 3D sub-stack (patch) using recorded coordinates.
          2) Interlace frames: even-indexed frames -> input, odd-indexed -> target.
          3) Zero-mean normalize each half independently (ignore zeros as NaNs).
          4) Optionally swap input/target and apply random 3D augmentation.
        """
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        # ----- Locate which full stack this patch comes from ------------------
        stack_index = self.stack_index[index]
        noise_img = self.noise_img_all[stack_index]
        # ----- Retrieve stored coordinates for this patch ---------------------
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        
        # ----- Split interlaced frames into two 3D tiles ----------------------
        #### Trainset patches zero-mean ####
        tmp1 = noise_img[init_s:end_s:2, init_h:end_h, init_w:end_w].copy()
        tmp2 = noise_img[init_s + 1:end_s:2, init_h:end_h, init_w:end_w].copy()
        # NaN masking
        tmp1[tmp1 == 0] = np.nan
        tmp2[tmp2 == 0] = np.nan
        # ----- Zero-mean per half (ignore NaNs) -------------------------------
        zm_noise_img_input = np.nanmean(tmp1,0)
        bias_zeromean_input = np.nanmedian(zm_noise_img_input)
        zm_noise_img_output = np.nanmean(tmp2,0)
        bias_zeromean_output = np.nanmedian(zm_noise_img_output)
        
        input = tmp1  - bias_zeromean_input
        target = tmp2  - bias_zeromean_output
        # Fill NaNs back to zeros
        input[np.isnan(input)] = 0
        target[np.isnan(target)] = 0
        
        # ----- Randomly decide whether to swap input/target -------------------
        p_exc = random.random()  
        if p_exc < 0.5:
            # Augment without swapping
            input, target = random_transform_3d(input, target)
        else:
            #### Swap input and target, then augment
            temp = input
            input = target
            target = temp  
            input, target = random_transform_3d(input, target)
        # ----- Convert to Torch tensors, add channel dim ----------------------
        input = torch.from_numpy(np.expand_dims(input, 0).copy())
        target = torch.from_numpy(np.expand_dims(target, 0).copy())
               
        return input, target

    def __len__(self):
        return len(self.name_list)


class testset(Dataset):
    """
    Test set generator for pytorch inference
    Iterates over all recorded patch coordinates for each test image,
    extracts the corresponding 3D sub-stack (patch), applies zero-mean
    normalization (median-based, ignoring zeros), and returns the patch
    plus metadata for stitching.
    """

    def __init__(self, name_lists, coordinate_lists, noise_imgs):
        self.name_lists = name_lists
        self.coordinate_lists = coordinate_lists
        self.noise_imgs = noise_imgs
        self.img_indices = []
        self.stack_indices = []
        # Flatten (image_id, patch_index) pairs into linear indices
        for img_id, name_list in enumerate(self.name_lists):
            self.img_indices.extend([img_id] * len(name_list))
            self.stack_indices.extend(range(len(name_list)))

        self.num_stacks = len(self.stack_indices)

    def __getitem__(self, index):
        """
        Generate one noisy sub-stack (patch) for inference.

        Args:
            index: linear index across all (image, patch) pairs

        Returns:
            img_index        : which image this patch belongs to
            noise_patch      : torch.FloatTensor (1, t, h, w) zero-mean normalized
            single_coordinate: dict with slicing coords for later stitching
            mean_val         : scalar median used for de-biasing (to be added back)
        """
        # Map linear index -> (image id, patch id within image)
        img_index, stack_index = self.img_indices[index], self.stack_indices[index]
        # Fetch coordinates for this patch
        single_coordinate = self.coordinate_lists[img_index][self.name_lists[img_index][stack_index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        # Crop 3D patch from full noisy stack
        noise_patch = self.noise_imgs[img_index][init_s:end_s, init_h:end_h, init_w:end_w]
        
        ################################################
        ####  Remove median value as for inference #####
        ################################################
        tmp1 = noise_patch.copy()
        # NaN masking
        tmp1[tmp1 == 0] = np.nan
        # Median across (h, w) of the per-voxel median along temporal domain
        zm_noise_patch = np.nanmedian(tmp1,0)
        mean_val = np.nanmedian(zm_noise_patch) # scalar bias
        # De-bias and fill NaNs back to zeros
        noise_patch = tmp1  - mean_val
        noise_patch[np.isnan(noise_patch)] = 0
        if np.isnan(mean_val):
            mean_val = 0    

        # Add channel dim for 3D CNNs: (1, t, h, w)
        noise_patch = torch.from_numpy(np.expand_dims(noise_patch, 0))
        return img_index, noise_patch, single_coordinate, mean_val

    def __len__(self):
        return self.num_stacks


def singlebatch_test_save(single_coordinate, output_image, raw_image):
    """
    Subtract overlapping regions (both lateral and temporal) from the output sub-stack
    when batch size == 1, and return the cropped patches along with their placement
    coordinates in the full stack.

    Args:
        single_coordinate (dict): coordinates describing where the patch lies in the full stack
            and which part of the sub-stack is valid (non-overlapped) to keep.
        output_image (np.ndarray): network output sub-stack (shape either (t,h,w) or (h,w))
        raw_image (np.ndarray): original noisy sub-stack (shape (t,h,w))

    Returns:
        output_patch (np.ndarray): cropped network output (overlaps removed)
        raw_patch (np.ndarray): cropped raw input (overlaps removed)
        stack_start_w, stack_end_w (int, int): lateral W placement in full stack
        stack_start_h, stack_end_h (int, int): lateral H placement in full stack
        stack_start_s, stack_end_s (int, int): temporal S placement in full stack
    """
    # ----- Read placement & valid (non-overlap) ranges from metadata ----------
    # Stack & patch coordinates
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])

    # ----- Crop the valid region from network output --------------------------
    if (output_image.ndim == 3):
        # 3D output: crop temporal + spatial ranges
        output_patch = output_image[
            patch_start_s:patch_end_s, 
            patch_start_h:patch_end_h, 
            patch_start_w:patch_end_w
            ]
    else:
        # 2D output: crop spatial ranges only
        output_patch = output_image[
            patch_start_h:patch_end_h, 
            patch_start_w:patch_end_w
            ]
    # ----- Crop the corresponding valid region from the raw input -------------
    raw_patch = raw_image[
        patch_start_s:patch_end_s, 
        patch_start_h:patch_end_h, 
        patch_start_w:patch_end_w
        ]
    
    return output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def multibatch_test_save(single_coordinate, id, output_image, raw_image):
    """
    Subtract overlapping regions (lateral + temporal) from batched output sub-stacks
    (for batch size > 1). Extract the valid (non-overlapped) region for the given
    sample `id` in the batch, and return the cropped patches along with placement
    coordinates in the full stack.

    Args:
        single_coordinate (dict of torch.Tensors): each value is a tensor of shape (B,)
            holding per-sample coordinates for the batch (e.g., stack_start_w[id]).
        id (int): which sample in the batch to process.
        output_image (np.ndarray or torch.Tensor): model outputs, shape (B, ...) with
            either (t,h,w) or (h,w) per sample.
        raw_image (np.ndarray or torch.Tensor): raw inputs, shape (B, t, h, w).

    Returns:
        output_patch (ndarray/Tensor): cropped model output for sample `id` (overlaps removed).
        raw_patch (ndarray/Tensor): cropped raw input for sample `id` (overlaps removed).
        stack_start_w, stack_end_w (int, int): width placement in full stack.
        stack_start_h, stack_end_h (int, int): height placement in full stack.
        stack_start_s, stack_end_s (int, int): temporal placement in full stack.
    """
    # ---- Extract per-sample placement & valid ranges from batched tensors ----
    stack_start_w_id = single_coordinate['stack_start_w'].numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = single_coordinate['stack_end_w'].numpy()
    stack_end_w = int(stack_end_w_id[id])
    patch_start_w_id = single_coordinate['patch_start_w'].numpy()
    patch_start_w = int(patch_start_w_id[id])
    patch_end_w_id = single_coordinate['patch_end_w'].numpy()
    patch_end_w = int(patch_end_w_id[id])

    stack_start_h_id = single_coordinate['stack_start_h'].numpy()
    stack_start_h = int(stack_start_h_id[id])
    stack_end_h_id = single_coordinate['stack_end_h'].numpy()
    stack_end_h = int(stack_end_h_id[id])
    patch_start_h_id = single_coordinate['patch_start_h'].numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_end_h_id = single_coordinate['patch_end_h'].numpy()
    patch_end_h = int(patch_end_h_id[id])

    stack_start_s_id = single_coordinate['stack_start_s'].numpy()
    stack_start_s = int(stack_start_s_id[id])
    stack_end_s_id = single_coordinate['stack_end_s'].numpy()
    stack_end_s = int(stack_end_s_id[id])
    patch_start_s_id = single_coordinate['patch_start_s'].numpy()
    patch_start_s = int(patch_start_s_id[id])
    patch_end_s_id = single_coordinate['patch_end_s'].numpy()
    patch_end_s = int(patch_end_s_id[id])
    # ---- Select id-th sample from batched outputs/inputs --------------------
    output_image_id = output_image[id]
    raw_image_id = raw_image[id]
    # ---- Crop valid (non-overlap) region ------------------------------------
    if (output_image_id.ndim == 3):
        output_patch = output_image_id[
            patch_start_s:patch_end_s, 
            patch_start_h:patch_end_h, 
            patch_start_w:patch_end_w
            ]
    else:
        output_patch = output_image_id[
            patch_start_h:patch_end_h, 
            patch_start_w:patch_end_w
            ]
    # Raw patch always has shape (t,h,w)
    raw_patch = raw_image_id[
        patch_start_s:patch_end_s, 
        patch_start_h:patch_end_h, 
        patch_start_w:patch_end_w
        ]

    return output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s

    
def test_preprocess(args):
    """
    Partition each original noisy stack into overlapping 3D sub-stacks (patches)
    for inference, and record both:
      - the *valid* (non-overlapping) region inside each patch (for stitching),
      - the *placement* coordinates in the full stack.

    Returns per-image lists:
        name_lists        : list of patch IDs
        noise_imgs        : list of zero-mean stacks (np.float32)
        coordinate_lists  : list of dicts {patch_name: coords}
        im_names          : list of image file names
        img_means         : list of per-image scalar medians subtracted before inference
    """
    random_flag = args.random_flag
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t
    gap_y = args.gap_y
    gap_x = args.gap_x
    gap_t2 = args.patch_t
    # Amount to trim from each side when stitching
    cut_w = (patch_x - gap_x) / 2
    cut_h = (patch_y - gap_y) / 2
    cut_s = (patch_t2 - gap_t2) / 2

    name_lists = []
    coordinate_lists = []
    im_names = []
    img_means = []
    noise_imgs = []

    print("Loading Image...")
    for img_id in tqdm(range(len(args.img_list))):
        im_dir = args.img_list[img_id]
        im_name = os.path.basename(im_dir)
        im_names.append(im_name)
        
        ####################################
        # Read stack and set up containers #
        ####################################
        noise_im = tiff.imread(im_dir) # (T, H, W)
        
        name_list = []
        coordinate_list = {}
        
        if noise_im.shape[0] >= args.test_datasize:
            noise_im = noise_im[args.test_datasize-args.patch_t:args.test_datasize, :, :]
        
        if args.print_img_name:
            print('Testing image name -----> ', im_name)
            print('Testing image shape -----> ', noise_im.shape)
        
        # -------- Randomize frame order --------------------------------------
        if random_flag:
            np.random.shuffle(noise_im) 
        # -------- Zero-mean per stack (ignore zeros as NaNs) -----------------
        tmp1 = noise_im.astype(np.float32)
        tmp1[tmp1 == 0] = np.nan # NaN masking
        zm_noise_img_input = np.nanmean(tmp1,0)
        img_mean = np.nanmedian(zm_noise_img_input)
        noise_im = tmp1  - img_mean
        noise_im[np.isnan(noise_im)] = 0 # fill back to 0

        whole_x = noise_im.shape[2]
        whole_y = noise_im.shape[1]
        whole_t = noise_im.shape[0]
        # Number of patches along each dimension (ceil to cover borders)
        num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
        num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
        num_s = math.ceil((whole_t - patch_t2 + gap_t2) / gap_t2)
        # -------- Enumerate all patch positions ------------------------------
        for x in range(0, num_h):
            for y in range(0, num_w):
                for z in range(0, num_s):
                    single_coordinate = {
                        'init_h': 0, 'end_h': 0, 
                        'init_w': 0, 'end_w': 0, 
                        'init_s': 0, 'end_s': 0
                        }
                    # Compute crop bounds inside the stack (handle last tile clamping to edge)
                    if x != (num_h - 1):
                        init_h = gap_y * x
                        end_h = gap_y * x + patch_y
                    elif x == (num_h - 1):
                        init_h = whole_y - patch_y
                        end_h = whole_y

                    if y != (num_w - 1):
                        init_w = gap_x * y
                        end_w = gap_x * y + patch_x
                    elif y == (num_w - 1):
                        init_w = whole_x - patch_x
                        end_w = whole_x

                    if z != (num_s - 1):
                        init_s = gap_t2 * z
                        end_s = gap_t2 * z + patch_t2
                    elif z == (num_s - 1):
                        init_s = whole_t - patch_t2
                        end_s = whole_t
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    # -------- Stitching metadata: where to place and what to keep ----------
                    if y == 0:
                        # First column: keep left half fully, drop right overlap half
                        single_coordinate['stack_start_w'] = y * gap_x
                        single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = patch_x - cut_w
                    elif y == num_w - 1:
                        # Last column: drop left overlap half, keep right to the edge
                        single_coordinate['stack_start_w'] = whole_x - patch_x + cut_w
                        single_coordinate['stack_end_w'] = whole_x
                        single_coordinate['patch_start_w'] = cut_w
                        single_coordinate['patch_end_w'] = patch_x
                    else:
                        # Middle columns: drop both overlap halves
                        single_coordinate['stack_start_w'] = y * gap_x + cut_w
                        single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                        single_coordinate['patch_start_w'] = cut_w
                        single_coordinate['patch_end_w'] = patch_x - cut_w

                    if x == 0:
                        single_coordinate['stack_start_h'] = x * gap_y
                        single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = patch_y - cut_h
                    elif x == num_h - 1:
                        single_coordinate['stack_start_h'] = whole_y - patch_y + cut_h
                        single_coordinate['stack_end_h'] = whole_y
                        single_coordinate['patch_start_h'] = cut_h
                        single_coordinate['patch_end_h'] = patch_y
                    else:
                        single_coordinate['stack_start_h'] = x * gap_y + cut_h
                        single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                        single_coordinate['patch_start_h'] = cut_h
                        single_coordinate['patch_end_h'] = patch_y - cut_h

                    if z == 0:
                        single_coordinate['stack_start_s'] = z * gap_t2
                        single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = patch_t2 - cut_s
                    elif z == num_s - 1:
                        single_coordinate['stack_start_s'] = whole_t - patch_t2 + cut_s
                        single_coordinate['stack_end_s'] = whole_t
                        single_coordinate['patch_start_s'] = cut_s
                        single_coordinate['patch_end_s'] = patch_t2
                    else:
                        single_coordinate['stack_start_s'] = z * gap_t2 + cut_s
                        single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                        single_coordinate['patch_start_s'] = cut_s
                        single_coordinate['patch_end_s'] = patch_t2 - cut_s
                    # Unique patch name
                    patch_name = args.datasets_name + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                    name_list.append(patch_name)
                    coordinate_list[patch_name] = single_coordinate
        # Collect per-image outputs
        name_lists.append(name_list)
        coordinate_lists.append(coordinate_list)
        noise_imgs.append(noise_im)
        img_means.append(img_mean)

    return name_lists, noise_imgs, coordinate_lists, im_names, img_means

def _zero_ratio_of_input(noise_img, coord):
    """
    tmp1 = noise_img[init_s:end_s:2, init_h:end_h, init_w:end_w]
    """
    init_h, end_h = coord['init_h'], coord['end_h']
    init_w, end_w = coord['init_w'], coord['end_w']
    init_s, end_s = coord['init_s'], coord['end_s']
    tmp1 = noise_img[init_s:end_s:2, init_h:end_h, init_w:end_w]
    # 以整个 3D 输入块（T/2, H, W）为统计基准
    return float(np.mean(tmp1 == 0))


def filter_samples(name_list, coordinate_list, noise_img_all, stack_index,
                   zero_threshold=0.2, verbose=True):

    kept_names, kept_stack_idx = [], []
    dropped, ratios = 0, []

    for i, name in enumerate(name_list):
        coord = coordinate_list[name]
        nz_ratio = _zero_ratio_of_input(noise_img_all[stack_index[i]], coord)
        ratios.append(nz_ratio)
        if nz_ratio <= zero_threshold:
            kept_names.append(name)
            kept_stack_idx.append(stack_index[i])
        else:
            dropped += 1

    if verbose:
        total = len(name_list)
        kept = len(kept_names)
        print(f"[Filter] total={total}, kept={kept}, dropped={dropped}, "
              f"threshold={zero_threshold:.2f}, "
              f"kept_ratio={kept/total if total else 0:.2%}")
        if total:
            import numpy as np
            arr = np.array(ratios, dtype=float)
            print(f"[Filter] zero_ratio stats (all before filtering): "
                  f"min={arr.min():.3f}, "
                  f"median={np.median(arr):.3f}, "
                  f"max={arr.max():.3f}")

    return kept_names, kept_stack_idx