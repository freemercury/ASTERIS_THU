import os
import datetime
import numpy as np
import yaml
import torch.nn as nn
import tifffile as tiff
import math
import torch
import time
import datetime
import swanlab
from .ASTERIS_net_8 import ASTERIS8 
from .ASTERIS_net_4 import ASTERIS4
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .data_process import trainset
from .data_process import filter_samples

class training_class():
    """
    Class implementing training process
    """ 
    
    def __init__(self, params_dict):
        """
        Constructor class for training process

        Args:
           params_dict: dict
               The collection of training params set by users
        Returns:
           self
        """
        self.overlap_factor = 0.1
        self.datasets_path = ''
        self.n_epochs = 10
        self.fmap = 24
        self.pth_dir = './pth/'
        self.batch_size = 1
        self.continue_train = 0
        self.patch_t = 8
        self.patch_x = 128
        self.patch_y = 128
        self.lr = 1e-4
        self.b1 = 0.9
        self.b2 = 0.999
        self.GPU = '0'
        self.ngpu = 1
        self.num_workers = 0
        self.checkpoint_path = ''
        self.set_params(params_dict)
        self.mask_train = 0

    def check_nan(self, arr):
        return 1 if np.isnan(arr).any() else 0
        
    def run(self):
        """
        General function for training ASTERIS network.
        """
        # create essential files for result storage
        self.prepare_file()
        # crop input tiff file into 3D patches
        self.train_preprocess()
        # save some essential training parameters in para.yaml
        self.save_yaml_train()
        # initialize denoise network with training parameters.
        self.initialize_network()
        # specifies the GPU for the training program.
        self.distribute_GPU()
        # start training and result visualization during training period (optional)
        self.train()
    
    def prepare_file(self):
        """
        Make data folder to store training results
        Important Fields:
            self.datasets_name: the sub folder of the dataset
            self.pth_path: the folder for pth file storage
        """
        if self.datasets_path[-1]!='/':
           self.datasets_name=self.datasets_path.split("/")[-1]
        else:
            self.datasets_name=self.datasets_path.split("/")[-2]

        pth_name = self.datasets_name + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.pth_path = self.pth_dir + '/' + pth_name
        if not os.path.exists(self.pth_path):
            os.makedirs(self.pth_path)

    def set_params(self, params_dict):
        """
        Set the params set by user to the training class object and calculate some default parameters for training

        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # patch gap in x
        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))  
        # patch gap in y
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))  
        # patch gap in t
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))  
        # check the number of GPU used for training
        self.ngpu = str(self.GPU).count(',') + 1             
        # By default, the batch size is equal to the number of GPU for minimal memory consumption       
        self.batch_size = self.ngpu * self.batch_size               
        print('\033[1;31mTraining parameters -----> \033[0m')
        print(self.__dict__)

    def initialize_network(self):
        """
        Initialize ASTERIS network

        Important Fields:
           self.fmap: the number of the feature map in U-Net 3D network.
           self.local_model: the denoise network
        """
        if self.patch_t == 4:
            ASTERIS = ASTERIS4
            num_HEAD = [4, 6, 8]

        elif self.patch_t == 8:
            ASTERIS = ASTERIS8
            num_HEAD = [4, 6, 6, 8]
            
        # Initialize ASTERIS network
        denoise_generator = ASTERIS(inp_channels=1, 
                                out_channels=1, 
                                f_maps = self.fmap,
                                num_blocks = num_HEAD,
                                num_refinement_blocks = 4)
        self.local_model = denoise_generator

    def load_checkpoint(self, model, optimizer, scheduler, filename):
        checkpoint = torch.load(filename)
        
        if isinstance(self.local_model, nn.DataParallel):           
            self.local_model.module.load_state_dict(checkpoint["model_state_dict"])
            self.local_model.eval()                  
        else:
            self.local_model.load_state_dict(checkpoint["model_state_dict"])
            self.local_model.eval()  
        
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        return start_epoch

    def train_preprocess(self):
        """
        The original noisy stack is partitioned into thousands of 3D sub-stacks (patch) with the setting
        overlap factor in each dimension.

        Important Fields:
           self.name_list : the coordinates of 3D patch are indexed by the patch name in name_list.
           self.coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack.
           self.stack_index : the index of the noisy stacks.
           self.noise_im_all : the collection of all noisy stacks.

        """
        self.name_list = []
        self.coordinate_list = {}
        self.stack_index = []
        self.noise_im_all = []
        ind = 0
        print('\033[1;31mImage list for training -----> \033[0m')
        
        ###### Collect all image file paths under dataset folder and subfolders ########
        self.all_file_paths = []
        for root, dirs, files in os.walk(self.datasets_path):
            for file in files:
                self.all_file_paths.append(os.path.join(root, file))
        self.stack_num = len(self.all_file_paths)   
        print('Total stack number -----> ', self.stack_num)
        
        # Iterate through each image stack file
        for im_name in self.all_file_paths:
            print('Noise image name -----> ', im_name)
            # Load 3D stack in tif format
            noise_im = tiff.imread(im_name)
            # Dimensions
            self.whole_x = noise_im.shape[2]
            self.whole_y = noise_im.shape[1]
            self.whole_t = noise_im.shape[0]
            
            # Set zero values to nan
            tmp1 = noise_im.astype(np.float32)
            tmp1[tmp1 == 0] = np.nan
            # Background removal
            zm_noise_img_input = np.nanmean(tmp1,0)
            img_mean = np.nanmedian(zm_noise_img_input)
            noise_im = tmp1  - img_mean
            # Set nans back to zero
            noise_im[np.isnan(noise_im)] = 0            
            self.noise_im_all.append(noise_im)
            
            # Set temporal overlap
            patch_t2 = self.patch_t * 2
            self.gap_t = math.floor((self.whole_t - patch_t2) * 0.4)

            if self.gap_t < self.patch_t:
                self.gap_t = 0
            else:
                self.gap_t = self.patch_t
            # Number of patches along temporal dimension           
            if (self.whole_t - patch_t2 + self.gap_t == 0) | (self.gap_t == 0) :
                flagg = 1
            else:
                flagg = int((self.whole_t - patch_t2 + self.gap_t) / self.gap_t)
            # Partitioning
            for x in range(0, int((self.whole_y - self.patch_y + self.gap_y) / self.gap_y)):
                for y in range(0, int((self.whole_x - self.patch_x + self.gap_x) / self.gap_x)):
                    for z in range(0, flagg):
                        single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                        init_h = self.gap_y * x
                        end_h = self.gap_y * x + self.patch_y
                        init_w = self.gap_x * y
                        end_w = self.gap_x * y + self.patch_x
                        init_s = self.gap_t * z
                        end_s = self.gap_t * z + patch_t2
                        single_coordinate['init_h'] = init_h
                        single_coordinate['end_h'] = end_h
                        single_coordinate['init_w'] = init_w
                        single_coordinate['end_w'] = end_w
                        single_coordinate['init_s'] = init_s
                        single_coordinate['end_s'] = end_s
                        # Set a unique name to each patch
                        patch_name = self.datasets_name + '_' + im_name.replace('.tif', '') + '_x' + str(
                            x) + '_y' + str(y) + '_z' + str(z)
                        self.name_list.append(patch_name)
                        self.coordinate_list[patch_name] = single_coordinate
                        self.stack_index.append(ind)
            ind = ind + 1

    def save_yaml_train(self):
        """
        Save some essential params in para.yaml.

        """
        yaml_name = self.pth_path + '//para.yaml'
        para = {'n_epochs': 0, 'datasets_path': 0, 'overlap_factor': 0,
                'pth_path': 0, 'GPU': 0, 'batch_size': 0,
                'patch_x': 0, 'patch_y': 0, 'patch_t': 0, 'gap_y': 0, 'gap_x': 0,
                'gap_t': 0, 'lr': 0, 'b1': 0, 'b2': 0, 'fmap': 0}
        para["n_epochs"] = self.n_epochs
        para["datasets_path"] = self.datasets_path
        para["pth_path"] = self.pth_path
        para["GPU"] = self.GPU
        para["batch_size"] = self.batch_size
        para["patch_x"] = self.patch_x
        para["patch_y"] = self.patch_y
        para["patch_t"] = self.patch_t
        para["gap_x"] = self.gap_x
        para["gap_y"] = self.gap_y
        para["gap_t"] = self.gap_t
        para["lr"] = self.lr
        para["b1"] = self.b1
        para["b2"] = self.b2
        para["fmap"] = self.fmap
        para["overlap_factor"] = self.overlap_factor
        with open(yaml_name, 'w') as f:
            yaml.dump(para, f)

    def distribute_GPU(self):
        """
        Allocate the GPU for the training program. Print the using GPU information to the screen.
        For acceleration, multiple GPUs parallel training is recommended.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU)
        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()
            self.local_model = nn.DataParallel(self.local_model, device_ids=range(self.ngpu))
            print('\033[1;31mUsing {} GPU(s) for training -----> \033[0m'.format(torch.cuda.device_count()))

    def train(self):

        """
        Pytorch training workflow
        """
        # ---- Optimizer & LR scheduler ------------------------------------------        
        optimizer_G = torch.optim.AdamW(self.local_model.parameters(), 
                                        lr=self.lr, 
                                        betas=(0.9, 0.999), 
                                        weight_decay=1e-4)
        # Cosine annealing over many steps
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer_G, 
                                                   T_max=2000000, 
                                                   eta_min=1e-6)

        # ---- Timers for ETA reporting ------------------------------------------
        prev_time = time.time()
        time_start = time.time()
        
        # ---- Loss functions (moved to GPU) -------------------------------------
        L1_pixelwise = torch.nn.SmoothL1Loss()
        L2_pixelwise = torch.nn.MSELoss()
        L2_pixelwise.cuda()
        L1_pixelwise.cuda()

        # ---- Optional resume from checkpoint -----------------------------------
        if self.continue_train == 1:
            checkpoint_path = self.checkpoint_path
            try:
                start_epoch = self.load_checkpoint(self.local_model, optimizer_G, scheduler, checkpoint_path)
            except FileNotFoundError:
                print("No checkpoint found, starting from scratch.")
        else:
            start_epoch = 0
            print("train from the beginning !")
        # ======================= EPOCH LOOP =====================================
        for epoch in range(start_epoch, self.n_epochs):
                
            # ---- Randomly shuffle frames inside each 3D stack -------------------
            # This creates different temporal combinations each epoch for data augmentation.
            shuffled_noise_im_all = []
            for arr in self.noise_im_all:
                if arr.shape[0] > 1:  
                    shuffled_indices = np.random.permutation(arr.shape[0])  
                    shuffled_arr = arr[shuffled_indices, :, :]
                else:
                    shuffled_arr = arr      
                shuffled_noise_im_all.append(shuffled_arr)
            # ---- Build dataset & dataloader ------------------------------------
            filtered_names, filtered_stack_index = filter_samples(self.name_list, 
                                                                  self.coordinate_list, 
                                                                  shuffled_noise_im_all, 
                                                                  self.stack_index,
                                                                  zero_threshold=0.20, 
                                                                  verbose=True)       
            
            train_data = trainset(filtered_names, 
                                  self.coordinate_list, 
                                  shuffled_noise_im_all, 
                                  filtered_stack_index)
            
            trainloader = DataLoader(train_data, 
                                     batch_size=self.batch_size, 
                                     shuffle=True, 
                                     num_workers=self.num_workers)
            
            # ================== ITERATION  ======================================
            for iteration, (input, target) in enumerate(trainloader):
                # The input volume and corresponding target volume from data loader to train the deep neural network
                real_A = Variable(input.cuda())
                fake_B = self.local_model(real_A)
                # -- Prepare target with NaN mask ---------------------------------
                real_B = target.cuda()
                real_B[real_B == 0] = torch.nan
                mask_target = (~torch.isnan(real_B)).float()
                # Compute mean
                mean_real_B = torch.nanmean(real_B, 2)
                mask_target_mean = (~torch.isnan(mean_real_B)).float()
                mean_real_B = torch.nan_to_num(mean_real_B, nan=0.0)
                real_B = torch.nan_to_num(real_B, nan=0.0)
                # Mean of prediction across time
                mean_fake_B = torch.mean(fake_B,2)
                mean_fake_B = torch.clamp(mean_fake_B, max=mean_real_B.max())
                
                # Mask the bad pixels for training
                if self.mask_train == 0:
                    mask_target = 1  
                    mask_target_mean = 1                       
                

                # ---- Compute losses --------------------------------------------
                L1_loss_stack = L1_pixelwise(fake_B * mask_target , real_B * mask_target)  * 1e6 * 0.125
                L2_loss_mean = L2_pixelwise(mean_fake_B * mask_target_mean, mean_real_B * mask_target_mean)  * 1e6                               
                # ---- Backprop & optimization -----------------------------------
                optimizer_G.zero_grad()  
                Total_loss = L2_loss_mean + L1_loss_stack 
                Total_loss.backward()
                optimizer_G.step()
                scheduler.step()
                # ---- ETA / logging ---------------------------------------------
                batches_done = epoch * len(trainloader) + iteration
                batches_left = self.n_epochs * len(trainloader) - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
                prev_time = time.time()
                # Console progress line
                if iteration % 1 == 0:
                    time_end = time.time()
                    print(
                        '\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f, L1 Loss stack: %.2f, L2 Loss mean: %.2f, [ETA: %s] [Time cost: %.2d s]     '  
                        % (
                            epoch + 1,
                            self.n_epochs,
                            iteration + 1,
                            len(trainloader),
                            Total_loss.item(),
                            L1_loss_stack.item(),
                            L2_loss_mean.item(),
                            time_left,
                            time_end - time_start
                        ), 
                        end=' '
                    )
                # Log to SwanLab
                swanlab.log({"Total_loss": Total_loss, 
                             "L1 loss stack": L1_loss_stack, 
                             "L2 loss mean": L2_loss_mean})
                # ---- End-of-epoch checkpoint -----------------------------------
                if (iteration + 1) % (len(trainloader)) == 0:
                    print('\n', end=' ')
                    # Save model at the end of every epoch
                    self.save_model(epoch, optimizer_G,scheduler, iteration)

        print('Training finished. All models saved.')
        swanlab.finish()

    def save_model(self, epoch, optimizer,scheduler, iteration):  
        model_save_name = self.pth_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(iteration + 1).zfill(4) + '.pth'
        if isinstance(self.local_model, nn.DataParallel):
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.local_model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            }
        else:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.local_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            }
        torch.save(checkpoint, model_save_name)          