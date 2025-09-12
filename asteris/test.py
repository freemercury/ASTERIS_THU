import os
import sys
import numpy as np
import yaml
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from asteris.utils import restore_fits
from .ASTERIS_net_8 import ASTERIS8
from .ASTERIS_net_4 import ASTERIS4
from .data_process import test_preprocess, testset, multibatch_test_save, singlebatch_test_save
from tqdm import tqdm

class testing_class():
    """
    Class implementing testing process
    """

    def __init__(self, params_dict):
        """
        Constructor class for testing process

        Args:
           params_dict: dict
               The collection of testing params set by users
        Returns:
           self

        """
        self.restore_clip_part = False
        self.overlap_factor = 0.1
        self.datasets_path = ''
        self.test_datasize = 8
        self.fmap = 24
        self.output_dir = ''
        self.pth_dir = ''
        self.batch_size = 1
        self.patch_t = 8
        self.patch_x = 128
        self.patch_y = 128
        self.random_flag = 0
        self.GPU = '0'
        self.ngpu = 1
        self.num_workers = 0
        self.denoise_model = ''
        self.result_display = ''
        self.scale_factor = 4
        self.set_params(params_dict)

    def run(self):
        """
        General function for testing ASTERIS network.

        """
        # create some essential file for result storage
        self.prepare_file()
        # get models for processing
        self.read_modellist()
        # get stacks for processing
        self.read_imglist()
        # save some essential testing parameters in para.yaml
        self.save_yaml_test()
        # initialize denoise network with testing parameters.
        self.initialize_network()
        # specifies the GPU for the testing program.
        self.distribute_GPU()
        # start testing and result visualization during testing period (optional)
        self.test()
        


    def prepare_file(self):
        """
        Make data folder to store testing results
        Important Fields:
            self.datasets_name: the sub folder of the dataset
            self.pth_path: the folder for pth file storage

        """
        # Take the dir name
        if self.datasets_path[-1]!='/':
           self.datasets_name=self.datasets_path.split("/")[-1]
        else:
           self.datasets_name=self.datasets_path.split("/")[-2]

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        # ---- Create subfolder for this dataset + model combination -------------
        self.output_path = (
            self.output_dir + '/' + 
            'Data_' + self.datasets_name + 
            '_Model_' + self.denoise_model
        )
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def set_params(self, params_dict):
        """
        Set the params set by user to the testing class object and calculate some default parameters for testing

        """
        # ---- Update object attributes from user-provided dictionary ------------
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # ---- Compute patch strides ("gaps") in each dimension ------------------
        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))  # patch gap in x
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))  # patch gap in y
        self.gap_t = int(self.patch_t)  # patch gap in t
        
        self.ngpu = str(self.GPU).count(',') + 1  # check the number of GPU used for testing
        self.batch_size = self.ngpu * self.batch_size    # By default, the batch size is equal to the number of GPU for minimal memory consumption
        print('\033[1;31mTesting parameters -----> \033[0m')
        print(self.__dict__)

    def read_imglist(self):
        """
        Collect the list of image stacks from the dataset path.
        Stores them in self.img_list for later processing.
        """
        # Grab the list of filenames in the bottom level of dir
        im_folder = self.datasets_path
        self.img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
        self.img_list = [os.path.join(im_folder, img_name) for img_name in self.img_list]
        # Sort the list for deterministic order
        self.img_list.sort()
        print('\033[1;31mStacks for processing -----> \033[0m')
        print('Total stack number -----> ', len(self.img_list))
        

    def read_modellist(self):
        """
        Collect the list of model weight files (.pth) for denoising.
        Stores them in self.model_list for later loading.
        """
        # Grab the list of filenames in the bottom level of dir
        model_path = self.pth_dir + '/' + self.denoise_model
        model_file_list = list(os.walk(model_path, topdown=False))[-1][-1]
        model_list = [item for item in model_file_list if '.pth' in item]
        model_list.sort()
        
        try:
            # Load into object
            self.model_list = model_list
        except Exception as e:
            print('\033[1;31mThere is no .pth file in the models directory! \033[0m')
            sys.exit()
        self.model_list_length = len(model_list) 

    def initialize_network(self):
        """
        Initialize ASTERIS according to the temporal patch size
        Important Fields:
           self.fmap: the number of the feature map in ASTERIS.
           self.local_model: the denoise network

        """
        # ---- Select ASTERIS architecture variant --------------------------------
        if self.patch_t == 4:
            ASTERIS = ASTERIS4
            num_HEAD = [4, 6, 8]
        elif self.patch_t == 8:
            ASTERIS = ASTERIS8
            num_HEAD = [4, 6, 6, 8]
            
        # ---- Instantiate denoising generator network ----------------------------
        denoise_generator = ASTERIS(inp_channels=1, 
                                out_channels=1, 
                                f_maps = self.fmap,
                                num_blocks = num_HEAD,
                                num_refinement_blocks = 4)
        self.local_model = denoise_generator

    def save_yaml_test(self):
        """
        Save some essential params in para.yaml.

        """
        yaml_name = self.output_path + '//para.yaml'
        para = {'datasets_path': 0, 'denoise_model': 0,
                'output_dir': 0, 'pth_dir': 0, 'GPU': 0, 'batch_size': 0,
                'patch_x': 0, 'patch_y': 0, 'patch_t': 0, 'gap_y': 0, 'gap_x': 0,
                'gap_t': 0, 'fmap': 0, 'overlap_factor': 0}
        para["datasets_path"] = self.datasets_path
        para["denoise_model"] = self.denoise_model
        para["output_dir"] = self.output_dir
        para["pth_dir"] = self.pth_dir
        para["GPU"] = self.GPU
        para["batch_size"] = self.batch_size
        para["patch_x"] = self.patch_x
        para["patch_y"] = self.patch_y
        para["patch_t"] = self.patch_t
        para["gap_x"] = self.gap_x
        para["gap_y"] = self.gap_y
        para["gap_t"] = self.gap_t
        para["fmap"] = self.fmap
        para["overlap_factor"] = self.overlap_factor
        with open(yaml_name, 'w') as f:
            yaml.dump(para, f)

    def distribute_GPU(self):
        """
        Allocate the GPU for the testing program. 
        Print the using GPU information to the screen.
        For acceleration, multiple GPUs parallel testing is recommended.

        """
        # ---- Control which GPUs are visible to PyTorch --------------------------
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU)
        
        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()
            
            # Wrap model in DataParallel for multi-GPU usage
            self.local_model = nn.DataParallel(self.local_model, device_ids=range(self.ngpu))
            print('\033[1;31mUsing {} GPU(s) for testing -----> \033[0m'.format(torch.cuda.device_count()))
        self.cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor


    def test(self): 
        """
        Pytorch testing workflow
            - Iterate over all .pth files
            - Run inference over all test stacks
            - Stitch patch outputs back to full stacks
            - Save raw & denoised results
        """        
        pth_count=0

        if any('.pth' in s for s in self.model_list):     
            # Walk through model list in reverse order (latest first, typically)       
            for i in reversed(range(self.model_list_length)):
                pth_count=pth_count+1
                pth_name = self.model_list[i]
                output_path_name = self.output_path + '/' + pth_name.replace('.pth', '') + '/'
                output_path_name_raw = self.output_path + '/raw_' + pth_name.replace('.pth', '') + '/'
                if not os.path.exists(output_path_name):
                    os.mkdir(output_path_name)

                # ----------------------- Load model weights -----------------------
                model_name = self.pth_dir + '/' + self.denoise_model + '//' + pth_name
                checkpoint = torch.load(model_name)
                if isinstance(self.local_model, nn.DataParallel):                  
                    self.local_model.module.load_state_dict(checkpoint["model_state_dict"])
                    self.local_model.eval()                  
                else:
                    self.local_model.load_state_dict(checkpoint["model_state_dict"])
                    self.local_model.eval()  
                
                self.local_model.cuda()
                self.print_img_name = False
                # ----------------------- Build test dataset -----------------------
                print("Testing the " , i ,"th model:")
                name_list, noise_imgs, coordinate_list, test_im_names, img_means = test_preprocess(self)
                test_data = testset(name_list, coordinate_list, noise_imgs)
                testloader = DataLoader(test_data, 
                                        batch_size=self.batch_size, 
                                        shuffle=False,
                                        num_workers=self.num_workers)
                
                outputs = []
                start_time = time.time()
                # ----------------------- Inference loop ---------------------------
                with tqdm(total=len(testloader), desc=f"[Model {pth_count}/{self.model_list_length}, {pth_name}]", leave=False) as pbar:
                    for iteration, (img_ids, noise_patchs, coordinates, mean_vals) in enumerate(testloader):
                        noise_patchs = noise_patchs.cuda()
                        mean_vals = mean_vals.cuda()                       
                        real_A = noise_patchs
                        # Input volume to the model
                        real_A = Variable(real_A)
                        with torch.no_grad():
                            # Forward pass
                            fake_B = self.local_model(real_A)
                            # Restore original intensity scale
                            fake_B = fake_B + mean_vals.view(fake_B.shape[0],1,1,1,1)
                            real_A = real_A + mean_vals.view(real_A.shape[0],1,1,1,1)

                        # Prepare for stitching
                        output_imgs = np.squeeze(fake_B.cpu().detach().numpy())
                        raw_imgs = np.squeeze(real_A.cpu().detach().numpy())
                        outputs.append({
                            'output_imgs': output_imgs,
                            'raw_imgs': raw_imgs,
                            'img_ids': img_ids,
                            'coordinates': coordinates
                        })
                        
                        pbar.update(1)
                # Print Inference Time
                total_time = time.time() - start_time
                print(f"Average inference time: {total_time:.3f} seconds")
                        
                # ----------------------- Allocate stitch buffers -------------------
                denoise_imgs = [np.zeros_like(noise_img) for noise_img in noise_imgs]
                input_imgs = [np.zeros_like(input_img) for input_img in noise_imgs]
                # ----------------------- Stitch patches back ----------------------- 
                for output in outputs:
                    output_imgs = output['output_imgs']
                    raw_imgs = output['raw_imgs']
                    img_ids = output['img_ids']
                    coordinates = output['coordinates']

                    # If multi-sample batch (ndim != 3), iterate per-sample
                    if output_imgs.ndim != 3:
                        for i, N in enumerate(img_ids):
                            # Compute the patch placement within the full stack
                            (output_patch, raw_patch, 
                             stack_start_w, stack_end_w, 
                             stack_start_h, stack_end_h, 
                             stack_start_s, stack_end_s) = multibatch_test_save(
                                coordinates, i, output_imgs, raw_imgs)
                            # Restore global mean 
                            raw_patch=raw_patch+img_means[N]
                            output_patch=output_patch+img_means[N]
                            # Write patch into the correct region
                            denoise_imgs[N][stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = output_patch
                            input_imgs[N][stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = raw_patch
                    # Single-sample batch
                    else:
                        N = img_ids
                        output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                            coordinates, output_imgs, raw_imgs)
                                                    
                        raw_patch=raw_patch+img_means[N]
                        output_patch=output_patch+img_means[N]
                        
                        denoise_imgs[N][stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = output_patch
                        input_imgs[N][stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = raw_patch
                            
                # ----------------------- Save outputs ------------------------------
                print("Saving Image...")
                for N in tqdm(range(len(self.img_list))):
                    # Stitching finish
                    output_img = denoise_imgs[N].squeeze().astype(np.float32)
                    # Nan-masking
                    mask_nan = np.where(noise_imgs[N] == 0, np.nan, 1)
                    # Taking temporal average
                    input_img_single = np.nanmean(mask_nan * input_imgs[N],0)
                    output_img_single = np.nanmean(mask_nan * output_img,0)
                    # Restore raw image to .fits format         
                    restore_fits(self.scale_factor,self.restore_clip_part, input_img_single, input_img_single, 
                                 self.datasets_path.replace('/images_for_test/','/reference_files/'), 
                                 test_im_names[N].replace(f'_test_im_mean{self.patch_t}.tif',''), 
                                 output_path_name_raw)
                    # Restore denoised image to .fits format 
                    restore_fits(self.scale_factor,self.restore_clip_part, output_img_single, input_img_single, 
                                 self.datasets_path.replace('/images_for_test/','/reference_files/'),
                                 test_im_names[N].replace(f'_test_im_mean{self.patch_t}.tif',''), 
                                 output_path_name)                   
                       
        print('Testing finished. All results saved.')