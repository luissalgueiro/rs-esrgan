import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import pandas as pd
import data.util as util

from skimage.transform import rescale, resize, downscale_local_mean
from scipy import ndimage

class GdalDataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''
    def __init__(self, opt, transform=None):
        "Initialization"
        self.opt = opt
        # print("Type opt: ", type(opt))
        # print(" opt: ",opt)
        self.lr_array = None
        self.hr_array = None
        self.transform = transform
        self.lr_bands = None
        self.hr_bands = None
        self.lr = None
        self.hr = None
        self.Patch_Size = self.opt['HR_size']
        self.seed= 100
        self.scale = self.opt["scale"]
        self.PreUP = self.opt["PreUP"]
        self.LR_down = self.opt["LR_down"]
        self.norm = self.opt["norm"]
        self.stand = self.opt["stand"]
        self.PreUP = self.opt["PreUP"]
        self.up_lr = self.opt["up_lr"] # image LR is downsampled and for PREUP, needed to be upscaled again
        # super(GdalDataset, self).__init__()
        self.paths_LR = self.opt["dataroot_LR"]
        self.paths_HR = self.opt["dataroot_HR"]
        self.HF = self.opt["HF"]
        self.list_IDs = pd.read_csv(self.opt["data_IDs"])
        # # # self.LR_env = None  # environment for lmdb
        # self.HR_env = None
    def __getitem__(self, index):
        # " get the samples"
        if self.opt["phase"] == "train":
            Patch_Size = self.opt['HR_size']
            aux_path_hr = str(self.list_IDs.iloc[index, 1]).replace("LR","HR")
            aux_path_hr = aux_path_hr.replace("sent","wv")
            lr_array = np.load(self.paths_LR + "/" + str(self.list_IDs.iloc[index, 1]))
            hr_array = np.load(self.paths_HR + "/" + aux_path_hr )
            assert lr_array is not None
            assert hr_array is not None
            if lr_array.shape[2]<=4:  # Cambia a formato CHW
                lr_array = np.transpose(lr_array,(2,0,1))
            if hr_array.shape[2]<=4:  # Cambia a formato CHW
                hr_array = np.transpose(hr_array,(2,0,1))
            if self.HF == True:
                lowpass_hr = ndimage.gaussian_filter(hr_array, 3)
                hr_array = hr_array - lowpass_hr
                lowpass_lr = ndimage.gaussian_filter(lr_array, 3)
                lr_array = lr_array - lowpass_lr
            else: 
                lowpass_lr=lowpass_hr=None
            # # # ###===  Normalizing  ===##
            if self.norm:
                # if lr_array.max() > 1.0 or hr_array.max() > 1.0:
                print("Normaliza Dividiendo por el máximo y clipping 0")
                lr_array = lr_array / 32767.0
                hr_array = hr_array / 32767.0
                lr_array = np.clip(lr_array, 0, 1)
                hr_array = np.clip(hr_array, 0, 1)
            # standarization = self.stand
            # print("*******==== hr shape **** ", hr_array.shape)
            # print("******* ===lr shape **** ", lr_array.shape)
            if self.stand==True:
                # assuming format CHW
                # print("*******==== hr shape **** ", hr_array.shape)
                # print("******* ===lr shape **** ", lr_array.shape)
                lr_array_mean = lr_array.mean(axis=(1,2), keepdims=True)
                lr_array_stddev = lr_array.std(axis=(1, 2), keepdims=True)
                hr_array_mean = hr_array.mean(axis=(1, 2), keepdims=True)
                hr_array_stddev = hr_array.std(axis=(1, 2), keepdims=True)
                lr_array = (lr_array - lr_array_mean)/lr_array_stddev
                hr_array = (hr_array - hr_array_mean) / hr_array_stddev
                # # print("DATA STANDARIZED")
            ####===  Cropping ===##  # Format assume CHW
            cc,hh,ww = lr_array.shape
            #print("TTT****HH: %d, WW: %d"%(hh,ww))
            #np.random.seed()
            idx = np.random.randint(0, hh - self.Patch_Size)
            idy = np.random.randint(0, ww - self.Patch_Size)
            hr = hr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
            lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
            # #### DOWNSAMPLING LR ######
            if self.LR_down == True:
                lr = np.transpose(lr, (1, 2, 0))  # formato HWC
                # lr = resize(lr, (lr.shape[0] / self.scale, lr.shape[1] / self.scale), mode="symmetric",
                #             anti_aliasing=True)
                lr = rescale(lr, 0.2,anti_aliasing=True, mode='reflect')
                lr = np.transpose(lr, (2, 0, 1))  # formato CHW
            ### augmentation - flip, rotate
            lr, hr = util.augment([lr, hr], self.opt['use_flip'], self.opt['use_rot'])
            ### change  Channels ####
            hr = torch.from_numpy(hr[(2,1,0,3),:,:].copy()).float()  # solo escoge ch RGB
            lr = torch.from_numpy(lr[(2,1,0,3),:,:].copy()).float()
            #############################
            if self.stand == True:
                sample = {"LR": lr, "LR_mean":lr_array_mean, "LR_stddev":lr_array_stddev,
                      "HR": hr, "HR_mean":hr_array_mean, "HR_stddev": hr_array_stddev}
            else:
                sample = {"LR": lr,"HR": hr}
            return sample
#######################################################################################################
        ### VALIDATION ###
###################################################################33
        else:  ### VALIDATION ###
            self.Patch_Size = self.opt['HR_size']
            seed=100
            self.PreUP = self.opt["PreUP"]
            self.scale = self.opt["scale"]
            self.LR_down = self.opt["LR_down"]
            self.norm = self.opt["norm"]
            self.stand = self.opt["stand"]
            self.up_lr = self.opt["up_lr"] # image LR is downsampled and for PREUP, needed to be upscaled again
            self.HF = self.opt["HF"]
            aux_path_hr = str(self.list_IDs.iloc[index, 1]).replace("LR","HR")
            aux_path_hr = aux_path_hr.replace("sent","wv")
            LR_path = self.paths_LR+"/" + str(self.list_IDs.iloc[index, 1]).replace(".npy","")
            HR_path = self.paths_HR+"/" + aux_path_hr
            lr_array = np.load(self.paths_LR + "/" + str(self.list_IDs.iloc[index, 1])).astype(np.float32())
            hr_array = np.load(self.paths_HR + "/" + aux_path_hr).astype(np.float32())
            if lr_array.shape[2]<=4:  # Cambia a formato CHW
                lr_array = np.transpose(lr_array,(2,0,1))
            if hr_array.shape[2]<=4:  # Cambia a formato CHW
                hr_array = np.transpose(hr_array,(2,0,1))
            
            # #####===  Normalizing  ===##
            if self.norm:
            # if lr_array.max() > 1.0 or hr_array.max() > 1.0:
                # print("Normalize Dividing by max and clipping 0")
                lr_array = lr_array / 32767.0
                hr_array = hr_array / 32767.0
                lr_array = np.clip(lr_array, 0, 1)
                hr_array = np.clip(hr_array, 0, 1)
            # standarization = self.stand
            if self.stand == True:
                # assuming format CHW
                lr_array_mean = lr_array.mean(axis=(1, 2), keepdims=True)
                lr_array_stddev = lr_array.std(axis=(1, 2), keepdims=True)
                hr_array_mean = hr_array.mean(axis=(1, 2), keepdims=True)
                hr_array_stddev = hr_array.std(axis=(1, 2), keepdims=True)
                lr_array = (lr_array - lr_array_mean) / lr_array_stddev
                hr_array = (hr_array - hr_array_mean) / hr_array_stddev
                # print("DATA STANDARIZED")
            ###===  Cropping ===##  # Format assume CHW
            cc, hh, ww = lr_array.shape
            idx = np.random.randint(0, hh - self.Patch_Size-1)
            idy = np.random.randint(0, ww - self.Patch_Size-1)
            hr = hr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
            lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
            # #### DOWNSAMPLING LR ######
            if self.LR_down == True:
                lr = np.transpose(lr, (1, 2, 0))  # formato HWC
                lr = rescale(lr, 0.2,anti_aliasing=True, mode='reflect')
                lr = np.transpose(lr, (2, 0, 1))  # formato CHW

            if self.HF == True:
                lowpass_hr = ndimage.gaussian_filter(hr, 3)
                hr = hr - lowpass_hr
                lowpass_lr = ndimage.gaussian_filter(lr, 3)
                lr = lr - lowpass_lr
                lowpass_hr = torch.from_numpy(lowpass_hr[(2,1,0,3),:,:]).float()
                lowpass_lr = torch.from_numpy(lowpass_lr[(2,1,0,3),:,:]).float()
            else: 
                lowpass_lr=lowpass_hr=0.0
            ### change channels  ####

            hr = torch.from_numpy(hr[(2,1,0,3),:,:]).float()
            lr = torch.from_numpy(lr[(2,1,0,3),:,:]).float()
            if self.stand == True:
                sample = {"LR": lr, "LR_mean":lr_array_mean, "LR_std":lr_array_stddev,
                      "HR": hr, "HR_mean":hr_array_mean, "HR_std": hr_array_stddev, 
                      'LR_path': LR_path, 'HR_path': HR_path}
            else:
                 sample = {"LR": lr, "HR": hr, 'LR_path': LR_path, 'HR_path': HR_path,
                "HR_min": hr_array.min(), 'HR_max': hr_array.max(), 'LR_min': lr_array.min(),
                 'LR_max': lr_array.max(), 'lowpass_lr': lowpass_lr, 'lowpass_hr': lowpass_hr
                  }
            return sample


    def __len__(self):
        return len(self.list_IDs)
