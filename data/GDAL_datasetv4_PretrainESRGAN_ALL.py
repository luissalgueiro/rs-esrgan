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
        self.Patch_Size = 128 #self.opt['HR_size']
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
        " get the samples"

        if self.opt["phase"] == "train":
            Patch_Size = 128 #self.opt['HR_size']

            ### EU-Cities ###
            dir_file = str(self.list_IDs.iloc[index, 1])
            if 'file' in dir_file:
                ## eu_cities ##
                print('Entro en EU CITIES')
                dir_file_lr = dir_file #str(self.list_IDs.iloc[index, 1])
                print('dirfile: ', dir_file)
                dir_file_hr = dir_file.replace('LR', 'HR')
                lr_array = np.load(self.paths_LR + "/" + dir_file_lr )
                hr_array = np.load(self.paths_LR + "/" + dir_file_hr )
                ## LR shape: 200,200,4
                ## HR shape: 1000, 1000, 4
                assert lr_array is not None
                assert hr_array is not None
                # print(lr_array.shape)
                # print(hr_array.shape)

                ## RESIZE LR
                dim = (hr_array.shape[0], hr_array.shape[1])
                lr_array = cv2.resize(lr_array, dim, interpolation=cv2.INTER_CUBIC)
                ## LR shape 1000,1000,4

                #cambia a formato CHW
                lr_array = np.transpose(lr_array,(2,0,1))
                hr_array = np.transpose(hr_array,(2,0,1))
                hr_array = hr_array*32767.0

                # if self.stand==True:
                    # assuming format CHW
                lr_array_mean = lr_array.mean(axis=(1,2), keepdims=True)
                lr_array_stddev = lr_array.std(axis=(1, 2), keepdims=True)

                hr_array_mean = hr_array.mean(axis=(1, 2), keepdims=True)
                hr_array_stddev = hr_array.std(axis=(1, 2), keepdims=True)

                lr_array = (lr_array - lr_array_mean)/lr_array_stddev
                hr_array = (hr_array - hr_array_mean) / hr_array_stddev
                # print("DATA STANDARIZED")

                # assert lr_array is not None
                # assert hr_array is not None
                
                ####===  Cropping ===##  # Format assume CHW
                # print('Shape lr', lr_array.shape)
                # print('')

                cc,hh,ww = lr_array.shape
                # print('hh,ww'.format(hh,ww))
                idx = np.random.randint(0, hh - self.Patch_Size)
                idy = np.random.randint(0, ww - self.Patch_Size)
                
                hr = hr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
                lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
                
                augmentation = util.Data_augm(p=0.6)
                lr, hr = augmentation(lr, hr)  # assumes format "HWC"

                # lr, hr = util.augment([lr, hr], self.opt['use_flip'], self.opt['use_rot'])
                hr = torch.from_numpy(hr[(2,1,0,3),:,:].copy()).float()  # solo escoge ch RGB+NIR
                lr = torch.from_numpy(lr[(2,1,0,3),:,:].copy()).float()

                sample = {"LR": lr, "LR_mean":lr_array_mean, "LR_stddev":lr_array_stddev,
                      "HR": hr, "HR_mean":hr_array_mean, "HR_stddev": hr_array_stddev}
                return sample 
                            
            else:
                ## WV_Sent ##
                print('Entro en WV_Sent')
                dir_file_lr = str(self.list_IDs.iloc[index, 1])
                dir_file_hr = dir_file_lr.replace('LR', 'HR')
                dir_file_hr = dir_file_hr.replace('/sent_roi', '/wv_roi')

                lr_array = np.load(self.paths_LR + "/" + dir_file_lr )
                hr_array = np.load(self.paths_LR + "/" + dir_file_hr )
                assert lr_array is not None
                assert hr_array is not None
                # print(lr_array.shape)
                # print(hr_array.shape)

                ## LR shape: 4,140,140
                ## HR shape: 4, 140, 140

                ## CAMBIA a formato CHW
                # lr_array = np.transpose(lr_array,(2,0,1))
                # hr_array = np.transpose(hr_array,(2,0,1))

                # if self.stand==True:
                    # assuming format CHW
                lr_array_mean = lr_array.mean(axis=(1,2), keepdims=True)
                lr_array_stddev = lr_array.std(axis=(1, 2), keepdims=True)

                hr_array_mean = hr_array.mean(axis=(1, 2), keepdims=True)
                hr_array_stddev = hr_array.std(axis=(1, 2), keepdims=True)

                lr_array = (lr_array - lr_array_mean)/lr_array_stddev
                hr_array = (hr_array - hr_array_mean) / hr_array_stddev
                print("DATA STANDARIZED")

                assert lr_array is not None
                assert hr_array is not None


                # print("Patch size: ", self.Patch_Size)
                # print('lr_array shape', lr_array.shape)
                


                ####===  Cropping ===##  # Format assume CHW
                cc,hh,ww = lr_array.shape
                # print("HH , Ww: ".format(hh, ww))
                idx = 2  #np.random.randint(0, hh - self.Patch_Size)
                idy = 2 #np.random.randint(0, ww - self.Patch_Size)
                
                hr = hr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
                lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
                lr, hr = util.augment([lr, hr], self.opt['use_flip'], self.opt['use_rot'])
                hr = torch.from_numpy(hr[(2,1,0,3),:,:].copy()).float()  # solo escoge ch RGB+NIR
                lr = torch.from_numpy(lr[(2,1,0,3),:,:].copy()).float()

                sample = {"LR": lr, "LR_mean":lr_array_mean, "LR_stddev":lr_array_stddev,
                      "HR": hr, "HR_mean":hr_array_mean, "HR_stddev": hr_array_stddev}
                return sample 
                 
#######################################################################################################
        ### VALIDATION ###
###################################################################33
        else:  ### VALIDATION ###
            ## WV_Sent ##
            Patch_Size = self.opt['HR_size']
            dir_file_lr = str(self.list_IDs.iloc[index, 1])
            dir_file_hr = dir_file_lr.replace('LR', 'HR')
            dir_file_hr = dir_file_hr.replace('/sent_roi', '/wv_roi')

            lr_array = np.load(self.paths_LR + "/" + dir_file_lr )
            hr_array = np.load(self.paths_LR + "/" + dir_file_hr )

            HR_path = self.paths_HR+"/" + dir_file_hr.replace('.npy','') #str(self.list_IDs.iloc[index, 1]).replace(".npy","")
            LR_path = self.paths_LR+"/" + dir_file_lr.replace('.npy','') #str(self.list_IDs.iloc[index, 1]).replace(".npy","")

            ## LR shape: 4,140,140
            ## HR shape: 4,140,140

            ## CAMBIA a formato CHW
            # lr_array = np.transpose(lr_array,(2,0,1))
            # hr_array = np.transpose(hr_array,(2,0,1))

            # if self.stand==True:
                # assuming format CHW
            lr_array_mean = lr_array.mean(axis=(1,2), keepdims=True)
            lr_array_stddev = lr_array.std(axis=(1, 2), keepdims=True)

            hr_array_mean = hr_array.mean(axis=(1, 2), keepdims=True)
            hr_array_stddev = hr_array.std(axis=(1, 2), keepdims=True)

            lr_array = (lr_array - lr_array_mean)/lr_array_stddev
            hr_array = (hr_array - hr_array_mean) / hr_array_stddev
            print("DATA STANDARIZED")
            
            hr = torch.from_numpy(hr_array[(2,1,0,3),:,:].copy()).float()  # solo escoge ch RGB+NIR
            lr = torch.from_numpy(lr_array[(2,1,0,3),:,:].copy()).float()

            
            # sample = {"LR": lr, "LR_mean":lr_array_mean, "LR_stddev":lr_array_stddev,
            #         "HR": hr, "HR_mean":hr_array_mean, "HR_stddev": hr_array_stddev,
            #         'LR_path': LR_path, 'HR_path': HR_path}
            sample = {"LR": lr, "LR_mean":lr_array_mean, "LR_std":lr_array_stddev,
                      "HR": hr, "HR_mean":hr_array_mean, "HR_std": hr_array_stddev, 
                      'LR_path': LR_path, 'HR_path': HR_path}


            return sample 
       

    def __len__(self):
        # print("EJecuta LEN")
        "Denotes the total number of samples"
        return len(self.list_IDs)
