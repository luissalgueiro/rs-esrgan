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
from sklearn.preprocessing import StandardScaler as StandardScaler


# import gdal


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
        self.up_lr = self.opt["up_lr"] # image LR is downsampled and for PREUP, needed to be upscaled again
        # super(GdalDataset, self).__init__()
        self.HF = self.opt["HF"]

        self.paths_LR = self.opt["dataroot_LR"]
        self.paths_HR = self.opt["dataroot_HR"]

        self.list_IDs = pd.read_csv(self.opt["data_IDs"])
        # self.LR_env = None  # environment for lmdb
        # self.HR_env = None
    def __getitem__(self, index):
        " get the samples"

        if self.opt["phase"] == "train":
            self.Patch_Size = self.opt['HR_size']
            self.up_lr = self.opt["up_lr"]

            lr_array = np.load(self.paths_LR + "/" + str(self.list_IDs.iloc[index, 1]))
            hr_array = np.load(self.paths_HR + "/" + str(self.list_IDs.iloc[index, 1]))
            # print("PATH LR: ", self.paths_LR + "/" + str(self.list_IDs.iloc[index, 1]))
            # print("PATH HR: ", self.paths_HR + "/" + str(self.list_IDs.iloc[index, 1]))
            # print("[DEBUG...] Cargo como tipo: ", lr_array.dtype)
            if lr_array.shape[2]<=4:  # Cambia a formato CHW
                lr_array = np.transpose(lr_array,(2,0,1))
            if hr_array.shape[2]<=4:  # Cambia a formato CHW
                hr_array = np.transpose(hr_array,(2,0,1))
            
            print("Shape: ", lr_array.shape)
            print("Shape: ", hr_array.shape)


            if self.HF == True:
                lowpass_hr = ndimage.gaussian_filter(hr_array, 3)
                hr_array = hr_array - lowpass_hr

                lowpass_lr = ndimage.gaussian_filter(lr_array, 3)
                lr_array = lr_array - lowpass_lr
            else: 
                lowpass_lr=lowpass_hr=None



            ###===  Normalizing  ===##
            # if lr_array.max() > 1.0 or hr_array.max() > 1.0:
            if self.norm==True:
                print("Normaliza Dividiendo por el máximo y clipping 0")
                lr_array = lr_array / 32767.0
                # hr_array = hr_array / 32767.0
                lr_array = np.clip(lr_array, 0, 1)
                # hr_array = np.clip(hr_array, 0, 1)

            
            if self.up_lr == True:
                # print("DATA__SHAPE GT: ", hr_array.shape)
                dim22 = (hr_array.shape[2], hr_array.shape[1])
                # print("dim22: ", dim22)
                # print("SHAPE Lr: ", lr_array.dtype)
                lr_array = np.transpose(np.float32(lr_array), (1, 2, 0))
                lr_array = cv2.resize(lr_array, dim22, interpolation=cv2.INTER_CUBIC)
                lr_array = np.transpose(lr_array, (2, 0, 1))
                # print("SHAPE LR RESIZED: ", lr_array.shape)
                # lr = lr_resized

            # standarization = self.stand
            if self.stand==True:
                # assuming format CHW
                # lr_array = lr_array*32767.0
                hr_array = hr_array * 32767.0

                ## #version 1  ###3
                lr_array_mean = lr_array.mean(axis=(1,2), keepdims=True)
                lr_array_stddev = lr_array.std(axis=(1, 2), keepdims=True)
                hr_array_mean = hr_array.mean(axis=(1, 2), keepdims=True)
                hr_array_stddev = hr_array.std(axis=(1, 2), keepdims=True)
                lr_array = (lr_array - lr_array_mean)/lr_array_stddev
                hr_array = (hr_array - hr_array_mean) / hr_array_stddev

                # ### version 2 ####
                # lr_stand, hr_stand = [], []
                # scaler= StandardScaler()
                # ## ASUMMING AN ARRAY OF SHAPE [h,w,c]
                # for ii in range(hr_array.shape[-1]):
                #     lr_array_mean = lr_array.mean(axis=(1,2), keepdims=True)
                #     lr_array_stddev = lr_array.std(axis=(1, 2), keepdims=True)
                #     hr_array_mean = hr_array.mean(axis=(1, 2), keepdims=True)
                #     hr_array_stddev = hr_array.std(axis=(1, 2), keepdims=True)

                #     lr_stand.append(scaler.fit_transform( lr_array[:,:,ii]) )
                #     hr_stand.append(scaler.fit_transform( hr_array[:,:,ii]) )
                
                # lr_stand=np.asarray(lr_stand) # ahora esta en formato [C,H,W]
                # lr_stand=np.asarray(lr_stand)
                # lr_array = np.transpose(lr_stand,(1,2,0)) # vuelta a formato [H,W,C]
                # hr_array = np.transpose(lr_stand,(1,2,0))

                # print("STATS LR mean :", lr_array.mean(axis=(1,2)))
                # print("STATS LR std :", lr_array.std(axis=(1,2)) )

                # print("STATS HR mean :", hr_array.mean(axis=(1,2)))
                # print("STATS HR std :", hr_array.std(axis=(1,2)) )


                # print("STATS HR: [%f, %f] "%( hr_array.mean(axis=(1,2)), hr_array.std(axis=(1,2))   ))
                print("DATA STANDARIZED")


            ##===  Cropping ===##  # Format assume CHW
            cc,hh,ww = lr_array.shape
            np.random.seed()
            idx = np.random.randint(0, hh - self.Patch_Size)
            idy = np.random.randint(0, ww - self.Patch_Size)
            np.random.seed(self.seed)

            
            if self.up_lr == True:
                lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
                hr = hr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
            else:
                lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float16()) # Formato CHW
                hr = hr_array[:, idx*self.opt["scale"]:idx*self.opt["scale"] + self.Patch_Size*self.opt["scale"], idy*self.opt["scale"]:idy*self.opt["scale"] + self.Patch_Size*self.opt["scale"]].astype(np.float16()) # Formato CHW


            # #### DOWNSAMPLING LR ######
            if self.LR_down == True:
                print("Downloading...")
                lr = np.transpose(lr, (1, 2, 0))  # formato HWC
                # lr = resize(lr, (lr.shape[0] / self.scale, lr.shape[1] / self.scale), mode="edge",
                #             anti_aliasing=True)
                lr = rescale(lr, 0.2,anti_aliasing=True)
                lr = np.transpose(lr, (2, 0, 1))  # formato CHW
            
            
            ### augmentation - flip, rotate
            lr, hr = util.augment([lr, hr], self.opt['use_flip'], self.opt['use_rot'])

            hr = torch.from_numpy(hr[(2,1,0,3),:,:].copy()).float()
            lr = torch.from_numpy(lr[(2,1,0,3),:,:].copy()).float()
           


            if self.stand == True:
                sample = {"LR": lr, "LR_mean":lr_array_mean, "LR_std":lr_array_stddev,
                      "HR": hr, "HR_mean":hr_array_mean, "HR_std": hr_array_stddev}
            else:
                sample = {"LR": lr,"HR": hr}

            return sample
################################################################################################################
        else:
            # print("REALIZA else.....")
            self.Patch_Size = self.opt['HR_size']
            self.PreUP = self.opt["PreUP"]
            self.LR_down = self.opt["LR_down"]
            self.up_lr = self.opt["up_lr"]
            self.norm = self.opt["norm"]
            self.scale=self.opt["scale"]
            self.stand = self.opt["stand"]
            self.HF = self.opt["HF"]


            seed=100
            # print("Paths VAL....", self.paths_LR + "/" + str(self.list_IDs.iloc[index, 1]))
            HR_path = self.paths_HR+"/" + str(self.list_IDs.iloc[index, 1]).replace(".npy","")
            LR_path = self.paths_LR+"/" + str(self.list_IDs.iloc[index, 1]).replace(".npy","")


            # img_LR = util.read_img(self.LR_env, LR_path)
            lr_array = np.load(self.paths_LR + "/" + str(self.list_IDs.iloc[index, 1])).astype(np.float32())
            hr_array = np.load(self.paths_HR + "/" + str(self.list_IDs.iloc[index, 1])).astype(np.float32())

            if lr_array.shape[2]<=4:  # Cambia a formato CHW
                lr_array = np.transpose(lr_array,(2,0,1))
            if hr_array.shape[2]<=4:  # Cambia a formato CHW
                hr_array = np.transpose(hr_array,(2,0,1))

            print("Shape: ", lr_array.shape)
            print("Shape: ", hr_array.shape)


            # if self.HF == True:
            #     lowpass_hr = ndimage.gaussian_filter(hr_array, 3)
            #     hr_array = hr_array - lowpass_hr

            #     lowpass_lr = ndimage.gaussian_filter(lr_array, 3)
            #     lr_array = lr_array - lowpass_lr
            # else: 
            #     lowpass_lr=lowpass_hr=0.0

            if self.up_lr == True:

                # print("DATA__SHAPE GT: ", hr_array.shape)
                dim22 = (hr_array.shape[2], hr_array.shape[1])
                # print("dim22: ", dim22)
                # print("SHAPE Lr: ", lr_array.dtype)
                lr_array = np.transpose(np.float32(lr_array),(1,2,0))
                lr_array = cv2.resize(lr_array, dim22, interpolation=cv2.INTER_CUBIC)
                lr_array = np.transpose(lr_array, (2, 0,1))
                # print("SHAPE LR RESIZED: ", lr_array.shape)

            

            # #####===  Normalizing  ===##
            # if lr_array.max() > 1.0 or hr_array.max() > 1.0:
            if self.norm==True:

                print("Normaliza Dividiendo por el máximo y clipping 0")
                lr_array = lr_array / 32767.0
                # hr_array = hr_array / 32767.0
                lr_array = np.clip(lr_array, 0, 1)
                # hr_array = np.clip(hr_array, 0, 1)

            if self.stand==True:
                # assuming format CHW
                # lr_array = lr_array*32767.0
                hr_array = hr_array * 32767.0

                ## #version 1  ###3
                lr_array_mean = lr_array.mean(axis=(1,2), keepdims=True)
                lr_array_stddev = lr_array.std(axis=(1, 2), keepdims=True)
                hr_array_mean = hr_array.mean(axis=(1, 2), keepdims=True)
                hr_array_stddev = hr_array.std(axis=(1, 2), keepdims=True)
                lr_array = (lr_array - lr_array_mean)/lr_array_stddev
                hr_array = (hr_array - hr_array_mean) / hr_array_stddev

                # ### version 2 ####
                # lr_stand, hr_stand = [], []
                # scaler= StandardScaler()
                # ## ASUMMING AN ARRAY OF SHAPE [h,w,c]
                # for ii in range(hr_array.shape[-1]):
                #     lr_array_mean = lr_array.mean(axis=(1,2), keepdims=True)
                #     lr_array_stddev = lr_array.std(axis=(1, 2), keepdims=True)
                #     hr_array_mean = hr_array.mean(axis=(1, 2), keepdims=True)
                #     hr_array_stddev = hr_array.std(axis=(1, 2), keepdims=True)

                #     lr_stand.append(scaler.fit_transform( lr_array[:,:,ii]) )
                #     hr_stand.append(scaler.fit_transform( hr_array[:,:,ii]) )
                
                # lr_stand=np.asarray(lr_stand) # ahora esta en formato [C,H,W]
                # lr_stand=np.asarray(lr_stand)
                # lr_array = np.transpose(lr_stand,(1,2,0)) # vuelta a formato [H,W,C]
                # hr_array = np.transpose(lr_stand,(1,2,0))

                # print("STATS LR mean :", lr_array.mean(axis=(1,2)))
                # print("STATS LR std :", lr_array.std(axis=(1,2)) )

                # print("STATS HR mean :", hr_array.mean(axis=(1,2)))
                # print("STATS HR std :", hr_array.std(axis=(1,2)) )


                # print("STATS HR: [%f, %f] "%( hr_array.mean(axis=(1,2)), hr_array.std(axis=(1,2))   ))
                print("DATA STANDARIZED")

            # lr = lr_array[:, idx:idx + Patch_Size, idy:idy + Patch_Size].astype(
            #     np.float16())  # Formato CHW
            # hr = hr_array[:, idx*self.opt["scale"]:idx*self.opt["scale"] + Patch_Size*self.opt["scale"], idy*self.opt["scale"]:idy*self.opt["scale"] + Patch_Size*self.opt["scale"]].astype(
            #     np.float16())  # Formato CHW

            ##===  Cropping ===##  # Format assume CHW
            cc, hh, ww = lr_array.shape
            ## para que compare siempre con el mismo recorte...
            idx = np.random.randint(0, hh - self.Patch_Size)
            idy = np.random.randint(0, ww - self.Patch_Size)
            np.random.seed(seed)

           

            # lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32())  # Formato CHW
            # hr = hr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32())  # Formato CHW
            
            if self.up_lr == True:
                lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
                hr = hr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
            else:
                lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
                hr = hr_array[:, idx*self.opt["scale"]:idx*self.opt["scale"] + self.Patch_Size*self.opt["scale"], idy*self.opt["scale"]:idy*self.opt["scale"] + self.Patch_Size*self.opt["scale"]].astype(np.float32()) # Formato CHW

            
            # #### DOWNSAMPLING LR ######
            if self.LR_down == True:
                print("Downloading...")
                lr = np.transpose(lr, (1, 2, 0))  # formato HWC
                # lr = resize(lr, (lr.shape[0] / self.scale, lr.shape[1] / self.scale), mode="edge",
                #             anti_aliasing=True)
                lr = rescale(lr, 0.2,anti_aliasing=True)
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

            
            hr = torch.from_numpy(hr[(2,1,0,3),:,:]).float()
            lr = torch.from_numpy(lr[(2,1,0,3),:,:]).float()

            # print("Filter hr:  ", lowpass_hr.shape)
            # print("Filter lr:  ", lowpass_lr.shape)



            if self.stand == True:
                sample = {"LR": lr, "LR_mean":lr_array_mean, "LR_std":lr_array_stddev,
                      "HR": hr, "HR_mean":hr_array_mean, "HR_std": hr_array_stddev, 
                      'LR_path': LR_path, 'HR_path': HR_path}
            else:
                sample = {"LR": lr, "HR": hr, 'LR_path': LR_path, 'HR_path': HR_path,
                "HR_min": hr_array.min(), 'HR_max': hr_array.max(), 'LR_min': lr_array.min(),
                 'LR_max': lr_array.max(), 'lowpass_lr': lowpass_lr, 'lowpass_hr': lowpass_hr
                  }

            # , 'lowpass_lr': lowpass_lr, 'lowpass_r': lowpass_lr,

            # sample = {"LR": lr, "HR": hr, 'LR_path': LR_path, 'HR_path': HR_path, 'HR_min': hr_array.min(),
            #           'HR_max': hr_array.max(), 'LR_min': lr_array.min(), 'LR_max': lr_array.max(),
            #           "LR_mean": lr_array_mean, "LR_std": lr_array_stddev,
            #           "HR_mean": hr_array_mean, "HR_std": hr_array_stddev,}
            # print("CARGOOO.... VALID:  ", lr.shape)
            return sample


    def __len__(self):
        # print("EJecuta LEN")
        "Denotes the total number of samples"
        return len(self.list_IDs)
