import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import pandas as pd
import data.util as util


# import gdal
from skimage.transform import rescale, resize, downscale_local_mean


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
        self.LR_down = self.opt["LR_down"]
        # self.opt["scale"]
        self.scale=5


        # super(GdalDataset, self).__init__()

        self.paths_LR = self.opt["dataroot_LR"]
        self.paths_HR = self.opt["dataroot_HR"]

        self.list_IDs = pd.read_csv(self.opt["data_IDs"])
        # self.LR_env = None  # environment for lmdb
        # self.HR_env = None
    def __getitem__(self, index):
        " get the samples"

        if self.opt["phase"] == "train":
            self.Patch_Size = self.opt['HR_size']

            lr_array = np.load(self.paths_LR + "/" + str(self.list_IDs.iloc[index, 1]))
            hr_array = np.load(self.paths_HR + "/" + str(self.list_IDs.iloc[index, 1]))
            print("PATH LR: ", self.paths_LR + "/" + str(self.list_IDs.iloc[index, 1]))
            print("PATH HR: ", self.paths_HR + "/" + str(self.list_IDs.iloc[index, 1]))


            if lr_array.shape[2]<=4:  # Cambia a formato CHW
                lr_array = np.transpose(lr_array,(2,0,1))
            if hr_array.shape[2]<=4:  # Cambia a formato CHW
                hr_array = np.transpose(hr_array,(2,0,1))

            #assert lr_array is not None

            ###===  Normalizing  ===##
            if lr_array.max() > 1.0 or hr_array.max() > 1.0:
                print("Normaliza Dividiendo por el máximo y clipping 0")
                lr_array = lr_array / 32767.0
                hr_array = hr_array / 32767.0
                lr_array = np.clip(lr_array, 0, 1)
                hr_array = np.clip(hr_array, 0, 1)

            standarization = False

            if standarization==True:
                # assuming format CHW
                lr_array_mean = lr_array.mean(axis=(1,2), keepdims=True)
                lr_array_stddev = lr_array.std(axis=(1, 2), keepdims=True)

                hr_array_mean = hr_array.mean(axis=(1, 2), keepdims=True)
                hr_array_stddev = hr_array.std(axis=(1, 2), keepdims=True)

                lr_array = (lr_array - lr_array_mean)/lr_array_stddev
                hr_array = (hr_array - hr_array_mean) / hr_array_stddev
                print("DATA STANDARIZED")

            if self.LR_down == True:
                # #### DOWNSAMPLING LR ######
                lr_array = np.transpose(lr_array, (1, 2, 0))  # formato HWC
                lr_array = resize(lr_array, (lr_array.shape[0] / self.scale, lr_array.shape[1] / self.scale), anti_aliasing=True)
                lr_array = np.transpose(lr_array, (2, 0, 1))  # formato CHW


            ##===  Cropping ===##  # Format assume CHW
            cc,hh,ww = lr_array.shape
            np.random.seed()
            idx = np.random.randint(0, hh - self.Patch_Size)
            idy = np.random.randint(0, ww - self.Patch_Size)
            np.random.seed(self.seed)


            lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(np.float32()) # Formato CHW
            hr = hr_array[:, idx*self.opt["scale"]:idx*self.opt["scale"]
                                                   + self.Patch_Size*self.opt["scale"], idy*self.opt["scale"]:idy*self.opt["scale"]
                                                   + self.Patch_Size*self.opt["scale"]].astype(np.float32()) # Formato CHW


            ### augmentation - flip, rotate
            lr, hr = util.augment([lr, hr], self.opt['use_flip'], self.opt['use_rot'])


### ELIGE CANALES ####

            hr = torch.from_numpy(hr[(2,1,0,3),:,:].copy()).float()  # solo escoge ch RGB
            lr = torch.from_numpy(lr[(2,1,0,3),:,:].copy()).float()
    #############################

            print(" *** HR: ", hr.shape)
            print(" ****** LR: ", lr.shape)



            if standarization == True:
                sample = {"LR": lr, "LR_mean":lr_array_mean, "LR_stddev":lr_array_stddev,
                      "HR": hr, "HR_mean":hr_array_mean, "HR_stddev": hr_array_stddev}
            else:
                sample = {"LR": lr,"HR": hr}

            return sample

        else:
            # print("REALIZA else.....")
            # Patch_Size = 32 #self.opt['HR_size']

            self.Patch_Size = self.opt['HR_size']
            self.seed = 100
            self.LR_down = self.opt["LR_down"]
            self.scale = 5  #self.opt["scale"]

            HR_path = self.paths_HR+"/" + str(self.list_IDs.iloc[index, 1]).replace(".npy","")
            LR_path = self.paths_LR+"/" + str(self.list_IDs.iloc[index, 1]).replace(".npy","")


            # img_LR = util.read_img(self.LR_env, LR_path)
            lr_array = np.load(self.paths_LR + "/" + str(self.list_IDs.iloc[index, 1])).astype(np.float32())
            hr_array = np.load(self.paths_HR + "/" + str(self.list_IDs.iloc[index, 1])).astype(np.float32())

            if lr_array.shape[2]<=4:  # Cambia a formato CHW
                lr_array = np.transpose(lr_array,(2,0,1))
            if hr_array.shape[2]<=4:  # Cambia a formato CHW
                hr_array = np.transpose(hr_array,(2,0,1))

            # #####===  Normalizing  ===##
            if lr_array.max() > 1.0 or hr_array.max() > 1.0:
                print("Normaliza Dividiendo por el máximo y clipping 0")
                lr_array = lr_array / 32767.0
                hr_array = hr_array / 32767.0
                lr_array = np.clip(lr_array, 0, 1)
                hr_array = np.clip(hr_array, 0, 1)

            if self.LR_down == True:
                # #### DOWNSAMPLING LR ######
                lr_array = np.transpose(lr_array, (1, 2, 0))  # formato HWC
                lr_array = resize(lr_array, (lr_array.shape[0] / self.scale,
                                             lr_array.shape[1] / self.scale), anti_aliasing=True)
                lr_array = np.transpose(lr_array, (2, 0, 1))  # formato CHW




            ##===  Cropping ===##  # Format assume CHW
            cc, hh, ww = lr_array.shape
            # print("SELF PATHC: ", Patch_Size)
            # np.random.seed() ### Pseudo-aleatorio
            ## para que compare siempre con el mismo recorte...
            idx = np.random.randint(0, hh - self.Patch_Size)
            idy = np.random.randint(0, ww - self.Patch_Size)
            # np.random.seed(seed)

            # print("[DEBUG...] ANTES del cambio a float16 tiene el tipo: ", lr_array.dtype)
            lr = lr_array[:, idx:idx + self.Patch_Size, idy:idy + self.Patch_Size].astype(
                np.float32())  # Formato CHW
            hr = hr_array[:, idx*self.opt["scale"]:idx*self.opt["scale"] + self.Patch_Size*self.opt["scale"],
                 idy*self.opt["scale"]:idy*self.opt["scale"] + self.Patch_Size*self.opt["scale"]].astype(
                np.float32())  # Formato CHW
            # print("[DEBUG...] cambio a float64 tiene el tipo: ", lr.dtype)
### ELIGE CANALES ####

            hr = torch.from_numpy(hr[ (2,1,0,3), :, :]).float() # solo escoge ch RGB
            lr = torch.from_numpy(lr[ (2,1,0,3), :, :]).float()
            
            print(" HR: ", hr.shape)
            print(" LR: ", lr.shape)
            
            # print("LR VALID shape", lr.shape)

            sample = {"LR": lr, "HR": hr, 'LR_path': LR_path, 'HR_path': HR_path, 'HR_min': hr_array.min(),
                      'HR_max': hr_array.max(), 'LR_min': lr_array.min(), 'LR_max': lr_array.max()}
            # print("CARGOOO.... VALID:  ", lr.shape)
            return sample


    def __len__(self):
        # print("EJecuta LEN")
        "Denotes the total number of samples"
        return len(self.list_IDs)
