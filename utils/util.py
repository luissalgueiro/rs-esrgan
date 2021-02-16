import os
import math
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import random
import torch
import logging
from skimage.measure import compare_ssim as ssimL
from skimage import exposure


### new Imports
import earthpy as et
import geopandas as gpd
import earthpy.plot as ep


# sscale = 5.0
####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)


####################
# image convert
####################


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        print("NDIM: ", n_dim)
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        print("NDIM: ", n_dim)
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        print("NDIM: ", n_dim)
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def tensor2imgNorm(tensor, out_type=np.uint16, min_max=(0, 1), MinVal=-1, MaxVal=-1, freqpass=0):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [MinVal,MaxVal], np.uint16 (default), depending of the channel
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        # print("NDIM: ", n_dim)
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        # print("NDIM: ", n_dim) # AQUI ENTRA
        print("Len tensor: ", tensor.shape)
        freqpass = freqpass.squeeze().float().cpu()  # clamp
        print("Len tensor 2: ", freqpass.shape)
        img_np = tensor +  freqpass
        img_np = img_np.numpy()
        
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    img_np = (img_np * 32767.0).round()
    return img_np

def tensor2imgStand(tensor,  MeanVal, StdVal):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [MinVal,MaxVal], np.uint16 (default), depending of the channel
    '''
    tensor = tensor.squeeze().float().cpu()
    n_dim = tensor.dim()
    # print(f"Tensor dtype: {tensor.dtype}")
    # print(f"Shapes= tensor:{tensor.shape} \t  mean:{MeanVal.shape} \t Std:{StdVal.shape} ")
    # print(f"Range:  Tensor=[{tensor.min()} - {tensor.max()} ] ")
    # print(f"Mean={MeanVal}")
    # print(f"StdVal={StdVal}")
    # MeanVal = MeanVal.squeeze()
    # StdVal = StdVal.squeeze()
    img_np = (tensor.numpy())*StdVal[0].numpy() + MeanVal[0].numpy()    



    # if n_dim == 4:
    #     # print("NDIM: ", n_dim)
    #     n_img = len(tensor)
    #     img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
    #     img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    # elif n_dim == 3:
    #     img_np = tensor.numpy() # aqui entra

    # elif n_dim == 2:
    #     # print("NDIM: ", n_dim)
    #     img_np = ( tensor.numpy())*StdVal + MeanVal
    # else:
    #     raise TypeError(
    #         'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    # if out_type == np.uint8:
    #     img_np = (img_np * 255.0).round()
    # img_np = (img_np * 32767.0).round()
    # print("*** UTIL STAND***")
    # img_np = img_np.round()
    # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    # print(f"Range AFTER:  Img_NP=[{img_np.min()} - {img_np.max()} ] ")

    return img_np

def save_imgSR(img, img_path, mode='RGB'):
    # print("**** img min: %f  , max: %f  "%(img.min(), img.max() ))
    # print("***  type: ", type(img))
    # print("***  SHAPE: ", img.shape)
    ep.plot_rgb(img, rgb=[0,1,2], title="SR_RGB", stretch=True )
    plt.savefig(img_path+"_RGB_SR.png")
    plt.close()

def save_imgHR(img, img_path, mode='RGB'):
    # print("**** img min: %f , max: %f "%(img.min(), img.max() ))
    ep.plot_rgb(img, rgb=[0, 1, 2], title="HR_RGB", stretch=True )
    plt.savefig(img_path + "_RGB_HR.png")
    plt.close()

def save_imgLR(img, img_path, mode='RGB'):
    ep.plot_rgb(img, rgb=[0, 1, 2], title="LR_RGB", stretch=True)
    plt.savefig(img_path + "_RGB_LR.png")
    plt.close()

def save_imgCROP(imgLR, imgHR, imgSR, img_path,sscale1, PreUp=True):
    lr_int8 = reacondiciona_img(imgLR)
    lr_int8 = np.transpose(lr_int8, [1, 2, 0])
    hr_int8 = reacondiciona_img(imgHR)
    hr_int8 = np.transpose(hr_int8, [1, 2, 0])
    sr_int8 = reacondiciona_img(imgSR)
    sr_int8 = np.transpose(sr_int8, [1, 2, 0])
    # print("PREUP: ", PreUp)
    dim2 = (hr_int8.shape[1], hr_int8.shape[0])    
    psnr_hr = calculate_psnr2(imgHR, imgHR)
    ssim_hr = calculate_ssim2(imgHR, imgHR)
    ergas_hr = calculate_ergas(imgHR, imgHR, pixratio=sscale1)
    if PreUp == True:
        psnr_lr = calculate_psnr2(imgLR, imgHR)
        ssim_lr = calculate_ssim2(imgLR, imgHR)
        ergas_lr = calculate_ergas(imgLR, imgHR, pixratio=sscale1)
    else:
        psnr_lr = calculate_psnr2(cv2.resize(imgLR, dim2, interpolation=cv2.INTER_NEAREST), imgHR)
        ssim_lr = calculate_ssim2(cv2.resize(imgLR, dim2, interpolation=cv2.INTER_NEAREST), imgHR)
        ergas_lr = calculate_ergas(cv2.resize(imgLR, dim2, interpolation=cv2.INTER_NEAREST), imgHR, pixratio=sscale1)
    psnr_sr = calculate_psnr2(imgSR, imgHR)
    ssim_sr = calculate_ssim2(imgSR, imgHR)
    ergas_sr = calculate_ergas(imgSR, imgHR, pixratio=sscale1)

    ######### PLOTING CROP ##############################
    fig = plt.figure(figsize=(18, 18))
    fig.subplots_adjust(hspace=0.35, wspace=0.5)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(hr_int8[:,:,:3])
    plt.axis("off")
    plt.title("HR \n [PSNR: %.2f, SSIM: %f, ERGAS: %.2f, SZ: %d]" % (psnr_hr, ssim_hr, ergas_hr, hr_int8.shape[1]))

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(lr_int8[:,:,:3])
    plt.axis("off")
    plt.title("LR \n [PSNR: %.2f, SSIM: %f, ERGAS: %.2f, SZ: %d]" % (psnr_lr, ssim_lr, ergas_lr, lr_int8.shape[1]))

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(sr_int8[:,:,:3])
    plt.axis("off")
    plt.title("SR \n [PSNR: %.2f, SSIM: %f, ERGAS: %.2f, SZ: %d]" % (psnr_sr, ssim_sr, ergas_sr, sr_int8.shape[1]))
   
    #################### FALSE COLOUR #####
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(hr_int8[:, :, (3,0,1)])
    plt.axis("off")
    plt.title("HR \n [PSNR: %.2f, SSIM: %f, ERGAS: %.2f, SZ: %d]" % (psnr_hr, ssim_hr, ergas_hr, hr_int8.shape[1]))

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(lr_int8[:, :, (3,0,1)])
    plt.axis("off")
    plt.title("LR \n [PSNR: %.2f, SSIM: %f, ERGAS: %.2f, SZ: %d]" % (psnr_lr, ssim_lr, ergas_lr, lr_int8.shape[1]))

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(sr_int8[:, :, (3,0,1)])
    plt.axis("off")
    plt.title("SR \n [PSNR: %.2f, SSIM: %f, ERGAS: %.2f, SZ: %d]" % (psnr_sr, ssim_sr, ergas_sr, sr_int8.shape[1]))

    plt.savefig(img_path + "_CROPS.png", bbox_inches="tight", pad_inches=0)
    # print("Guardo Recorte")
    plt.close()


def reacondiciona_img(arr):
    rgb=arr
    smin = 2
    smax = 98
    arr_rescaled = np.zeros_like(rgb)
    for ii, band in enumerate(rgb):
        lower, upper = np.percentile(band, (smin, smax))
        arr_rescaled[ii] = exposure.rescale_intensity(band, in_range=(lower, upper))

    high = 255
    low = 0
    cmin = float(arr_rescaled.min())
    cmax = float(arr_rescaled.max())
    crange = cmax - cmin
    sscale = float(high - low) / crange
    bytedata = (arr_rescaled - cmin) * sscale + low

    return (bytedata.clip(low, high) + 0.5).astype("uint8")
    


####################
# metric
####################



def calculate_psnr2(img1, img2):
    img1 = img1.astype(np.uint16)
    img2 = img2.astype(np.uint16)
    # print("***** range values IMG1: [%f, %f]" % (img1.min(), img1.max()))
    # print("***** range values IMG2: [%f, %f]" % (img2.min(), img2.max()))
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(65535.0 / np.sqrt(mse) )


def rmse(img1, img2):
    return ((img1-img2)**2).mean()**0.5



def calculate_ssim2(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: cambio a np.float16
    '''
    if img1.shape[2]>5:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[2] > 5:
        img2 = np.transpose(img2, (1, 2, 0))
    img1 = img1.astype(np.uint16)
    img2 = img2.astype(np.uint16)
    s = ssimL(img1, img2, data_range=65535.0, multichannel=True)
    return s

    
def calculate_ergas(img1, img2, pixratio=5.0):
    """
    :param img1: resultado  shape=[CHW]
    :param img2: target
    :param pixratio: 5.0 ratio HR/LR
    :return: ERGAS
    """
    if pixratio==1:
        pixratio=5.0

    if img1.shape[0] < 5:
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0)) # HWC
    bands = img1.shape[2]
    add_bands= np.zeros(bands)
    # print(f"Shapes: Img1:{img1.shape} \t Img2:{img2.shape}")
    for band in range(bands):
        add_bands[band] = ((rmse(img1[:,:,band], img2[:,:,band]))/(img2[:,:,band].mean()))**2
    ergas = 100*pixratio*((1.0/bands)*add_bands.sum())**0.5
    return ergas






