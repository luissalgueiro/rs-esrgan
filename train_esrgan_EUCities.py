import os.path
import sys
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import matplotlib
matplotlib.use("Agg")

import torch
import matplotlib.pyplot as plt

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

### Imports luis
import earthpy as et
import geopandas as gpd
import earthpy.plot as ep
import cv2

import os


def main():
    PreUp = False

    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    ratio = opt["scale"]
    if PreUp == True:
        ratio=5

    # train from scratch OR resume training
    if opt['path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['path']['resume_state'])
    else:  # training from scratch
        resume_state = None
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        option.check_resume(opt)  # check resume options

    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        from tensorboardX import SummaryWriter
        tb_logger_train = SummaryWriter(log_dir='/mnt/gpid07/users/luis.salgueiro/git/mnt/BasicSR/tb_logger/' + opt['name'] + "/train")
        tb_logger_val = SummaryWriter(log_dir='//mnt/gpid07/users/luis.salgueiro/git/mnt/BasicSR/tb_logger/' + opt['name'] + "/val" )


    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = 100  #random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True
    # print("OLAAAA_-...", os.environ['CUDA_VISIBLE_DEVICES'])

# #########################################
# ######## DATA LOADER ####################
# #########################################
    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            print("Entro DATASET train......")
            train_set = create_dataset(dataset_opt)
            print("CREO DATASET train_set ", train_set)

            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
            print("CREO train loader: ", train_loader)
        elif phase == 'val':
            print("Entro en phase VAL....")
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                      len(val_set)))
            # for _,ii in enumerate(val_loader):
            #     print("VAL LOADER:........", ii)
            # print(val_loader[0])
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    assert val_loader is not None

    # create model
    model = create_model(opt)
    #print("PASO.....   MODEL ")

    # resume training
    if resume_state:
        print("RESUMING state")
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        print("PASO.....   INIT ")

    # #########################################
    # #########    training    ################
    # #########################################
    # ii=0
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs):
        # print("Entro EPOCH...", ii)
        for _, train_data in enumerate(train_loader):


            # print("Entro TRAIN_LOADER...")
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            model.update_learning_rate()

            # training
            #print("....... TRAIN DATA..........", train_data)
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # log train
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                # print(".............MESSAGE: ", message)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    #print("MSG: ", message)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        # print("K: ", k)
                        # print("V: ", v)
                        if "test" in k:
                            tb_logger_val.add_scalar(k, v, current_step)
                        else:
                            tb_logger_train.add_scalar(k, v, current_step)
                logger.info(message)

            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr_sr  = 0.0
                avg_psnr_lr  = 0.0
                avg_psnr_dif = 0.0
                avg_ssim_lr, avg_ssim_sr, avg_ssim_dif    = 0.0, 0.0, 0.0
                avg_ergas_lr, avg_ergas_sr, avg_ergas_dif = 0.0, 0.0, 0.0
                idx = 0
                # for val_data in val_loader:
                for _, val_data in enumerate(val_loader):
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    # print("Img nameVaL: ", img_name)


                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()

                    sr_img = util.tensor2imgNorm(visuals['SR'],out_type=np.uint8, min_max=(0, 1), MinVal=val_data["LR_min"], MaxVal=val_data["LR_max"])  # uint16
                    gt_img = util.tensor2imgNorm(visuals['HR'],out_type=np.uint8, min_max=(0, 1), MinVal=val_data["HR_min"], MaxVal=val_data["HR_max"])  # uint16
                    lr_img = util.tensor2imgNorm(visuals['LR'], out_type=np.uint8, min_max=(0, 1),
                                                 MinVal=val_data["LR_min"], MaxVal=val_data["LR_max"])  # uint16

                    # Save SR images for reference
                    if idx < 10:
                        # print(idx)
                        util.mkdir(img_dir)
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}'.format(img_name, current_step))
                        util.save_imgSR(sr_img, save_img_path)
                        util.save_imgHR(gt_img, save_img_path)
                        util.save_imgLR(lr_img, save_img_path)
                        print("SAVING CROPS")
                        util.save_imgCROP(lr_img,gt_img,sr_img , save_img_path, ratio, PreUp=PreUp)

                    if PreUp==False:
                        dim2 = (gt_img.shape[1], gt_img.shape[1])
                        print("DIM:", dim2)
                        print("LR image shape ", lr_img.shape)
                        print("HR image shape ", gt_img.shape)
                        lr_img = cv2.resize(np.transpose(lr_img,(1,2,0)), dim2, interpolation=cv2.INTER_NEAREST)
                        lr_img = np.transpose(lr_img,(2,0,1))
                        print("LR image 2 shape ", lr_img.shape)
                        print("LR image 2 shape ", lr_img.shape)

                    avg_psnr_sr += util.calculate_psnr2(sr_img, gt_img)
                    avg_psnr_lr += util.calculate_psnr2(lr_img, gt_img)
                    avg_ssim_lr += util.calculate_ssim2(lr_img, gt_img)
                    avg_ssim_sr += util.calculate_ssim2(sr_img, gt_img)
                    avg_ergas_lr += util.calculate_ergas(lr_img, gt_img, pixratio=ratio)
                    avg_ergas_sr += util.calculate_ergas(sr_img, gt_img, pixratio=ratio)
                    #avg_psnr += util.calculate_psnr2(cropped_sr_img, cropped_gt_img)


                avg_psnr_sr = avg_psnr_sr / idx
                avg_psnr_lr = avg_psnr_lr / idx
                avg_psnr_dif = avg_psnr_lr - avg_psnr_sr
                avg_ssim_lr = avg_ssim_lr / idx
                avg_ssim_sr = avg_ssim_sr / idx
                avg_ssim_dif = avg_ssim_lr - avg_ssim_sr
                avg_ergas_lr  = avg_ergas_lr / idx
                avg_ergas_sr  = avg_ergas_sr / idx
                avg_ergas_dif = avg_ergas_lr - avg_ergas_sr
                # print("IDX: ", idx)

                # log VALIDATION
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr_sr))
                logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim_sr))
                logger.info('# Validation # ERGAS: {:.4e}'.format(avg_ergas_sr))

                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr_SR: {:.4e}'.format(
                    epoch, current_step, avg_psnr_sr))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr_LR: {:.4e}'.format(
                    epoch, current_step, avg_psnr_lr))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr_DIF: {:.4e}'.format(
                    epoch, current_step, avg_psnr_dif))

                logger_val.info('<epoch:{:3d}, iter:{:8,d}> ssim_LR: {:.4e}'.format(
                    epoch, current_step, avg_ssim_lr))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> ssim_SR: {:.4e}'.format(
                    epoch, current_step, avg_ssim_sr))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> ssim_DIF: {:.4e}'.format(
                    epoch, current_step, avg_ssim_dif))

                logger_val.info('<epoch:{:3d}, iter:{:8,d}> ergas_LR: {:.4e}'.format(
                    epoch, current_step, avg_ergas_lr))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> ergas_SR: {:.4e}'.format(
                    epoch, current_step, avg_ergas_sr))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> ergas_DIF: {:.4e}'.format(
                    epoch, current_step, avg_ergas_dif))

                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger_val.add_scalar('dif_PSNR', avg_psnr_dif, current_step)
                    # tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger_val.add_scalar('dif_SSIM', avg_ssim_dif, current_step)
                    tb_logger_val.add_scalar('dif_ERGAS', avg_ergas_dif, current_step)

                    tb_logger_val.add_scalar('psnr_LR', avg_psnr_lr, current_step)
                    # tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger_val.add_scalar('ssim_LR', avg_ssim_lr, current_step)
                    tb_logger_val.add_scalar('ERGAS_LR', avg_ergas_lr, current_step)

                    tb_logger_val.add_scalar('psnr_SR', avg_psnr_sr, current_step)
                    # tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger_val.add_scalar('ssim_SR', avg_ssim_sr, current_step)
                    tb_logger_val.add_scalar('ERGAS_SR', avg_ergas_sr, current_step)


                    print("****** SR_IMG: ", sr_img.shape)
                    print("****** LR_IMG: ", lr_img.shape)
                    print("****** GT_IMG: ", gt_img.shape)

                    fig1,ax1 = ep.plot_rgb(sr_img, rgb=[2, 1, 0], stretch=True)
                    tb_logger_val.add_figure("SR_plt", fig1, current_step,close=True)
                    fig2, ax2 = ep.plot_rgb(gt_img, rgb=[2, 1, 0], stretch=True)
                    tb_logger_val.add_figure("GT_plt", fig2, current_step, close=True)
                    fig3, ax3 = ep.plot_rgb(lr_img, rgb=[2, 1, 0], stretch=True)
                    tb_logger_val.add_figure("LR_plt", fig3, current_step, close=True)
                    # print("TERMINO GUARDAR IMG TB")
            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)
        # ii=ii+1
    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
