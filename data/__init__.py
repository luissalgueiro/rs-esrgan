'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(dataset, dataset_opt):
    '''create dataloader '''
    phase = dataset_opt['phase']
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers= 4, #dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


def create_dataset(dataset_opt):
    '''create dataset'''
    mode = dataset_opt['mode']
    # elif mode == 'LRHR':
    #     from data.LRHR_dataset import LRHRDataset as D
    if mode == "NoUpsamplingGDAL_PretrainRGB_v3":
        from data.GDAL_datasetv4_PretrainESRGAN_v3 import GdalDataset as D
    elif mode == "NoUpsamplingGDAL_PretrainRGB_ALL":
        from data.GDAL_datasetv4_PretrainESRGAN_ALL import GdalDataset as D
    elif mode == "NoUpsamplingGDAL_PretrainRGB_LRup":
        from data.GDAL_datasetv4_PretrainESRGAN_Up import GdalDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
