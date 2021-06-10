# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:49:41 2019

@author: papastrat

"""
import datetime
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from data_loader.dataset import islr_datasets
from models.model_utils import RGBD_model
from models.model_utils import select_optimizer, load_checkpoint
from trainer.trainer_rgbdsk import TrainerRGBDSK
from utils.logger import Logger
from utils.util import arguments, getopts

config_file = 'config/RGBDSK/trainer_RGBD_config.yml'


def main():
    args = arguments()
    myargs = getopts(sys.argv)
    now = datetime.datetime.now()

    cwd = os.getcwd()

    if len(myargs) > 0:
        if 'c' in myargs:
            config_file = myargs['c']
    else:
        config_file = 'config/RGBDSK/trainer_RGBD_config.yml'

    config = OmegaConf.load(os.path.join(cwd, config_file))['trainer']
    if len(myargs) > 0:
        for key, v in myargs.items():
            if key in config.keys():
                config[key] = v
    config.cwd = cwd
    print(config)
    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    cpkt_fol_name = os.path.join(config.cwd,
                                 'checkpoints/model_' + config.model.name + '/dataset_' + config.dataset.name + '/date_' + dt_string)

    log = Logger(path=cpkt_fol_name, name=config.logger).get_logger()

    writer_path = os.path.join(config.cwd,
                               'runs/model_' + config.model.name + '/dataset_' + config.dataset.name + '/date_' + dt_string)

    writer = SummaryWriter(writer_path)

    shutil.copy(os.path.join(config.cwd, config_file), cpkt_fol_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    log.info(f'pyTorch VERSION:{torch.__version__}', )
    log.info(f'CUDA VERSION')

    log.info(f'CUDNN VERSION:{torch.backends.cudnn.version()}')
    log.info(f'Number CUDA Devices: {torch.cuda.device_count()}')
    ## Reproducibility seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if (config.cuda and torch.cuda.is_available()):
        torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    training_generator, val_generator, test_generator, classes = islr_datasets(config)

    model = RGBD_model(config, len(classes))

    log.info(f'{len(classes)}')
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f'device {device}')
    if config.load:
        log.info(f'LOAD RGB CPKT')
        pth_file, _ = load_checkpoint(
            config.rgb_cpkt,
            model.rgb_encoder, strict=False, load_seperate_layers=False)
        log.info(f'LOAD DEPTH CPKT')

        pth_file, _ = load_checkpoint(
            config.depth_cpkt,
            model.depth_encoder, strict=False, load_seperate_layers=False)

        _, _ = load_checkpoint(
            '/home/papastrat/PycharmProjects/SLVTP/checkpoints/model_STGCN/dataset_AUTSL_SK/date_08_04_2021_21.01.21/best_model_epoch_53.pth',
            model.sk_encoder, strict=False, load_seperate_layers=False)

    if torch.cuda.device_count() > 1:
        log.info(f"Let's use {torch.cuda.device_count()} GPUs!")

        model = torch.nn.DataParallel(model)

    model.to(device)
    optimizer, scheduler = select_optimizer(model, config['model'])
    log.info(f"{model}")
    log.info(f"Checkpoint Folder {cpkt_fol_name} ")
    trainer = TrainerRGBDSK(config, model=model, optimizer=optimizer,
                            data_loader=training_generator, writer=writer, logger=log,
                            valid_data_loader=val_generator, test_data_loader=test_generator,
                            lr_scheduler=scheduler,
                            checkpoint_dir=cpkt_fol_name)

    trainer.train()


if __name__ == '__main__':
    main()
