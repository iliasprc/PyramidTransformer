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

# import configargparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from data_loader.dataset import data_generators,islr_datasets
from models.model_utils import ISLR_video_encoder
from models.model_utils import select_optimizer, load_checkpoint
from trainer.trainer import Trainer
from utils.logger import Logger
from utils.util import arguments, getopts

config_file = 'config/ISLR/trainer_config.yml'


def main():
    args = arguments()
    myargs = getopts(sys.argv)
    now = datetime.datetime.now()
    cwd = os.getcwd()
    if len(myargs) > 0:
        if 'c' in myargs:
            config_file = myargs['c']
    else:
        config_file = 'config/ISLR/trainer_config.yml'

    config = OmegaConf.load(os.path.join(cwd, config_file))['trainer']

    config.cwd = cwd

    if len(myargs) > 0:
        for key, v in myargs.items():
            if key in config.keys():
                config[key] = v

    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    cpkt_fol_name = os.path.join(config.cwd,
                                 'checkpoints/model_' + config.model.name + '/dataset_' + config.dataset.name + '/date_' + dt_string)

    log = Logger(path=cpkt_fol_name, name=config.logger).get_logger()

    writer_path = os.path.join(config.cwd,
                               'checkpoints/model_' + config.model.name + '/dataset_' + config.dataset.name + '/date_' + dt_string + '/runs/')
    # os.path.join(config.cwd,
    # 'runs/model_' + config.model.name + '/dataset_' + config.dataset.name + '/date_' + dt_string)

    writer = SummaryWriter(writer_path)
    shutil.copy(os.path.join(config.cwd, config_file), cpkt_fol_name)

    # log.info("date and time =", dt_string)
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

    model = ISLR_video_encoder(config, len(classes))

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    log.info(f'device: {device}')





    if (config.load):
        model.replace_logits(2000)

        pth_file, _ = load_checkpoint(config.pretrained_cpkt, model, strict=True, load_seperate_layers=False)

        model.replace_logits(2042)

    else:
        pth_file = None
    log.info(f'{model}')
    if (config.cuda and use_cuda):
        if torch.cuda.device_count() > 1:
            log.info(f"Let's use {torch.cuda.device_count()} GPUs!")

            model = torch.nn.DataParallel(model)
    model.to(device)


    optimizer, scheduler = select_optimizer(model, config['model'], None)
    log.info(f'{model}')
    log.info(f"Checkpoint Folder {cpkt_fol_name} ")
    trainer = Trainer(config, model=model, optimizer=optimizer,
                      data_loader=training_generator, writer=writer, logger=log,
                      valid_data_loader=val_generator, test_data_loader=test_generator,
                      lr_scheduler=scheduler,
                      checkpoint_dir=cpkt_fol_name)



    trainer.train()


if __name__ == '__main__':
    main()
