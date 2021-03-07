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

from data_loader.dataset import data_generators
from models.model_utils import SLR_video_encoder
from models.model_utils import load_checkpoint
from trainer.tester import Tester
from utils import getopts, arguments
from utils.logger import Logger


def main():
    now = datetime.datetime.now()

    cwd = os.getcwd()
    args = arguments()
    myargs = getopts(sys.argv)
    if len(myargs) > 0:
        if 'c' in myargs:
            config_file = myargs['c']
    else:
        config_file = 'config/test_config.yml'
    config = OmegaConf.load(os.path.join(cwd, config_file))['tester']

    if len(myargs) > 0:
        for key, v in myargs.items():
            if key in config.keys():
                config[key] = v
    config.cwd = cwd

    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    cpkt_fol_name = os.path.join(config.cwd,
                                 'checkpoints/model_' + config.model.name + '/dataset_' + config.dataset.name +
                                 '/date_' + dt_string)

    log = Logger(path=cpkt_fol_name, name=config.logger).get_logger()

    writer_path = os.path.join(config.cwd,
                               'runs/model_' + config.model.name + '/dataset_' + config.dataset.name + '/date_' +
                               dt_string)

    writer = SummaryWriter(writer_path)
    shutil.copy(os.path.join(config.cwd, config_file), cpkt_fol_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    log.info(f'pyTorch VERSION:{torch.__version__}', )
    log.info(f'CUDA VERSION')

    log.info(f'CUDNN VERSION:{torch.backends.cudnn.version()}')
    log.info(f'Number CUDA Devices: {torch.cuda.device_count()}')
    # dd/mm/YY H:M:S

    # log.info("date and time =", dt_string)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if (config.cuda and torch.cuda.is_available()):
        torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    training_generator, val_generator, test_generator, classes = data_generators(config)

    model = SLR_video_encoder(config, len(classes))

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f'device: {device}')

    # log.info(f"{model}")
    log.info(f'{len(classes)}')
    if (config.cuda and use_cuda):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = torch.nn.DataParallel(model)
        model.to(device)

    if (config.load):
        pth_file, _ = load_checkpoint(config.pretrained_cpkt, model, strict=True, load_seperate_layers=False)

    tester = Tester(config, model=model,
                    data_loader=training_generator, writer=writer, logger=log,
                    valid_data_loader=val_generator, test_data_loader=test_generator,
                    checkpoint_dir=cpkt_fol_name)
    validation_loss = tester._valid_epoch(0, 'validation', val_generator)
    tester.predict()


if __name__ == '__main__':
    main()
