# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:49:41 2019

@author: papastrat

"""
import datetime
import os
import random
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from data_loader.dataset import RGBD_generators
from models.model_utils import RGBD_model
from models.model_utils import load_checkpoint
from trainer.tester_rgbd import TesterRGBD
from utils.logger import Logger

config_file = 'config/RGBD/tester_RGBD_config.yml'


def main():
    now = datetime.datetime.now()

    cwd = os.getcwd()
    config = OmegaConf.load(os.path.join(cwd, config_file))['tester']

    config.cwd = cwd

    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    cpkt_fol_name = os.path.join(config.cwd,
                                 'checkpoints/model_' + config.model.name + '/dataset_' + config.dataset.name + '/date_' + dt_string)

    log = Logger(path=cpkt_fol_name, name=config.logger).get_logger()

    writer_path = os.path.join(config.cwd,
                               'runs/model_' + config.model.name + '/dataset_' + config.dataset.name + '/date_' + dt_string)

    writer = SummaryWriter(writer_path)

    shutil.copy(os.path.join(config.cwd, config_file), cpkt_fol_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    log.info(f'PyTorch VERSION:{torch.__version__}', )
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
    # data generators
    training_generator, val_generator, test_generator, classes = RGBD_generators(config)

    model = RGBD_model(config, len(classes))

    log.info(f'{len(classes)}')
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f'device {device}')

    if torch.cuda.device_count() > 1:
        log.info(f"Let's use {torch.cuda.device_count()} GPUs!")

        model = torch.nn.DataParallel(model)

    pth_file, _ = load_checkpoint(
        config.pretrained_cpkt, strict=True)

    model.to(device)

    log.info(f"{model}")
    log.info(f"Checkpoint Folder {cpkt_fol_name} ")
    tester = TesterRGBD(config, model=model,
                        data_loader=training_generator, writer=writer, logger=log,
                        valid_data_loader=val_generator, test_data_loader=test_generator,

                        checkpoint_dir=cpkt_fol_name)

    tester.predict(0)


if __name__ == '__main__':
    main()
