# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:49:41 2019

@author: papastrat

"""
import cv2
import datetime
import os
import random
import shutil
import sys
from einops import rearrange
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

config_file = 'config/trainer_config.yml'


def main():
    args = arguments()
    myargs = getopts(sys.argv)
    now = datetime.datetime.now()
    cwd = os.path.dirname(os.getcwd())
    print(cwd)
    if len(myargs) > 0:
        if 'c' in myargs:
            config_file = myargs['c']
    else:
        config_file = 'config/trainer_config.yml'

    config = OmegaConf.load(os.path.join(cwd, config_file))['trainer']

    config.cwd = cwd

    if len(myargs) > 0:
        for key, v in myargs.items():
            if key in config.keys():
                config[key] = v

    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")




    # print("date and time =", dt_string)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    print(f'pyTorch VERSION:{torch.__version__}', )
    print(f'CUDA VERSION')

    print(f'CUDNN VERSION:{torch.backends.cudnn.version()}')
    print(f'Number CUDA Devices: {torch.cuda.device_count()}')
    ## Reproducibility seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if (config.cuda and torch.cuda.is_available()):
        torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    training_generator, val_generator, test_generator, classes = islr_datasets(config)

    for batch_idx, (data, target) in enumerate(training_generator):
        assert len(data.shape) == 5, 'error in data shape'
        B,C,T,H,W = data.shape
        for i in range(T):
            frame = data[0,:,i,:,:].numpy()

            frame = cv2.cvtColor(rearrange(frame,'c h w -> h w c'),cv2.COLOR_RGB2BGR)
            cv2.imshow('Frame',frame)
            if cv2.waitKey(100) & 0xFF  == ord('q'):
                break




if __name__ == '__main__':
    main()
