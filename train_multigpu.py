# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:49:41 2019

@author: papastrat

"""
import argparse
import datetime
import os
import random

# import configargparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from data_loader.dataset import data_generators
from models.model_utils import SLR_video_encoder
from models.model_utils import select_optimizer, load_checkpoint
from trainer.trainer import Trainer
from utils.logger import Logger

checkpoints_dict = {
    'phoenix2014_feats': './checkpoints/model_cui/dataset_phoenix2014/best_date_19_02_2020_21.41.25/best_wer.pth',
    'gsl': '/media/papastrat/60E8EA1EE8E9F268/SLR_GAN/checkpoints/model_cui/dataset_GSL_SI'
           '/model_cui_GSL_SI_CPKT/best_wer.pth',
    'GSL_SD': './checkpoints/model_cui/dataset_GSL_SD/ws_loss_normal_test_dayMon Nov 18 10:23:59 '
              '2019/best_wer.pth',
    'phoenix2014T': './checkpoints/model_cui/dataset_phoenix2014T/date_13_02_2020_09.15.42/last.pth',
    'csl_iso': './checkpoints/model_cui/dataset_csl_split1/csl_cui_enctc/best_wer.pth',
    'csl': './checkpoints/model_cui/csl_iso/_test_dayTue Oct 29 10:36:24 2019/best.pth'}


# datetime object containing current date and time


def arguments():
    parser = argparse.ArgumentParser(description='SLR Challenge')

    parser.add_argument('--dataset', type=str, default='Autsl', metavar='rc',
                        help='slr dataset phoenix_iso, phoenix_iso_I5, ms_asl , signum_isolated , csl_iso')

    parser.add_argument('--model', type=str, default='IR_CSN_152', help=' ',
                        choices='GoogLeNet_TConvs,I3D,I3D_VAE,ECAI3D,R3D,X3D,ECA_IR_CSN_152,IR_CSN_152_VAE')

    parser.add_argument('--pseudo_batch', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 1)')

    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--resume', action='store_true', default=False,
                        help='For restoring Model')

    parser.add_argument('--pretrained_cpkt', type=str,
                        default='/home/papastrat/PycharmProjects/SLR_challenge/checkpoints/model_IR_CSN_152/dataset_Autsl/date_01_02_2021_17.18.01/best_model.pth',
                        help='fs checkpoint')
    args = parser.parse_args()

    args.cwd = os.getcwd()
    return args


def main():
    now = datetime.datetime.now()
    args = arguments()
    args_dict = args.__dict__
    conf = OmegaConf.create(args_dict)
    print(conf)

    config = OmegaConf.load(os.path.join(args.cwd, 'config/trainer_config.yml'))['trainer']
    config1 = OmegaConf.merge(config, conf)
    # exit()
    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    cpkt_fol_name = os.path.join(args.cwd,
                                 'checkpoints/model_' + args.model + '/dataset_' + args.dataset + '/date_' + dt_string)
    log = Logger(path=cpkt_fol_name, name=config.logger).get_logger()

    writer_path = os.path.join(args.cwd, 'runs/model_' + args.model + '/dataset_' + args.dataset + '/date_' + dt_string)

    writer = SummaryWriter(writer_path)

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
    training_generator, val_generator, test_generator, classes = data_generators(args, config)

    model = SLR_video_encoder(args, len(classes))
    # inp = torch.randn(1,3,64,224,224)
    # out = model(inp)
    # print(out.shape)
    # exit()

    log.info(f"{model}")
    log.info(f'{len(classes)}')
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f'device {device}')
    #
    if (args.resume):
        model.model.fc = torch.nn.Linear(2048, 226)
        #model.model.classifier = torch.nn.Linear(3840, 226)
        pth_file, _ = load_checkpoint(args.pretrained_cpkt, model, strict=False, load_seperate_layers=False)
        resume_epoch = pth_file['epoch']
        model.model.classifier = torch.nn.Linear(3840, 226)
        #torch.nn.init.xavier_uniform(model.fc.weight)


    if (config.cuda and use_cuda):
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = torch.nn.DataParallel(model,device_ids=[0,1])

        #model.to(device)
    #model = torch.nn.DataParallel(model)
    optimizer, scheduler = select_optimizer(model, config['model'], checkpoint=None)

    log.info(f"Checkpoint Folder {cpkt_fol_name} ")
    trainer = Trainer(config, args, model=model, optimizer=optimizer,
                      data_loader=training_generator, writer=writer, logger=log,
                      valid_data_loader=val_generator, test_data_loader=test_generator,
                      lr_scheduler=scheduler,
                      checkpoint_dir=cpkt_fol_name)

    trainer.train()


if __name__ == '__main__':
    main()
