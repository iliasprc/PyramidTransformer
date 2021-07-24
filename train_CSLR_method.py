import datetime
import os
import pathlib
import random
import shutil
import sys

import numpy as np
import pytorch_lightning
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

from data_loader.dataset import CSLR_DataModule
from models import CSLR_video_encoder
from models.model_utils import select_optimizer, load_checkpoint
from utils.logger import Logger
from utils.util import cslr_arguments, getopts

cwd_path = pathlib.Path.cwd()
print(' current path  ', cwd_path)

checkpoints_dict1 = {
    'augmentation'      : '/home/iliask/PycharmProjects/Epikoinwnw_repo/flask_server_epikoinwnw/app_checkpoints'
                          '/augmentation_checkpoint.pth',
    'bounding_box'      : '/home/iliask/PycharmProjects/Epikoinwnw_repo/flask_server_epikoinwnw/app_checkpoints'
                          '/bbox_checkpoint.pth',
    'gan'               : '/home/iliask/PycharmProjects/SLR_GAN/checkpoints/model_CLSR/dataset_GSL_SI'
                          '/date_08_07_2020_09.18.45/bestgenerator.pth',
    'full_video_trained': '/home/iliask/PycharmProjects/SLR_GAN/checkpoints/model_CLSR/dataset_GSL_SI'
                          '/date_26_08_2020_12.23.07/bestgenerator.pth',
    'both'              : '/home/iliask/PycharmProjects/SLR_GAN/checkpoints/model_CLSR/dataset_GSL_SI'
                          '/date_27_08_2020_16.19.24/bestgenerator.pth',
    'both1'             : '/home/iliask/PycharmProjects/Epikoinwnw_repo/flask_server_epikoinwnw/app_checkpoints'
                          '/date_28_08_2020_17.12.17/bestgenerator.pth'}

checkpoints_dict = {
    'phoenix2014_feats': './checkpoints/model_cui/dataset_phoenix2014/best_date_19_02_2020_21.41.25/best_wer.pth',
    'GSL_SI'           : './checkpoints/model_cui/dataset_greek_SI/ws_loss_normal_test_dayFri Nov 15 10_08_58 '
                         '2019/best_wer.pth',
    'GSL_SD'           : './checkpoints/model_cui/dataset_GSL_SD/ws_loss_normal_test_dayMon Nov 18 10:23:59 '
                         '2019/best_wer.pth',
    'phoenix2014T'     : './checkpoints/model_cui/dataset_phoenix2014T/date_13_02_2020_09.15.42/last.pth',
    'csl_split1'       : './checkpoints/model_cui/dataset_csl_split1/csl_cui_enctc/best_wer.pth',
    'GSL_isolated'     : './checkpoints/model_GoogLeNet_TConvs/dataset_gsl_iso/date_19_06_2020_11.19.22/best.pth'}

i3d_path = '/home/iliask/Desktop/ilias/pretrained_checkpoints/I3D/ws_loss_ent_ctc_test_daySun Nov 17 16:53:43 ' \
           '2019/best_wer.pth'


def main():
    args = cslr_arguments()
    myargs = getopts(sys.argv)
    now = datetime.datetime.now()
    cwd = os.getcwd()
    if len(myargs) > 0:
        if 'c' in myargs:
            config_file = myargs['c']
    else:
        config_file = 'config/CSLR/trainer_config.yml'

    config = OmegaConf.load(os.path.join(cwd, config_file))['trainer_cslr']

    config.cwd = cwd

    if len(myargs) > 0:
        for key, v in myargs.items():
            if key in config.keys():
                config[key] = v

    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    cpkt_fol_name = os.path.join(config.cwd,
                                 'checkpoints/model_' + config.model.name + '/dataset_' + config.dataset.name +
                                 '/date_' + dt_string)

    log = Logger(path=cpkt_fol_name, name=config.logger).get_logger()

    writer_path = os.path.join(config.cwd,
                               'checkpoints/model_' + config.model.name + '/dataset_' + config.dataset.name +
                               '/date_' + dt_string + '/runs/')
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

    datamodule = CSLR_DataModule(config)
    datamodule.setup()
    classes = datamodule.classes
    id2w = datamodule.id2w
    training_generator, val_generator, test_generator = datamodule.train_dataloader(), datamodule.val_dataloader(), None
    model = CSLR_video_encoder(config, len(classes))
    if (config.load):
        # model.cnn.replace_logits(2042)
        pth_file, _ = load_checkpoint(config.pretrained_cpkt, model, strict=True, load_seperate_layers=False)
        # model.cnn.replace_logits(311)
        model.fc = torch.nn.Linear(1024,311)
    # if config.load:
    #     model.load_from_checkpoint(config.pretrained_cpkt,strict=True)

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f'device: {device}')

    log.info(f'{len(classes)}')

    optimizer, scheduler = select_optimizer(model, config['model'], None)

    log.info(f"Checkpoint Folder {cpkt_fol_name} ")
    log.info(f"{model}")
    # model.to(device)
    # trainer = Trainer_CSLR_method(config=config, model=model, optimizer=optimizer,
    #                               data_loader=training_generator, writer=writer, id2w=id2w,
    #                               valid_data_loader=val_generator, test_data_loader=test_generator,
    #                               lr_schedulers=scheduler,
    #                               checkpoint_dir=cpkt_fol_name, logger=log)
    #
    # trainer.train()

    from pytorch_lightning.callbacks import GPUStatsMonitor, ModelCheckpoint
    gpu_stats = GPUStatsMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=cpkt_fol_name, verbose=True, monitor='valid_loss_epoch')
    trainer = pytorch_lightning.Trainer(gpus=1, logger=TensorBoardLogger(save_dir=writer_path),
                                        accumulate_grad_batches=2,
                                        stochastic_weight_avg=True, callbacks=[gpu_stats, checkpoint_callback])

    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
