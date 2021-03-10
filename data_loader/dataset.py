import os

import torch.utils.data as data

from data_loader.autsl.autsl_loader import AUTSL
# from data_loader.autsl.autsl_loaderv2 import AUTSLv2
from data_loader.autsl.rgbd_loader import AUTSL_RGBD
from data_loader.loader_utils import read_autsl_csv


def data_generators(config):
    """

    Args:
        config ():

    Returns:

    """
    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'num_workers': 2}
    val_params = {'batch_size': config.dataloader.val.batch_size,
                  'shuffle': config.dataloader.val.shuffle,
                  'num_workers': config.dataloader.val.num_workers,
                  'pin_memory': True}

    train_params = {'batch_size': config.dataloader.train.batch_size,
                    'shuffle': config.dataloader.train.shuffle,
                    'num_workers': config.dataloader.train.num_workers,
                    'pin_memory': True}



    train_prefix = "train"
    validation_prefix = "val"
    _, _, classes = read_autsl_csv(os.path.join(config.cwd, 'data_loader/autsl/train_labels.csv'))
    training_set = AUTSL(config, train_prefix, classes)
    training_generator = data.DataLoader(training_set, **train_params)

    validation_set = AUTSL(config, validation_prefix, classes)
    validation_generator = data.DataLoader(validation_set, **val_params)
    test_set = AUTSL(config, 'test', classes)
    test_generator = data.DataLoader(test_set, **test_params)
    return training_generator, validation_generator, test_generator, classes


def RGBD_generators(config):
    """

    Args:
        config (): configuration dictionary

    Returns:

    """
    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'num_workers': 2}
    val_params = {'batch_size': config.dataloader.val.batch_size,
                  'shuffle': config.dataloader.val.shuffle,
                  'num_workers': config.dataloader.val.num_workers,
                  'pin_memory': True}

    train_params = {'batch_size': config.dataloader.train.batch_size,
                    'shuffle': config.dataloader.train.shuffle,
                    'num_workers': config.dataloader.train.num_workers,
                    'pin_memory': True}

    train_prefix = "train"
    validation_prefix = "val"
    test_prefix = 'test'
    _, _, classes = read_autsl_csv(os.path.join(config.cwd, 'data_loader/autsl/train_labels.csv'))
    training_set = AUTSL_RGBD(config, train_prefix, classes)
    training_generator = data.DataLoader(training_set, **train_params)

    validation_set = AUTSL_RGBD(config, validation_prefix, classes)
    validation_generator = data.DataLoader(validation_set, **val_params)
    test_set = AUTSL_RGBD(config, test_prefix, classes)
    test_generator = data.DataLoader(test_set, **test_params)
    return training_generator, validation_generator, test_generator, classes
