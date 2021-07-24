import os

import torch.utils.data as data
from pytorch_lightning import LightningDataModule

from data_loader.autsl.autsl_loader import AUTSL
from data_loader.autsl.autsl_loader_rgb_sk import AUTSL_RGB_SK
from data_loader.autsl.autsl_loader_skeleton import AUTSLSkeleton
from data_loader.autsl.rgbd_loader import AUTSL_RGBD
from data_loader.autsl.rgbdsk_loader import AUTSL_RGBD_SK
from data_loader.gsl.dataloader_gsl_si import GSL_SI, read_gsl_continuous_classes
from data_loader.gsl.dataloader_gsl_si_sk import GSL_SI_Skeleton
from data_loader.gsl_iso.dataloader_greek_isolated import read_classes_file
from data_loader.loader_utils import read_autsl_csv
from data_loader.multi_slr.dataloader_multi_SLR import Multi_SLR
from data_loader.wasl.wasl_dataset import WASLdataset


class ISLR_DataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.test_params = {'batch_size': config.dataloader.test.batch_size,
                            'shuffle': False,
                            'num_workers': 2}
        self.val_params = {'batch_size': config.dataloader.val.batch_size,
                           'shuffle': config.dataloader.val.shuffle,
                           'num_workers': config.dataloader.val.num_workers,
                           'pin_memory': True}

        self.train_params = {'batch_size': config.dataloader.train.batch_size,
                             'shuffle': True,
                             'num_workers': config.dataloader.train.num_workers,
                             'pin_memory': True}

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        self.training_set, self.val_set, self.test_set, self.classes, self.id2w = select_isl_dataset(self.config)

    def train_dataloader(self):
        return data.DataLoader(self.training_set, **self.train_params)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, **self.val_params)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, **self.test_params)


class CSLR_DataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.test_params = {'batch_size': config.dataloader.test.batch_size,
                            'shuffle': False,
                            'num_workers': 2}
        self.val_params = {'batch_size': config.dataloader.val.batch_size,
                           'shuffle': config.dataloader.val.shuffle,
                           'num_workers': config.dataloader.val.num_workers,
                           'pin_memory': True}

        self.train_params = {'batch_size': config.dataloader.train.batch_size,
                             'shuffle': True,
                             'num_workers': config.dataloader.train.num_workers,
                             'pin_memory': True}

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        self.training_set, self.val_set, self.test_set, self.classes, self.id2w = cslr_datasets(self.config)

    def train_dataloader(self):
        return data.DataLoader(self.training_set, **self.train_params)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, **self.val_params)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, **self.test_params)



class SLDetection_DataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.test_params = {'batch_size': config.dataloader.test.batch_size,
                            'shuffle': False,
                            'num_workers': 2}
        self.val_params = {'batch_size': config.dataloader.val.batch_size,
                           'shuffle': config.dataloader.val.shuffle,
                           'num_workers': config.dataloader.val.num_workers,
                           'pin_memory': True}

        self.train_params = {'batch_size': config.dataloader.train.batch_size,
                             'shuffle': True,
                             'num_workers': config.dataloader.train.num_workers,
                             'pin_memory': True}

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        self.training_set, self.val_set, self.test_set, self.classes, self.id2w = cslr_datasets(self.config)

    def train_dataloader(self):
        return data.DataLoader(self.training_set, **self.train_params)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, **self.val_params)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, **self.test_params)



def data_generators(config):
    """

    Args:
        config ():

    Returns:

    """
    test_params = {'batch_size': config.dataloader.test.batch_size,
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
    test_params = {'batch_size': 4,
                   'shuffle': False,
                   'num_workers': 2}
    val_params = {'batch_size': 4,
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


def select_isl_dataset(config):
    if config.dataset.name == 'GSL_ISO':

        print("RUN ON GREEK ISOLATED")
        train_prefix = "train"
        val_prefix = "val"
        indices, classes, id2w = read_classes_file(
            os.path.join(config.cwd, 'data_loader/gsl_iso/files/iso_classes.csv'))
        print('Number of Classes {} \n \n  '.format(len(classes)))

        from data_loader.gsl_iso.dataloader_greek_isolated import GSL_ISO
        training_set = GSL_ISO(config, train_prefix, classes)

        val_set = GSL_ISO(config, 'val', classes)

        return training_set, val_set, None, classes, id2w
    elif config.dataset.name == 'GSL_ISO_SI':

        print("RUN ON GREEK SI PROT ISOLATED")
        train_prefix = "train"
        val_prefix = "val"
        indices, classes, id2w = read_classes_file(
            os.path.join(config.cwd, 'data_loader/gsl_iso/files/iso_classes.csv'))
        print('Number of Classes {} \n \n  '.format(len(classes)))

        from data_loader.gsl_iso.dataloader_gsl_si_isolated import GSL_SI
        training_set = GSL_SI(config, train_prefix, classes)

        val_set = GSL_SI(config, 'val', classes)

        return training_set, val_set, None, training_set.classes, training_set.id2w
    elif config.dataset.name == 'MULTI_ISLR':
        train_prefix = "train"
        val_prefix = "val"
        indices, classes, id2w = read_classes_file(
            os.path.join(config.cwd, 'data_loader/multi_slr/files/classes_mslr.txt'))
        training_set = Multi_SLR(config, train_prefix, classes)

        val_set = Multi_SLR(config, 'val', classes)

        return training_set, val_set, None, training_set.classes, None
    elif config.dataset.name == 'AUTSL':

        train_prefix = "train"
        validation_prefix = "val"
        _, _, classes = read_autsl_csv(os.path.join(config.cwd, 'data_loader/autsl/train_labels.csv'))
        training_set = AUTSL(config, train_prefix, classes)

        validation_set = AUTSL(config, validation_prefix, classes)

        test_set = AUTSL(config, 'test', classes)

        return training_set, validation_set, test_set, classes, None
    elif config.dataset.name == 'AUTSL_SK':

        train_prefix = "train"
        validation_prefix = "val"
        _, _, classes = read_autsl_csv(os.path.join(config.cwd, 'data_loader/autsl/train_labels.csv'))
        training_set = AUTSLSkeleton(config, train_prefix, classes)

        validation_set = AUTSLSkeleton(config, validation_prefix, classes)

        test_set = AUTSLSkeleton(config, 'test', classes)

        return training_set, validation_set, test_set, classes, None
    elif config.dataset.name == 'AUTSL_RGB_SK':

        train_prefix = "train"
        validation_prefix = "val"
        _, _, classes = read_autsl_csv(os.path.join(config.cwd, 'data_loader/autsl/train_labels.csv'))
        training_set = AUTSL_RGB_SK(config, train_prefix, classes)

        validation_set = AUTSL_RGB_SK(config, validation_prefix, classes)

        test_set = AUTSL_RGB_SK(config, 'test', classes)

        return training_set, validation_set, test_set, classes, None
    elif config.dataset.name == 'AUTSL_RGBD_SK':

        train_prefix = "train"
        validation_prefix = "val"
        _, _, classes = read_autsl_csv(os.path.join(config.cwd, 'data_loader/autsl/train_labels.csv'))
        training_set = AUTSL_RGBD_SK(config, train_prefix, classes)

        validation_set = AUTSL_RGBD_SK(config, validation_prefix, classes)

        test_set = AUTSL_RGBD_SK(config, 'test', classes)

        return training_set, validation_set, test_set, classes, None
    elif config.dataset.name == 'WASL':
        training_set = WASLdataset(config, 'train')

        test_set = WASLdataset(config, 'test')

        return training_set, None, test_set, classes, None


def cslr_datasets(config):
    test_params = {'batch_size': config.dataloader.test.batch_size,
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
    if config.dataset.name == 'GSL_SI':
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(config.cwd, 'data_loader/gsl/files/continuous_classes.csv'))
        w2id = {v: k for k, v in id2w.items()}

        training_set = GSL_SI(config=config, mode=train_prefix, classes=classes)

        validation_set = GSL_SI(config=config, mode=val_prefix, classes=classes)

        test_set = GSL_SI(config=config, mode=test_prefix, classes=classes)

        return training_set, validation_set, test_set, classes, id2w
    elif config.dataset.name == 'GSL_SI2':
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(config.cwd, 'data_loader/gsl/files/continuous_classes.csv'))
        w2id = {v: k for k, v in id2w.items()}
        from data_loader.gsl.dataloader_gsl_siv2 import GSL_SIv2
        training_set = GSL_SIv2(config=config, mode=train_prefix, classes=classes)

        validation_set = GSL_SIv2(config=config, mode=val_prefix, classes=classes)

       # test_set = GSL_SIv2(config=config, mode=test_prefix, classes=classes)

        return training_set, validation_set, None, classes, id2w
    elif config.dataset.name == 'GSLW':
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(config.cwd, 'data_loader/gsl/files/continuous_classes.csv'))
        w2id = {v: k for k, v in id2w.items()}

        training_set = GSLW(config=config, mode=train_prefix, classes=classes)

        validation_set = GSLW(config=config, mode=val_prefix, classes=classes)

        # test_set = GSL_SI(config=config,  mode=test_prefix, classes=classes)
        test_generator = None  # data.DataLoader(test_set, **test_params)

        return training_set, validation_set, test_set, classes, id2w
    elif config.dataset.name == 'GSL_SI_Skeleton':
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(config.cwd, 'data_loader/gsl/files/continuous_classes.csv'))
        w2id = {v: k for k, v in id2w.items()}

        training_set = GSL_SI_Skeleton(config=config, mode=train_prefix, classes=classes)

        validation_set = GSL_SI_Skeleton(config=config, mode=val_prefix, classes=classes)

        test_set = GSL_SI_Skeleton(config=config, mode=test_prefix, classes=classes)

        return training_set, validation_set, test_set, classes, id2w
    elif config.dataset.name == 'GSL_SI_MModal':
        from data_loader.gsl.dataloader_gsl_si_mmodal import GSL_SI_MModal
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(config.cwd, 'data_loader/gsl/files/continuous_classes.csv'))
        w2id = {v: k for k, v in id2w.items()}

        training_set = GSL_SI_MModal(config=config, mode=train_prefix, classes=classes)

        validation_set = GSL_SI_MModal(config=config, mode=val_prefix, classes=classes)

        test_set = GSL_SI_MModal(config=config, mode=test_prefix, classes=classes)

        return training_set, validation_set, test_set, classes, id2w



def sldetection_datasets(config):
    test_params = {'batch_size': config.dataloader.test.batch_size,
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

    if config.dataset.name == 'GSL_SI2':
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(config.cwd, 'data_loader/gsl/files/continuous_classes.csv'))
        w2id = {v: k for k, v in id2w.items()}
        from data_loader.gsl.dataloader_gsl_siv2 import GSL_SIv2
        training_set = GSL_SIv2(config=config, mode=train_prefix, classes=classes)

        validation_set = GSL_SIv2(config=config, mode=val_prefix, classes=classes)

       # test_set = GSL_SIv2(config=config, mode=test_prefix, classes=classes)

        return training_set, validation_set, None, classes, id2w
    elif config.dataset.name == 'GSLW':
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(config.cwd, 'data_loader/gsl/files/continuous_classes.csv'))
        w2id = {v: k for k, v in id2w.items()}

        training_set = GSLW(config=config, mode=train_prefix, classes=classes)

        validation_set = GSLW(config=config, mode=val_prefix, classes=classes)

        # test_set = GSL_SI(config=config,  mode=test_prefix, classes=classes)
        test_generator = None  # data.DataLoader(test_set, **test_params)

        return training_set, validation_set, test_set, classes, id2w
    elif config.dataset.name == 'GSL_SI_Skeleton':
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(config.cwd, 'data_loader/gsl/files/continuous_classes.csv'))
        w2id = {v: k for k, v in id2w.items()}

        training_set = GSL_SI_Skeleton(config=config, mode=train_prefix, classes=classes)

        validation_set = GSL_SI_Skeleton(config=config, mode=val_prefix, classes=classes)

        test_set = GSL_SI_Skeleton(config=config, mode=test_prefix, classes=classes)

        return training_set, validation_set, test_set, classes, id2w
    elif config.dataset.name == 'GSL_SI_MModal':
        from data_loader.gsl.dataloader_gsl_si_mmodal import GSL_SI_MModal
        train_prefix = "train"
        val_prefix = "val"
        test_prefix = "test"

        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(config.cwd, 'data_loader/gsl/files/continuous_classes.csv'))
        w2id = {v: k for k, v in id2w.items()}

        training_set = GSL_SI_MModal(config=config, mode=train_prefix, classes=classes)

        validation_set = GSL_SI_MModal(config=config, mode=val_prefix, classes=classes)

        test_set = GSL_SI_MModal(config=config, mode=test_prefix, classes=classes)

        return training_set, validation_set, test_set, classes, id2w
