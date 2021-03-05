import glob
import os
import re

import torch

from base.base_data_loader import Base_dataset
from data_loader.loader_utils import load_video, read_autsl_labelsv2


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


class AUTSL(Base_dataset):
    def __init__(self, config, mode, classes):
        """

        Args:
            config:
            args:
            mode:
            classes:
        """
        super(AUTSL, self).__init__(config, mode, classes)
        print(config)
        self.modality = self.config.dataset.modality
        self.mode = mode
        self.dim = self.config.dataset.dim
        self.num_classes = self.config.dataset.classes
        self.seq_length = self.config.dataset[self.mode]['seq_length']
        self.normalize = self.config.dataset.normalize
        self.padding = self.config.dataset.padding
        self.augmentation = self.config.dataset[self.mode]['augmentation']
        if self.modality == 'RGB':
            self.modal = 'color'
        else:
            self.modal = 'depth'
        self.list_rgb = sorted(glob.glob(os.path.join(config.cwd, f'challenge/train/*{self.modal}.mp4')))

        train_paths, train_labels, _ = read_autsl_labelsv2(
            './data_loader/autsl/train_labels.csv')
        val_paths, val_labels, _ = read_autsl_labelsv2('./data_loader/autsl/ground_truth_validation.csv')
        if mode == 'train':
            self.list_IDs = train_paths
            self.labels = train_labels
        elif mode == 'val':
            self.list_IDs = val_paths
            self.labels = val_labels
        elif mode == 'test':
            self.labels = []
            self.list_IDs = natural_sort(
                glob.glob(os.path.join(self.config.dataset.input_data, f'challenge/test/*{self.modal}.mp4')))

            for path in self.list_IDs:
                label = path.replace(self.config.dataset.input_data, '').replace('challenge/test/', '').replace(
                    f'_{self.modal}.mp4',
                    '')
                self.labels.append(label)

        print(f'Samples {len(self.list_IDs)} {self.mode} modality {self.modal}')

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        if self.mode == 'train':
            aug = 'train'

            vid_name = f'{self.list_IDs[index]}_{self.modal}.mp4'
            vid_tensor = load_video(os.path.join(self.config.dataset.input_data, 'challenge/train', vid_name),
                                    dim=self.dim,
                                    time_steps=self.seq_length, augmentation=aug)
        elif self.mode == 'val':
            aug = 'test'
            vid_name = f'{self.list_IDs[index]}_{self.modal}.mp4'
            vid_tensor = load_video(os.path.join(self.config.dataset.input_data, 'challenge/val', vid_name),
                                    dim=self.dim,
                                    time_steps=self.seq_length, augmentation=aug)

        elif self.mode == 'validation':
            aug = 'test'
            vid_name = f'{self.list_IDs[index]}_{self.modal}.mp4'
            vid_tensor = load_video(self.list_IDs[index], dim=self.dim,
                                    time_steps=self.seq_length, augmentation=aug)
            return vid_tensor, self.labels[index]
        target_label = torch.LongTensor([int(self.labels[index])])

        return vid_tensor, target_label
