import glob
import os
import re

import torch
import numpy as np
from base.base_data_loader import Base_dataset
from data_loader.loader_utils import load_video, read_autsl_labelsv2,sampling_mode,read_autsl_labels


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


class AUTSLSkeleton(Base_dataset):
    def __init__(self, config, mode, classes):
        """

        Args:
            config:
            args:
            mode:
            classes:
        """
        super(AUTSLSkeleton, self).__init__(config, mode, classes)
        #print(config)
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
        self.list_rgb = sorted(glob.glob(os.path.join(config.cwd, f'train/*color.mp4.npy')))

        train_paths, train_labels, _ = read_autsl_labelsv2(
            os.path.join(config.cwd,'data_loader/autsl/train_labels.csv'))
        val_paths, val_labels, _ = read_autsl_labelsv2(os.path.join(config.cwd,'data_loader/autsl/ground_truth_validation.csv'))

        # train_paths, train_labels, val_paths, val_labels, classes = read_autsl_labels('./data_loader/autsl/train_labels.csv')

        if mode == 'train':
            self.list_IDs = train_paths
            self.labels = train_labels
        elif mode == 'val':
            self.list_IDs = val_paths
            self.labels = val_labels
        elif mode == 'test':
            self.labels = []
            self.list_IDs = natural_sort(
                glob.glob(os.path.join(self.config.dataset.input_data, f'test/*{self.modal}.mp4')))

            for path in self.list_IDs:
                label = path.replace(self.config.dataset.input_data, '').replace('test/', '').replace(
                    f'_{self.modal}.mp4',
                    '')
                self.labels.append(label)

        print(f'Samples {len(self.list_IDs)} {self.mode} modality {self.modal}')
        print(self.config.dataset.input_data)
        self.data_dir = os.path.join(self.config.dataset.input_data,'challenge/skeleton')
        #self.config.dataset.input_data = os.path.join(self.config.dataset.input_data,'skeleton')
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        selected =  np.concatenate(([0,5,6,7,8,9,10],
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0)

        if self.mode == 'train':
            aug = 'train'

            vid_name = f'{self.list_IDs[index]}_color.mp4.npy'
            vid_tensor = np.load(os.path.join(self.data_dir, 'train', vid_name))/512.0 -0.5
            #print(vid_tensor.shape)
            T = vid_tensor.shape[0]
            num_of_images = list(range(T))
            if len(num_of_images) > self.seq_length:
                num_of_images = sampling_mode(True, num_of_images, self.seq_length)
                vid_tensor = vid_tensor[num_of_images, ...]
            else:
                pad = np.zeros((self.seq_length-T,133,3),dtype=np.float)*0.0-0.5

                vid_tensor = np.concatenate((pad,vid_tensor))
                #print(vid_tensor.shape)

            #print(vid_tensor[:,selected,:].shape)

            #print(vid_tensor.shape)
            #print(vid_tensor.max(), vid_tensor.min())
            target_label = torch.LongTensor([int(self.labels[index])])

            return torch.from_numpy(vid_tensor[:,selected,:]).float(), target_label
        elif self.mode == 'val':
            aug = 'test'
            #print(self.list_IDs[index])
            vid_name = f'{self.list_IDs[index]}_color.mp4.npy'
            vid_tensor = np.load(os.path.join(self.data_dir, 'val', vid_name))/512.0-0.5
            #print(vid_tensor.max(),vid_tensor.min())
            T = vid_tensor.shape[0]
            num_of_images = list(range(T))
            # num_of_images = sampling_mode(False, num_of_images, self.seq_length)
            #print(vid_tensor.shape)
            #vid_tensor = vid_tensor[num_of_images,...]

            if len(num_of_images) > self.seq_length:
                num_of_images = sampling_mode(True, num_of_images, self.seq_length)
                vid_tensor = vid_tensor[num_of_images, ...]
            else:
                pad = np.zeros((self.seq_length-T,133,3),dtype=np.float)*0.0-0.5

                vid_tensor = np.concatenate((pad,vid_tensor))
            #
            # if vid_tensor.shape[0]<T:
            #     print(vid_tensor.shape)

            target_label = torch.LongTensor([int(self.labels[index])])
            return torch.from_numpy(vid_tensor[:,selected,:]).float(), target_label
        elif self.mode == 'test':
            aug = 'test'
            vid_name = f'{self.list_IDs[index]}_{self.modal}.mp4'
            vid_tensor = load_video(self.list_IDs[index], dim=self.dim,
                                    time_steps=self.seq_length, augmentation=aug)
            return vid_tensor, self.labels[index]

