import glob
import os
import random
import re

import numpy as np
import skvideo.io
import torch
from PIL import Image

from base.base_data_loader import Base_dataset
from data_loader.loader_utils import read_autsl_labelsv2, sampling_mode, VideoRandomResizedCrop, \
    video_transforms, pad_video


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


selected = np.concatenate(([0, 5, 6, 7, 8, 9, 10],
                           [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
                           [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0)


class AUTSL_RGBD_SK(Base_dataset):
    def __init__(self, config, mode, classes):
        """

        Args:
            config:
            args:
            mode:
            classes:
        """
        super(AUTSL_RGBD_SK, self).__init__(config, mode, classes)
        # print(config)
        self.modality = self.config.dataset.modality
        self.mode = mode
        self.data_dir = os.path.join(self.config.dataset.input_data,'challenge')
        self.dim = self.config.dataset.dim
        self.num_classes = self.config.dataset.classes
        self.seq_length = self.config.dataset[self.mode]['seq_length']
        self.normalize = self.config.dataset.normalize
        self.padding = self.config.dataset.padding
        self.augmentation = self.config.dataset[self.mode]['augmentation']

        self.modal = 'color'

        train_paths, train_labels, _ = read_autsl_labelsv2(
            os.path.join(config.cwd,'data_loader/autsl/train_labels.csv'))
        val_paths, val_labels, _ = read_autsl_labelsv2(os.path.join(config.cwd,'data_loader/autsl/ground_truth_validation.csv'))
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

        # print(f'Samples {len(self.list_IDs)} {self.mode} modality {self.modal}')

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        if self.mode == 'train':

            vid_name = f'{self.list_IDs[index]}_{self.modal}.mp4'
            vid_tensor,depth,sk_tensor = self.load_video(os.path.join(self.data_dir, self.mode, vid_name),
                                         os.path.join(self.data_dir,'skeleton', self.mode, vid_name+'.npy'),
                                         dim=self.dim,
                                         time_steps=self.seq_length, augmentation=True)
            target_label = torch.LongTensor([int(self.labels[index])])

            return vid_tensor,depth,sk_tensor, target_label
        elif self.mode == 'val':

            # print(self.list_IDs[index])
            vid_name = f'{self.list_IDs[index]}_{self.modal}.mp4'
            vid_tensor,depth,sk_tensor = self.load_video(os.path.join(self.data_dir, self.mode, vid_name),
                                         os.path.join(self.data_dir,'skeleton', self.mode, vid_name+'.npy'),
                                         dim=self.dim,
                                         time_steps=self.seq_length, augmentation=False)
            target_label = torch.LongTensor([int(self.labels[index])])
            return vid_tensor,depth,sk_tensor, target_label
        # elif self.mode == 'test':
        #     aug = 'test'
        #     vid_name = f'{self.list_IDs[index]}_{self.modal}.mp4'
        #     vid_tensor = self.load_video(self.list_IDs[index], dim=self.dim,
        #                                  time_steps=self.seq_length, augmentation=aug)
        #     return vid_tensor, self.labels[index]

    def load_video(self, vid_path, sk_path, time_steps, dim=(224, 224), augmentation=False, padding=True,
                   normalize=True,
                   scale=(0.8, 1.2), ratio=(0.8, 1.2), abs_brightness=.15, abs_contrast=.15,
                   training_random_sampling=True):
        """
        Load image sequence
        Args:
            vid_path ():
            time_steps ():
            dim (): Image output dimension
            augmentation (): Apply augmentation
            padding (): Pad video sequence
            normalize (): Normalize image
            scale (): Resized crop scale
            ratio (): Resized crop ratio
            abs_brightness ():
            abs_contrast ():
            img_type ():
            training_random_sampling ():
            sign_length_check ():
            gsl_extras ():

        Returns:

        """
        # video_array, _, _ = torchvision.io.read_video(path)
        # video_array = video_array.numpy().astype(np.uint8)
        video_array = skvideo.io.vreader(vid_path)
        video_meta_data = skvideo.io.ffprobe(vid_path)
        T = int(video_meta_data['video']['@nb_frames'])


        img_sequence = []
        num_of_images = list(range(T))
        if augmentation:
            temporal_augmentation = int((np.random.randint(65, 100) / 100.0) * T)

            num_of_images = sampling_mode(training_random_sampling, num_of_images, temporal_augmentation)
            if len(num_of_images) > time_steps:
                num_of_images = sampling_mode(training_random_sampling, num_of_images, time_steps)



        else:
            if T > time_steps:
                num_of_images = sampling_mode(False, num_of_images, time_steps)


        if not isinstance(abs_brightness, float) or not isinstance(abs_contrast, float):
            abs_brightness = float(abs_brightness)
            abs_contrast = float(abs_contrast)

        brightness = 1 + random.uniform(-abs(abs_brightness), abs(abs_brightness))
        contrast = 1 + random.uniform(-abs(abs_contrast), abs(abs_contrast))
        hue = random.uniform(0, 1) / 10.0
        to_flip = random.uniform(0, 1) > 0.5
        grayscale = random.uniform(0, 1) > 0.8
        r_resize = (dim[0] + 30, dim[0] + 30)

        t1 = VideoRandomResizedCrop(dim[0], scale, ratio)
        crop_h, crop_w = (400, 400)
        for idx,frame in enumerate(video_array):

            frame = Image.fromarray(frame)
            im_w, im_h = frame.size
            w1 = int(round((im_w - crop_w) / 2.))
            h1 = int(round((im_h - crop_h) / 2.))
            #print(h1,w1)
            frame = frame.crop((w1, h1, w1 + crop_w, h1 + crop_h))
            #print(frame.size)
            if augmentation:

                frame = frame.resize(r_resize)
                img_tensor = video_transforms(img=frame, bright=brightness, cont=contrast, h=hue,
                                              resized_crop=t1,
                                              augmentation=augmentation,
                                              normalize=normalize,grayscale=grayscale,to_flip=to_flip)
            else:
                frame = frame.resize(dim)
                img_tensor = video_transforms(img=frame, bright=1, cont=1, h=0, augmentation=False,
                                              normalize=normalize)
            img_sequence.append(img_tensor)
        img_sequence = [img_sequence[i] for i in num_of_images]
        pad_len = self.seq_length - len(num_of_images)
        tensor_imgs = torch.stack(img_sequence).float()

        if padding:
            tensor_imgs = pad_video(tensor_imgs, padding_size=pad_len, padding_type='images')
        #print( tensor_imgs.shape[0],skeleton_npy.shape[0])

        vidRGB = tensor_imgs.permute(1, 0, 2, 3)
        #############################################################################33


        depth_sequence = []
        depth_path = vid_path.replace('color','depth')
        depth_array = skvideo.io.vreader(depth_path)
        video_meta_data = skvideo.io.ffprobe(depth_path)
        T = int(video_meta_data['video']['@nb_frames'])
        for idx,frame in enumerate(depth_array):

            frame = Image.fromarray(frame)
            im_w, im_h = frame.size
            w1 = int(round((im_w - crop_w) / 2.))
            h1 = int(round((im_h - crop_h) / 2.))
            #print(h1,w1)
            frame = frame.crop((w1, h1, w1 + crop_w, h1 + crop_h))
            #print(frame.size)
            if augmentation:

                frame = frame.resize(r_resize)
                img_tensor = video_transforms(img=frame, bright=brightness, cont=contrast, h=hue,
                                              resized_crop=t1,
                                              augmentation=augmentation,
                                              normalize=normalize,grayscale=grayscale,to_flip=to_flip)
            else:
                frame = frame.resize(dim)
                img_tensor = video_transforms(img=frame, bright=1, cont=1, h=0, augmentation=False,
                                              normalize=normalize)
            depth_sequence.append(img_tensor)
        depth_sequence = [depth_sequence[i] for i in num_of_images]


        depth_imgs = torch.stack(depth_sequence).float()



        if padding:
            depth_imgs = pad_video(depth_imgs, padding_size=pad_len, padding_type='images')
        # print( tensor_imgs.shape[0],skeleton_npy.shape[0])

        vidDEPTH = depth_imgs.permute(1, 0, 2, 3)
        skeleton_npy = np.load(sk_path) / 512.0 - 0.5
        # print(vid_tensor.shape)
        T = skeleton_npy.shape[0]

        # if T > self.seq_length:

        skeleton_npy = skeleton_npy[num_of_images, ...]
        if padding:
            pad_len = self.seq_length-skeleton_npy.shape[0]
            #print(pad_len)
            pad = np.zeros((pad_len, 133, 3), dtype=np.float) * 0.0 - 0.5
            #print(pad.shape)
            skeleton_npy = np.concatenate((pad, skeleton_npy))


        #assert tensor_imgs.shape[0] == skeleton_npy.shape[0],print(vid_path,sk_path,pad.shape)
        #print(vidRGB.shape,vidDEPTH.shape)
        #print(vid.shape,torch.from_numpy(skeleton_npy[:, selected, :]).float().shape)
        return (vidRGB,vidDEPTH,torch.from_numpy(skeleton_npy[:, selected, :]).float())

    #
    # def get_skeleton(self, index, num_of_images):
    #
    #     if self.mode == 'train':
    #         aug = 'train'
    #
    #         vid_name = f'{self.list_IDs[index]}_color.mp4.npy'
    #         skeleton_npy = np.load(os.path.join(self.config.dataset.input_data, 'train', vid_name)) / 512.0 - 0.5
    #         # print(vid_tensor.shape)
    #         T = skeleton_npy.shape[0]
    #
    #         if len(num_of_images) > self.seq_length:
    #
    #             skeleton_npy = skeleton_npy[num_of_images, ...]
    #         else:
    #             pad = np.zeros((self.seq_length - T, 133, 3), dtype=np.float) * 0.0 - 0.5
    #
    #             skeleton_npy = np.concatenate((pad, skeleton_npy))
    #             # print(vid_tensor.shape)
    #
    #         # print(vid_tensor[:,selected,:].shape)
    #
    #         # print(vid_tensor.shape)
    #         # print(vid_tensor.max(), vid_tensor.min())
    #         target_label = torch.LongTensor([int(self.labels[index])])
    #
    #         return torch.from_numpy(skeleton_npy[:, selected, :]).float(), target_label
    #     elif self.mode == 'val':
    #
    #         # print(self.list_IDs[index])
    #         vid_name = f'{self.list_IDs[index]}_color.mp4.npy'
    #         skeleton_npy = np.load(os.path.join(self.config.dataset.input_data, 'val', vid_name)) / 512.0 - 0.5
    #         # print(vid_tensor.max(),vid_tensor.min())
    #         T = skeleton_npy.shape[0]
    #         num_of_images = list(range(T))
    #
    #         if len(num_of_images) > self.seq_length:
    #
    #             skeleton_npy = skeleton_npy[num_of_images, ...]
    #         else:
    #             pad = np.zeros((self.seq_length - T, 133, 3), dtype=np.float) * 0.0 - 0.5
    #
    #             skeleton_npy = np.concatenate((pad, skeleton_npy))
    #         #
    #         # if vid_tensor.shape[0]<T:
    #         #     print(vid_tensor.shape)
    #
    #         target_label = torch.LongTensor([int(self.labels[index])])
    #         return torch.from_numpy(skeleton_npy[:, selected, :]).float()
