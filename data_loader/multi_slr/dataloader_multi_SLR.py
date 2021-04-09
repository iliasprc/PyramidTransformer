import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from base.base_data_loader import Base_dataset
from data_loader.loader_utils import class2indextensor
from data_loader.loader_utils import pad_video, video_transforms, sampling, VideoRandomResizedCrop



def read_files(csv_path):
    paths, glosses_list = [], []
    classes = []
    data = open(csv_path, 'r').read().splitlines()
    for item in data:
        if (len(item.split('|')) < 2):
            print("\n {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n".format(item, path, csv_path))
        path, gloss = item.split('|')

        paths.append(path)

        glosses_list.append(gloss)

    return paths, glosses_list


train_filepath = "data_loader/multi_slr/files/train_mslr1.txt"
dev_filepath = "data_loader/multi_slr/files/test_mslr1.txt"
train_prefix = 'train'
test_prefix = 'val'

#
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
#
# torch.cuda.manual_seed(SEED)


class Multi_SLR(Base_dataset):
    def __init__(self, config, mode,classes ):
        super(Multi_SLR,self).__init__( config, mode,classes)
        """

        Args:
            mode : train or test and path prefix to read frames acordingly
            classes : list of classes
            channels: Number of channels of frames
            dim: Dimensions of the frames
        """
        self.classes = classes
        print('Classes {}'.format(len(classes)))

        # print(self.bbox)
        if mode == train_prefix:
            self.list_video_paths, self.list_glosses = read_files(os.path.join(self.config.cwd,train_filepath))
            print("{} {} instances  ".format(len(self.list_video_paths), mode))
            self.mode = mode
        elif mode == test_prefix:
            self.list_video_paths, self.list_glosses = read_files(os.path.join(self.config.cwd,dev_filepath))
            # print(self.list_video_paths)
            print("{} {} instances  ".format(len(self.list_video_paths), mode))
            self.mode = mode


        self.data_path = self.config.dataset.input_data
        self.dim = self.config.dataset.dim
        self.seq_length = self.config.dataset[self.mode]['seq_length']
        self.normalize = self.config.dataset.normalize
        self.padding = self.config.dataset.padding

    def __len__(self):
        return len(self.list_video_paths)

    def __getitem__(self, index):

        #index=0
        y = class2indextensor(classes=self.classes, target_label=self.list_glosses[index])
        #print(self.list_glosses[index],y)
        x = self.load_video_sequence(index, time_steps=self.seq_length, dim=self.dim,
                                     mode=self.mode, padding=self.padding, normalize=self.normalize,
                                     img_type='jpg')

        return x, y

    def load_video_sequence(self, index, time_steps, dim=(224, 224), mode='test', padding=False, normalize=True,
                            img_type='png'):

        path = os.path.join(self.data_path,self.list_video_paths[index])
        #print(path)
        imagespng = sorted(glob.glob(os.path.join(path, '*png' )))
        imagesjpg = sorted(glob.glob(os.path.join(path, '*jpg' )))
        if(len(imagesjpg)>len(imagespng)):
            images = imagesjpg
        else:
            images = imagespng
        # normalize = True
        # augmentation = 'test'
        h_flip = False
        img_sequence = []
        # print(images)
        if (mode == 'train'):
            ## training set temporal  AUGMENTATION
            temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
            if (temporal_augmentation > 15):
                images = sorted(sampling(images, temporal_augmentation))
            if (len(images) > time_steps):
                # random frame sampling
                images = sorted(sampling(images, time_steps))

        else:
            # test uniform sampling
            if (len(images) > time_steps):
                images = sorted(sampling(images, time_steps))
        # print(images)
        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)

        brightness = 1 + random.uniform(-0.2, +0.2)
        contrast = 1 + random.uniform(-0.2, +0.2)
        hue = random.uniform(0, 1) / 5.0
        to_flip = random.uniform(0, 1) > 0.5
        grayscale = random.uniform(0, 1) > 0.8
        r_resize = ((256, 256))
        angle = np.random.randint(-15, 15)

        # brightness = 1
        # contrast = 1
        # hue = 0
        t1 = VideoRandomResizedCrop(dim[0], scale=(0.8, 1.2), ratio=(0.8, 1.2))
        for img_path in images:

            frame = Image.open(img_path)
            frame.convert('RGB')

            ## CROP BOUNDING BOX
            if 'GSL_ISO' in path:
                crop_size = 120
                frame = np.array(frame)
                frame = frame[:, crop_size:648 - crop_size]
                frame = Image.fromarray(frame)
            if (mode == 'train'):

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame,  bright=brightness, cont=contrast, h=hue,
                                              resized_crop=t1,
                                              augmentation=True,
                                              normalize=normalize,to_flip=to_flip,grayscale=grayscale,angle=angle)
                img_sequence.append(img_tensor)
            else:
                # TEST set  NO DATA AUGMENTATION
                frame = frame.resize(dim)

                img_tensor = video_transforms(img=frame, bright=0, cont=0, h=0,  augmentation=False,
                                              normalize=normalize)
                img_sequence.append(img_tensor)
        pad_len = time_steps - len(images)
        if (len(img_sequence) < 1):
            #img_sequence.append(torch.zeros((3, dim[0], dim[0])))
            print('empty ',path)
            X1 = torch.stack(img_sequence).float()
            print(X1.shape)
            X1 = pad_video(X1, padding_size=pad_len - 1, padding_type='images')
        # print(len(mages))
        elif (padding):
            X1 = torch.stack(img_sequence).float()
            # print(X1.shape)
            X1 = pad_video(X1, padding_size=pad_len, padding_type='images')
        #print(X1.shape)

        # X2 = X1[1:]
        # k = X1[-1].unsqueeze(0)
        # X2 = torch.cat((X2,k))
        # #print(X2.shape)
        # X1 = X1-X2
        return X1.permute(1,0,2,3)
