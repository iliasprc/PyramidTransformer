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

from data_loader.loader_utils import class2indextensor
from data_loader.loader_utils import pad_video, video_transforms, sampling, VideoRandomResizedCrop, read_gsl_isolated


train_filepath = "../files/multi_slr/train_mslr1.txt"
dev_filepath = "../files/multi_slr/test_mslr1.txt"
train_prefix = 'train'
test_prefix = 'test'

#
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
#
# torch.cuda.manual_seed(SEED)


class Multi_SLR(Dataset):
    def __init__(self, args, mode, classes, dim=(224, 224)):
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
            self.list_video_paths, self.list_glosses = read_gsl_isolated(train_filepath)
            print("{} {} instances  ".format(len(self.list_video_paths), mode))
            self.mode = mode
        elif mode == test_prefix:
            self.list_video_paths, self.list_glosses = read_gsl_isolated(dev_filepath)
            # print(self.list_video_paths)
            print("{} {} instances  ".format(len(self.list_video_paths), mode))
            self.mode = mode


        self.root_path = args.input_data
        self.seq_length = args.seq_length
        self.dim = dim
        self.normalize = args.normalize
        self.padding = args.padding

    def __len__(self):
        return len(self.list_video_paths)

    def __getitem__(self, index):


        y = class2indextensor(classes=self.classes, target_label=self.list_glosses[index])

        x = self.load_video_sequence(index, time_steps=self.seq_length, dim=self.dim,
                                     augmentation='test', padding=self.padding, normalize=self.normalize,
                                     img_type='jpg')

        return x, y

    def load_video_sequence(self, index, time_steps, dim=(224, 224), augmentation='test', padding=False, normalize=True,
                            img_type='png'):

        path = os.path.join(self.root_path,self.list_video_paths[index])
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
        if (augmentation == 'train'):
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

        brightness = 1 + random.uniform(-0.1, +0.1)
        contrast = 1 + random.uniform(-0.1, +0.1)
        hue = random.uniform(0, 1) / 20.0

        r_resize = ((256, 256))

        # brightness = 1
        # contrast = 1
        # hue = 0
        t1 = VideoRandomResizedCrop(dim[0], scale=(0.9, 1.0), ratio=(0.9, 1.1))
        for img_path in images:

            frame = Image.open(img_path)
            frame.convert('RGB')

            if (augmentation == 'train'):

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=brightness, cont=contrast, h=hue, dim=dim,
                                              resized_crop=t1,
                                              augmentation='train',
                                              normalize=normalize)
                img_sequence.append(img_tensor)
            else:
                # TEST set  NO DATA AUGMENTATION
                frame = frame.resize(dim)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=0, cont=0, h=0, dim=dim, augmentation='test',
                                              normalize=normalize)
                img_sequence.append(img_tensor)
        pad_len = time_steps - len(images)
        if (len(img_sequence) < 1):
            #img_sequence.append(torch.zeros((3, dim[0], dim[0])))
            print('empty ',path)
            X1 = torch.stack(img_sequence).float()
            print(X1.shape)
            X1 = pad_video(X1, padding_size=pad_len - 1, padding_type='zeros')
        # print(len(mages))
        elif (padding):
            X1 = torch.stack(img_sequence).float()
            # print(X1.shape)
            X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
        #print(X1.shape)
        return X1.permute(1,0,2,3)
