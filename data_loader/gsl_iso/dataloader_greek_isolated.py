import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import glob
import os
import random
from omegaconf import OmegaConf
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pathlib
from base.base_data_loader import Base_dataset
from data_loader.loader_utils import multi_label_to_index, pad_video, video_transforms, sampling, VideoRandomResizedCrop,class2indextensor

root_path = 'Greek_isolated/GSL_isol/'
train_prefix = "train"
dev_prefix = "val"
test_augmentation = 'augment'
train_filepath = "data_loader/gsl_iso/files/train_greek_iso.csv"
dev_filepath = "data_loader/gsl_iso/files/dev_greek_iso.csv"



class GSL_ISO(Base_dataset):
    def __init__(self, config,  mode, classes):
        """

        Args:
            mode : train or test and path prefix to read frames acordingly
            classes : list of classes
            channels: Number of channels of frames
            dim: Dimensions of the frames
        """
        super(GSL_ISO,self).__init__(config,  mode, classes)
        #config = OmegaConf.load(os.path.join(config.cwd, "data_loader/gsl_iso/dataset.yml"))['dataset']

        self.dim = self.config.dataset.dim
        self.num_classes = len(classes)
        self.seq_length = self.config.dataset[self.mode]['seq_length']
        self.normalize = self.config.dataset.normalize
        self.padding = self.config.dataset.padding
        self.augmentation = self.config.dataset[self.mode]['augmentation']
        print('Classes {}'.format(len(classes)))
        cwd_path = self.config.cwd
        self.bbox = read_bounding_box(os.path.join(cwd_path,'data_loader/gsl_iso/files/bbox_for_gsl_isolated.txt'))
        # print(self.bbox)
        if mode == train_prefix:
            self.list_video_paths, self.list_glosses = read_gsl_isolated(os.path.join(cwd_path,train_filepath))
            print("{} {} instances  ".format(len(self.list_video_paths), mode))
            self.mode = mode
        elif mode == dev_prefix:
            self.list_video_paths, self.list_glosses = read_gsl_isolated(os.path.join(cwd_path,dev_filepath))
            # print(self.list_video_paths)
            print("{} {} instances  ".format(len(self.list_video_paths), mode))
            self.mode = mode
        elif mode == test_augmentation:
            self.list_video_paths, self.list_glosses = read_gsl_isolated(os.path.join(cwd_path,test_filepath))
            # print(self.list_video_paths)
            print("{} {} instances  ".format(len(self.list_video_paths), mode))
            self.mode = 'train'

        self.data_path = self.config.dataset.input_data + 'GSL_isolated/Greek_isolated'

    def __len__(self):
        return len(self.list_video_paths)

    def __getitem__(self, index):

        # print(self.list_glosses[index])
        #print(self.list_glosses[index])
        y = class2indextensor(classes=self.classes, target_label=self.list_glosses[index])
        # y = multi_label_to_index1(classes=self.classes, target_labels=self.list_glosses[index])
        # print(folder_path)
        x = self.load_video_sequence(index, time_steps=self.seq_length, dim=self.dim,
                                     augmentation=self.mode, padding=self.padding, normalize=self.normalize,
                                     img_type='jpg')
        #print(x.shape,y.shape)

        return x, y

    def load_video_sequence(self, index, time_steps, dim=(224, 224), augmentation='test', padding=False, normalize=True,
                            img_type='png'):

        path = os.path.join(self.data_path, self.list_video_paths[index])
        images = sorted(glob.glob(os.path.join(path, '*' + img_type)))
        #print(path,len(images))
        h_flip = False
        img_sequence = []
        # print(len(images))
        # print(self.bbox)
        bbox = None#self.bbox.get(self.list_video_paths[index])
        # print(self.list_video_paths[index],self.bbox.get(self.list_video_paths[index]))

        if (augmentation == 'train'):
            ## training set temporal  AUGMENTATION
            temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
            if (temporal_augmentation > 15):
                images = sorted(random.sample(images, k=temporal_augmentation))
            if (len(images) > time_steps):
                # random frame sampling
                images = sorted(random.sample(images, k=time_steps))

        else:
            # test uniform sampling
            if (len(images) > time_steps):
                images = sorted(sampling(images, time_steps))
        # print(images)
        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)
        brightness = 1 + random.uniform(-0.2, +0.2)
        contrast = 1 + random.uniform(-0.2, +0.2)
        hue = random.uniform(0, 1) / 10.0
        r_resize = ((256, 256))
        crop_or_bbox = random.uniform(0, 1) > 0.5
        to_flip = random.uniform(0, 1) > 0.5
        grayscale = random.uniform(0, 1) > 0.8


        if (len(images) == 0):
            print('frames zero ', path)

        t1 = VideoRandomResizedCrop(dim[0], scale=(0.8, 2), ratio=(0.8, 1.2))
        for img_path in images:

            frame_o = Image.open(img_path)
            frame_o.convert('RGB')

            crop_size = 120
            ## CROP BOUNDING BOX

            frame1 = np.array(frame_o)
            # if augmentation == 'test':
            #     if bbox != None:
            #         frame1 = frame1[:, bbox['x1']:bbox['x2']]
            #     else:
            #         frame1 = frame1[:, crop_size:648 - crop_size]
            # else:
            #
            #     if crop_or_bbox:
            #         frame1 = frame1[:, crop_size:648 - crop_size]
            #     elif bbox != None:
            #         frame1 = frame1[:, bbox['x1']:bbox['x2']]
            #     else:
            #print(frame1.shape)
            frame1 = frame1[:, crop_size:648 - crop_size]
            frame = Image.fromarray(frame1)
            if (augmentation == 'train'):

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame, bright=brightness, cont=contrast, h=hue,
                                              resized_crop=t1,
                                              augmentation=True,
                                              normalize=normalize,to_flip=to_flip,grayscale=grayscale)
                img_sequence.append(img_tensor)
            else:
                # TEST set  NO DATA AUGMENTATION
                frame = frame.resize(dim)

                img_tensor = video_transforms(img=frame,  bright=1, cont=1, h=0,  augmentation=False,
                                              normalize=normalize)
                img_sequence.append(img_tensor)
        pad_len = time_steps - len(images)

        X1 = torch.stack(img_sequence).float()

        if (padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='images')

        # X2 = X1[1:]
        # k = X1[-1]
        # X2 = torch.stack((X2,k))
        # print(X2.shape)
        X1 = X1.permute(1,0,2,3)
        #print(X1.shape)

        return X1
def read_gsl_isolated(csv_path):
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



def read_bounding_box(path):
    bbox = {}
    data = open(path, 'r').read().splitlines()
    for item in data:
        # p#rint(item)
        if (len(item.split('|')) < 2):
            print(item.split('|'))
            print("\n {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n".format(item, path, path))
        path, coordinates = item.split('|')
        coords = coordinates.split(',')
        # print(coords)
        x1, x2, y1, y2 = int(coords[0].split(':')[-1]), int(coords[1].split(':')[-1]), int(
            coords[2].split(':')[-1]), int(coords[3].split(':')[-1])
        # print(x1,x2,y1,y2)
        ks = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

        bbox[path] = ks
        # bbox[path]['x2'] = x2
        # bbox[path]['y1'] = y1
        # bbox[path]['y2'] = y2

        # print(a)

    # bbox.append(a)
    return bbox
def read_classes_file(path):
    indices, classes = [], []

    data = open(path, 'r').read().splitlines()
    count = 1
    for d in data:
        label = d

        indices.append(count)
        classes.append(label)
        count += 1

    id2w = dict(zip(indices, classes))

    return indices, classes, id2w