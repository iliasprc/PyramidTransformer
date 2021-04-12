import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import glob
import os
import random
import numpy as np
import torch
from PIL import Image
from base.base_data_loader import Base_dataset
from data_loader.loader_utils import multi_label_to_index, pad_video, video_transforms, VideoRandomResizedCrop, \
    sampling_mode, pad_skeleton, skeleton_augment
from data_loader.gsl.utils import read_json_sequence

feats_path = 'gsl_cont_features/'
train_prefix = "train"
val_prefix = "val"
test_prefix = "test"
train_filepath = "data_loader/gsl/files/gsl_split_SI_train.csv"
val_filepath = "data_loader/gsl/files/gsl_split_SI_dev.csv"
test_filepath = "data_loader/gsl/files/gsl_split_SI_test.csv"


class GSL_SI(Base_dataset):
    def __init__(self, config, mode, classes):
        """

        Args:
            config:
            args:
            mode:
            classes:
        """
        super(GSL_SI, self).__init__(config, mode, classes)

        cwd_path = config.cwd

        self.modality = self.config.dataset.modality
        self.mode = mode
        self.dim = self.config.dataset.dim
        self.num_classes = len(classes)
        self.seq_length = self.config.dataset[self.mode]['seq_length']
        self.normalize = self.config.dataset.normalize
        self.padding = self.config.dataset.padding
        self.augmentation = self.config.dataset[self.mode]['augmentation']
        self.return_context = False
        if self.mode == train_prefix:
            self.list_IDs, self.list_glosses = read_gsl_continuous(os.path.join(config.cwd, train_filepath))

        elif self.mode == val_prefix:
            self.list_IDs, self.list_glosses = read_gsl_continuous(os.path.join(config.cwd, val_filepath))

        elif self.mode == test_prefix:
            self.list_IDs, self.list_glosses = read_gsl_continuous(os.path.join(config.cwd, test_filepath))

        print(f"{len(self.list_IDs)} {self.mode} instances")

        if (self.modality == 'RGB'):
            self.data_path = os.path.join(self.config.dataset.input_data, '')
            self.get = self.video_loader
        elif (self.modality == 'features'):
            self.data_path = os.path.join(self.config.dataset.input_data, '')

            self.get = self.feature_loader

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        return self.get(index)

    def video_loader(self, index):

        vid_tensor, skeleton = self.load_mmodal(path=self.list_IDs[index],
                                                img_type='jpg')
        y = multi_label_to_index(classes=self.classes, target_labels=self.list_glosses[index])

        return vid_tensor, skeleton, y

    def load_mmodal(self, path,
                    img_type='png'):

        images = sorted(glob.glob(os.path.join(self.data_path,'GSL_continuous', path, ) + '/*' + img_type))

        h_flip = False
        img_sequence = []
        # print(images)
        if (len(images) < 1):
            print(os.path.join(self.data_path, path))
        T = len(images)
        num_of_images = list(range(T))
        if (self.augmentation):
            ## training set temporal  AUGMENTATION

            if len(num_of_images) > self.seq_length:
                num_of_images = sampling_mode(True, num_of_images, self.seq_length)
                # images = images[num_of_images]

        else:
            # test uniform sampling
            if len(num_of_images) > self.seq_length:
                num_of_images = sampling_mode(False, num_of_images, self.seq_length)
        images = [images[i] for i in num_of_images]
        print(images)
        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)
        brightness = 1 + random.uniform(-0.2, +0.2)
        contrast = 1 + random.uniform(-0.2, +0.2)
        hue = random.uniform(0, 1) / 10.0
        r_resize = ((256, 256))
        crop_or_bbox = random.uniform(0, 1) > 0.5
        to_flip = random.uniform(0, 1) > 1
        grayscale = random.uniform(0, 1) > 0.9
        t1 = VideoRandomResizedCrop(self.dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))
        for img_path in images:

            frame_o = Image.open(img_path)
            frame_o.convert('RGB')

            crop_size = 120
            ## CROP BOUNDING BOX
            ## CROP BOUNDING BOX

            frame1 = np.array(frame_o)
            # print(frame1.shape)

            frame1 = frame1[:, crop_size:648 - crop_size]
            frame = Image.fromarray(frame1)

            if self.augmentation:

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame, bright=brightness, cont=contrast, h=hue,

                                              resized_crop=t1,
                                              augmentation=True,
                                              normalize=self.normalize, to_flip=to_flip,
                                              )
                img_sequence.append(img_tensor)
            else:
                # TEST set  NO DATA AUGMENTATION
                frame = frame.resize(self.dim)

                img_tensor = video_transforms(img=frame, bright=1, cont=1, h=0,
                                              augmentation=False,
                                              normalize=self.normalize)
                img_sequence.append(img_tensor)
        pad_len = self.seq_length - len(images)

        X1 = torch.stack(img_sequence).float()

        if (self.padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
        if (len(images) < 16):
            X1 = pad_video(X1, padding_size=25 - len(images), padding_type='zeros')
        # print(X1.shape)

        path = os.path.join(self.data_path,'GSL_tf_lite_keypoints', path)
        pose, lh, rh = read_json_sequence(path)

        # print(pose.shape, lh.shape, rh.shape)
        x = torch.from_numpy(np.concatenate((pose, lh, rh), axis=1)[:, :, :3]).float()
        T = x.shape[0]
        num_of_images = list(range(T))
        if self.mode == 'train':
            if T > self.seq_length:
                num_of_images = sampling_mode(True, num_of_images, self.seq_length)
                x = x[num_of_images, :, :]
            x = skeleton_augment(x)
        else:
            if T > self.seq_length:
                num_of_images = sampling_mode(False, num_of_images, self.seq_length)
                x = x[num_of_images, :, :]
        x = (x - self.mean) / self.std
        # self.mean+= x.mean()
        # self.mean1+=x.view(-1,3).mean(dim=0)
        # self.std1+=x.view(-1,3).std(dim=0)
        # print(self.mean1/self.count,self.std1/self.count)
        # # self.std +=x.std()
        # self.count+=1
        # print(self.mean/self.count,self.std/self.count)
        if x.shape[0] < 16:
            x = pad_skeleton(x, 16 - x.shape[0])

        return X1.permute(1, 0, 2, 3), x


def read_gsl_continuous_classes(path):
    indices, classes = [], []
    classes.append('blank')
    indices.append(0)
    data = open(path, 'r').read().splitlines()
    count = 1
    for d in data:
        label = d

        indices.append(count)
        classes.append(label)
        count += 1

    id2w = dict(zip(indices, classes))

    return indices, classes, id2w


def read_gsl_continuous(csv_path):
    paths, glosses_list = [], []
    classes = []
    data = open(csv_path, 'r').read().splitlines()
    for item in data:
        if (len(item.split('|')) < 2):
            print("\n {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n".format(item, path, csv_path))
        path, glosses = item.split('|')
        # path = path.replace(' GSL_continuous','GSL_continuous')

        paths.append(path)
        # print(path)

        glosses_list.append(glosses)
    return paths, glosses_list
