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
from base.base_data_loader import Base_dataset
from data_loader.loader_utils import multi_label_to_index,sampling_mode, pad_video, video_transforms, sampling, VideoRandomResizedCrop,read_gsl_continuous,read_gsl_continuous_classes
from data_loader.gsl.utils import load_gsl_json
def read_bounding_box(path):
    bbox = {}
    data = open(path, 'r').read().splitlines()
    for item in data:
        # p#rint(item)
        if (len(item.split('|')) < 2):
            print(item.split('|'))
            print("\n {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n".format(item, path, path))
        path, coordinates = item.split('|')
        scenario = path.split('_')[0]
        #print(scenario)
        path = f'{scenario}/{path}'
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

feats_path = 'gsl_cont_features/'
train_prefix = "train"
val_prefix = "val"
test_prefix = "test"
train_filepath = "data_loader/gsl/files/train.json"
val_filepath = "data_loader/gsl/files/val.json"
test_filepath = "data_loader/gsl/files/gsl_split_SI_test.csv"


class GSL_SI(Base_dataset):
    def __init__(self, config,  mode, classes):
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
        self.bbox = read_bounding_box(os.path.join(config.cwd,'data_loader/gsl/files/bbox_for_gsl_continuous.txt'))
        if self.mode == train_prefix:
            self.list_IDs, self.list_glosses= load_gsl_json(os.path.join(config.cwd, train_filepath))

        elif self.mode == val_prefix:
            self.list_IDs, self.list_glosses = load_gsl_json(os.path.join(config.cwd, val_filepath))

        elif self.mode == test_prefix:
            self.list_IDs, self.list_glosses = read_gsl_continuous(os.path.join(config.cwd, test_filepath))

        print(f"{len(self.list_IDs)} {self.mode} instances")


        if (self.modality == 'RGB'):
            self.data_path = os.path.join(self.config.dataset.input_data, 'GSL_NEW')
            self.get = self.video_loader
        elif (self.modality == 'features'):
            self.data_path = os.path.join(self.config.dataset.input_data, '')

            self.get = self.feature_loader

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        return self.get(index)


    def video_loader(self, index):

        x,y = self.load_video_sequence(path=self.list_IDs[index],label = self.list_glosses[index],
                                     img_type='jpg')

        return x, y

    def load_video_sequence(self, path,label,
                            img_type='png'):

        images = sorted(glob.glob(os.path.join(self.data_path, path, ) + '/*' + img_type))
        num_of_images = list(range(len(images)))
        h_flip = False
#        print(os.path.join(self.data_path, path))
        img_sequence = []
        # print(images)
        if (len(images) < 1):
            print(os.path.join(self.data_path, path))

        bbox = self.bbox.get(path)
        #print(bbox)
        #print(path)
        if (self.augmentation):
            ## training set temporal  AUGMENTATION
            temporal_augmentation = int((np.random.randint(70, 100) / 100.0) * len(images))
            num_of_images = sampling_mode(True, num_of_images, temporal_augmentation)
            if len(num_of_images) > self.seq_length:
                num_of_images = sampling_mode(True, num_of_images, self.seq_length)

        else:
            # test uniform sampling
            if len(num_of_images) > self.seq_length:
                num_of_images = sampling_mode(True, num_of_images, self.seq_length)

        images = [images[i] for i in num_of_images]
        print(images)
        print(num_of_images)
        boundaries= label['boundaries']
        labels = label['glosses']
        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)
        brightness = 1 + random.uniform(-0.2, +0.2)
        contrast = 1 + random.uniform(-0.2, +0.2)
        hue = random.uniform(0, 1) / 10.0
        r_resize = ((256, 256))
        crop_or_bbox = random.uniform(0, 1) > 0.5
        to_flip = random.uniform(0, 1) > 0.5
        grayscale = random.uniform(0, 1) > 0.9
        t1 = VideoRandomResizedCrop(self.dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))
        for img_path in images:

            frame_o = Image.open(img_path)
            frame_o.convert('RGB')

            crop_size = 120
            ## CROP BOUNDING BOX
            ## CROP BOUNDING BOX

            frame1 = np.array(frame_o)
            #print(frame1.shape)
            #print(bbox)

            if bbox != None:
                bbox_scalex = random.uniform(0, 1)*0.2
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1  = bbox['y1']
                y2 = bbox['y2']
                bbox_scaley = random.uniform(0, 1)*0.2
                x1 = int(max(0, x1 - int(bbox_scalex * x1)))
                x2 = int(min(648, x2 + int(bbox_scalex * x2)))
                y1 = int(max(0, y1 - int(bbox_scaley * y1)))
                y2 = int(min(480, y2 + int(bbox_scaley * y2)))
                #print('dfasdfdsf')
                frame1 = frame1[y1:y2, x1:x2]
                #cv2.imshow()
            else:
                frame1 = frame1[:, crop_size:648 - crop_size]
            frame = Image.fromarray(frame1)

            if self.augmentation:

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame, bright=brightness, cont=contrast, h=hue,

                                              resized_crop=t1,
                                              augmentation=True,
                                              normalize=self.normalize, to_flip=to_flip,grayscale=grayscale
                                              )
                img_sequence.append(img_tensor)
            else:
                # TEST set  NO DATA AUGMENTATION
                frame = frame.resize(self.dim)

                img_tensor = video_transforms(img=frame,  bright=1, cont=1, h=0,
                                              augmentation=False,
                                              normalize=self.normalize)
                img_sequence.append(img_tensor)
        pad_len = self.seq_length - len(images)

        X1 = torch.stack(img_sequence).float()

        if (self.padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='images')
        if (len(images) < 25):
            X1 = pad_video(X1, padding_size=25 - len(images), padding_type='images')
        #print(X1.shape)
        # import torchvision
        # import cv2
        # X2 = X1[1:]
        # k = X1[-1].unsqueeze(0)
        # X1 = X1-torch.cat((X2,k))
        # # for i in range(len(X1)-1):
        # #     tensor = X1[i]# -X1[i+1]
        # #     #tensor2 =
        # #     inv_normalize = torchvision.transforms.Normalize(
        # #         mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        # #         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        # #     )
        # #     inv_tensor = inv_normalize(tensor).permute(1,2,0).numpy()
        # #     cv2.imshow('Frame',cv2.cvtColor(inv_tensor,cv2.COLOR_BGR2RGB))
        # #     cv2.waitKey(30)
        
        return X1.permute(1,0,2,3),None


