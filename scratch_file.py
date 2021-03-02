from vidaug import augmentors as va
import torchvision
import numpy as np
import cv2
import random
import torch.nn.functional as F

import torch
import numpy as np
import random

import skvideo
import skvideo.io

from PIL import Image
from torchvision import transforms
import glob
import os

import csv
import cv2
import math
import time
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
from torch.utils import data

import torch.optim as optim




from torchvision import models




def load_csv_file(path):
    """

    Args:
        path ():

    Returns:

    """
    data_paths = []
    labels = []
    with open(path) as fin:
        reader = csv.reader(fin)
        data = list(reader)
    for item in data:
        data_paths.append(item[0])
        labels.append(item[1])
    return data_paths, labels

p1 = '/mnt/784C5F3A4C5EF1FC/PROJECTS/Ipthl/SLR_challenge/checkpoints/predictions.csv'
p2 =  '/mnt/784C5F3A4C5EF1FC/PROJECTS/Ipthl/SLR_challenge/checkpoints/model_Pyramid_Transformer/dataset_Autsl/predictions.csv'
p3 = '/mnt/784C5F3A4C5EF1FC/PROJECTS/Ipthl/SLR_challenge/checkpoints/predictions90.csv/predictions.csv'

paths , labels = load_csv_file(p1)
paths2 , labels2 = load_csv_file(p2)
paths3 , labels3 = load_csv_file(p3)
true_valid_paths = []
true_valid_labels = []
for i in range(len(paths)):
    assert paths[i] == paths2[i]
   # print(labels[i],labels2[i])

    if  (labels[i]==labels3[i]) or labels2[2]==labels3[i]:
        true_valid_paths.append(paths[i])
        true_valid_labels.append(labels[i])
    else:
        print("FAGGHFDGHFS")


print(len(true_valid_paths))

def write_csv(data, name):
    """

    Args:
        data ():
        name ():
    """
    with open(name, 'w') as fout:
        for item in data:
            print(item)

            fout.write(f'{item[0]},{item[1]}\n')



valid_lias = list(zip(true_valid_paths,true_valid_labels))

write_csv(valid_lias,'/mnt/784C5F3A4C5EF1FC/PROJECTS/Ipthl/SLR_challenge/data_loader/autsl/valid_lias_labels.csv')
print(len(paths))




# 
# class VideoRandomResizedCrop(object):
#     """Crop the given PIL Image to random size and aspect ratio.
#     A crop of random size (default: of 0.08 to 1.0) of the original size and a random
#     aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
#     is finally resized to given size.
#     This is popularly used to train the Inception networks.
#     Args:
#         size: expected output size of each edge
#         scale: range of size of the origin size cropped
#         ratio: range of aspect ratio of the origin aspect ratio cropped
#         interpolation: Default: PIL.Image.BILINEAR
#     """
# 
#     def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
#         if isinstance(size, tuple):
#             self.size = size
#         else:
#             self.size = (size, size)
# 
#         self.interpolation = interpolation
#         self.scale = scale
#         self.ratio = ratio
#         self.parameters = self.get_params(self.scale, self.ratio)
# 
#     @staticmethod
#     def get_params(scale, ratio):
#         """Get parameters for ``crop`` for a random sized crop.
#         Args:
#             img (PIL Image): Image to be cropped.
#             scale (tuple): range of size of the origin size cropped
#             ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for a random
#                 sized crop.
#         """
#         area = 256 * 256
# 
#         for attempt in range(10):
#             target_area = random.uniform(*scale) * area
#             log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
#             aspect_ratio = math.exp(random.uniform(*log_ratio))
# 
#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))
# 
#             if w <= 256 and h <= 256:
#                 i = random.randint(0, 256 - h)
#                 j = random.randint(0, 256 - w)
#                 return i, j, h, w
# 
#         # Fallback to central crop
#         in_ratio = 256 / 256
#         if (in_ratio < min(ratio)):
#             w = 256
#             h = w / min(ratio)
#         elif (in_ratio > max(ratio)):
#             h = 256
#             w = h * max(ratio)
#         else:  # whole image
#             w = 256
#             h = 256
#         i = (256 - h) // 2
#         j = (256 - w) // 2
# 
#         return i, j, h, w
# 
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): Image to be cropped and resized.
#         Returns:
#             PIL Image: Randomly cropped and resized image.
#         """
#         i, j, h, w = self.parameters
#         return transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)
# 
# 
# 
# def sampling(clip,size):
#     return_ind = [int(i) for i in np.linspace(1, len(clip), num=size)]
# 
#     return [clip[i - 1] for i in return_ind]
# 
# def load_video_from_mp4( path, time_steps=181, dim=(224, 224), augmentation='test', padding=False, normalize=True,
#                         ):
#     capture = cv2.VideoCapture(path)
# 
#     read_flag, frame = capture.read()
#     img_sequence = []
# 
#     augmentation = 'test'
#     i = np.random.randint(0, 30)
#     j = np.random.randint(0, 30)
# 
#     brightness = 1 + random.uniform(-0.1, +0.1)
#     contrast = 1 + random.uniform(-0.1, +0.1)
#     hue = random.uniform(0, 1) / 50.0
# 
#     t1 = VideoRandomResizedCrop(dim[0], scale=(0.9, 1.0), ratio=(0.9, 1.1))
# 
#     while (read_flag):
# 
#         if (augmentation == 'train'):
#             pillow_frame = Image.fromarray(np.uint8(cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB)))
#             img_tensor = video_transforms(path, img=pillow_frame, i=i, j=j, bright=brightness, cont=contrast,
#                                                h=hue,
#                                                dim=dim, resized_crop=t1, augmentation='train', normalize=normalize)
# 
# 
#         else:
#             pillow_frame = Image.fromarray(np.uint8(cv2.resize(frame, (224, 224))))
#             img_tensor = video_transforms(path, img=pillow_frame, i=i, j=j, bright=0, cont=0, h=0, dim=dim,
#                                                augmentation='test',
#                                                normalize=normalize)
# 
#         img_sequence.append(img_tensor)
#         read_flag, frame = capture.read()
# 
# 
# 
#     new_list = sampling(img_sequence, time_steps)
# 
#     if (len(img_sequence) > time_steps):
# 
#         new_list = sampling(img_sequence, time_steps)
#     else:
#         new_list = img_sequence
#     new_list = img_sequence
#     if (len(new_list) == 0):
#         X1 = torch.zeros(time_steps, 3, dim[0], dim[1])
#     else:
#         X1 = torch.stack(new_list).float()
#     pad_len = time_steps - len(new_list)
#     # if (padding):
#     #     X1 = to
# 
#     return X1
# 
# 
# def video_transforms( path, img, i, j, bright, cont, h, dim=(224, 224), resized_crop=None, augmentation='test',
#                      normalize=True):
#     ##### LATER FOR FLIP
#     # ## img = img.transpose(Image.FLIP_LEFT_RIGHT)
# 
#     if (augmentation == 'train'):
#         t = transforms.ToTensor()
#         img = transforms.functional.resized_crop(img, i=i, j=j, h=dim[0], w=dim[1], size=dim)
#         img = transforms.functional.adjust_brightness(img, bright)
#         img = transforms.functional.adjust_contrast(img, cont)
#         img = transforms.functional.adjust_hue(img, h)
# 
# 
#     else:
#         t = transforms.ToTensor()
# 
#     # img1 = t(img).permute(1, 2, 0).numpy()
#     # import cv2
#     #
#     # cv2.imshow(path.split('/')[-1], cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
#     # cv2.waitKey(100)
#     if (normalize):
#         norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#     else:
# 
#         norm = transforms.Normalize(mean=[0, 0, 0],
#                                     std=[1, 1, 1])
# 
#     t1 = norm(t(img))
# 
#     return t1
# 
# 
# 
# 
# path = '/mnt/784C5F3A4C5EF1FC/PROJECTS/datasets/GSL_test_data/test_keng_v2/health1/γεια σας τι έχετε.wmv'
# t = 1
# if t==1:
# 
# 
# 
#     sometimes = lambda aug: va.Sometimes(0.1, aug) # Used to apply augmentor with 50% probability
#     seq = va.Sequential([
#         va.CenterCrop((480, 640)),
#         va.RandomCrop(size=(480, 640)), # rand,omly crop video with a size of (240 x 180)
#         #va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]
#         sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
#     ])
# 
# 
# 
#     for batch_idx in range(10):
#         video_array, _, _ = torchvision.io.read_video(path)
#         T, H, W, C = video_array.shape
#         video_array = video_array.permute(3,0,1,2).unsqueeze(0)
#         video_array=        torch.nn.functional.interpolate(video_array,(2*T,H,W)).squeeze().permute(1,2,3,0)
#         print(video_array.shape)
#         video_array = video_array.numpy().astype(np.uint8)
# 
#         T, H, W, C = video_array.shape
#         video_aug = seq(video_array)
#         print(video_array.shape,video_aug[2].shape)
#         print(video_aug[2])
#         for i in range(T):
#             cv2.imshow('f', video_aug[i].astype(np.uint8))#cv2.cvtColor(video_aug[100], cv2.COLOR_RGB2BGR))
#             cv2.waitKey(10)
# elif t==2:
#     x=   load_video_from_mp4(path)
#     for batch_idx in range(1000):
#         x = load_video_from_mp4(path)
#         print(x.shape)
# 
# elif t==3:
#     from torchvideo import transforms
# 
# else:
#     from torchvideotransforms import video_transforms, volume_transforms
# 
#     video_transform_list = [video_transforms.RandomRotation(30),
#                             video_transforms.RandomCrop((200, 200))
#                             ]
#     transforms = video_transforms.Compose(video_transform_list)
# 
#     for batch_idx in range(10):
#         video_array =  skvideo.io.vread(path)
#         video_array = video_array.astype(np.uint8)
#         T, H, W, C = video_array.shape
#         video_aug = transforms(video_array)
#         print(video_array.shape, video_aug[2].shape)
#         #print(video_aug[2])
#         cv2.imshow('f', video_aug[100].astype(np.uint8))  # cv2.cvtColor(video_aug[100], cv2.COLOR_RGB2BGR))
#         cv2.waitKey(100)