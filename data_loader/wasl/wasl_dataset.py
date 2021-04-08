import json
import math
import os
import os.path
import random
from PIL import Image
import cv2
import numpy as np
import torch
import torch.utils.data as data_utl
from data_loader.loader_utils import video_transforms,sampling, VideoRandomResizedCrop,class2indextensor,pad_video

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        try:
            img = cv2.imread(os.path.join(image_dir, vid, "image_" + str(i).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
        except:
            print(os.path.join(image_dir, vid, str(i).zfill(6) + '.jpg'))
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.)
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_rgb_frames_from_video(vid_root, vid, start, num,augmentation=False):
    if augmentation:
        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)
        brightness = 1 + random.uniform(-0.2, +0.2)
        contrast = 1 + random.uniform(-0.2, +0.2)
        hue = random.uniform(0, 1) / 10.0
        r_resize = ((256, 256))
        crop_or_bbox = random.uniform(0, 1) > 0.5
        to_flip = random.uniform(0, 1) > 0.9
        grayscale = random.uniform(0, 1) > 0.9
    else:
        brightness = 1
        contrast = 1
        hue = 0

    #print(vid_root,vid)
    video_path = os.path.join(vid_root, vid + '.mp4')
    normalize = True

    t1 = VideoRandomResizedCrop(224, scale=(0.8, 2), ratio=(0.8, 1.2))
    frames = []
    vidcap = cv2.VideoCapture(video_path)

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(min(num, int(total_frames - start))):
        success, img = vidcap.read()

        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))
        #print(img.shape)
        if not augmentation:
            img = cv2.resize(img, (224, 224))
        img = video_transforms(img=Image.fromarray(img), bright=brightness, cont=contrast, h=hue,
                                      resized_crop=t1,
                                      augmentation=augmentation,
                                      normalize=normalize, to_flip=False)
        #img = (img / 255.)
        #print(img.shape)
        frames.append(img)

    return frames


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.)
        imgy = (imgy / 255.)
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)



def read_dataset(path):
    dataset = []
    data = open(path, 'r').read().splitlines()
    for item in data:
        if (len(item.split('|')) != 5):
            raise ValueError
        dataset.append(item.split('|'))
    print(len(dataset))
    return dataset



# def make_dataset(split_file, split, root, mode, num_classes):
#     dataset = []
#     with open(split_file, 'r') as f:
#         data = json.load(f)
#
#     i = 0
#     count_skipping = 0
#     counter = 0
#
#     for vid in data.keys():
#         # print(data.keys())
#         # print(counter)
#         counter += 1
#         if split == 'train':
#             if data[vid]['subset'] not in ['train', 'val']:
#                 continue
#         else:
#             if data[vid]['subset'] != 'test':
#                 continue
#
#         vid_root = root['word']
#         src = 0
#
#         video_path = os.path.join(vid_root, vid + '.mp4')
#         if not os.path.exists(video_path):
#             continue
#
#         num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
#
#         if mode == 'flow':
#             num_frames = num_frames // 2
#
#         if num_frames - 0 < 9:
#             print("Skip video ", vid)
#             count_skipping += 1
#             continue
#
#         label = np.zeros((num_classes, num_frames), np.float32)
#
#         for l in range(num_frames):
#             c_ = data[vid]['action'][0]
#             # print(c_)
#
#             label[c_][l] = 1
#
#         if len(vid) == 5:
#             dataset.append((vid, label, src, 0, data[vid]['action'][2] - data[vid]['action'][1]))
#             print(vid, label.sum() / num_frames, src, 0, data[vid]['action'][2] - data[vid]['action'][1], num_frames)
#         elif len(vid) == 6:  ## sign kws instances
#             dataset.append((vid, label, src, data[vid]['action'][1], data[vid]['action'][2] - data[vid]['action'][1]))
#
#         i += 1
#     print("Skipped videos: ", count_skipping)
#     print(len(dataset))
#     return dataset


def get_num_class(split_file):
    classes = set()

    content = json.load(open(split_file))

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)

    return len(classes)


class WASLdataset(data_utl.Dataset):

    def __init__(self, config, mode):
        self.config = config
        if mode =='train':
            self.data = read_dataset(os.path.join(config.cwd,'data_loader/wasl/train_wasl_2000.txt'))
            self.augmentation = True
        elif mode == 'test':
            self.data = read_dataset(os.path.join(config.cwd, 'data_loader/wasl/test_wasl_2000.txt'))
            self.augmentation = False
        #self.num_classes = get_num_class(split_file)
        self.classes = list(range(2000))



        self.mode = mode
        self.root = self.config.dataset.input_data
        self.seq_length = 16

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, src, start_frame, nf = self.data[index]
        start_frame = int(start_frame)
        label = int(label)
        nf = int(nf)
        total_frames = self.seq_length

        try:
            start_f = random.randint(0, nf - total_frames - 1) + start_frame
        except ValueError:
            start_f = start_frame
        #print('AUG ',self.augmentation)
        imgs = load_rgb_frames_from_video(self.root, vid, start_f, total_frames,augmentation=self.augmentation )

        #print(len(imgs))
        X1 = torch.stack(imgs).float()
        #print(X1.shape)
        # if (self.padding):
        #     X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
        #print(label)
        y = torch.Tensor([label])
        #print(y.shape)
        #ret_img = video_to_tensor(imgs)

        return X1.permute(1,0,2,3),y

    def __len__(self):
        return len(self.data)

    # def pad(self, imgs, label, total_frames):
    #     if imgs.shape[0] < total_frames:
    #         num_padding = total_frames - imgs.shape[0]
    #
    #         if num_padding:
    #             prob = np.random.random_sample()
    #             if prob > 0.5:
    #                 pad_img = imgs[0]
    #                 pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
    #                 padded_imgs = np.concatenate([imgs, pad], axis=0)
    #             else:
    #                 pad_img = imgs[-1]
    #                 pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
    #                 padded_imgs = np.concatenate([imgs, pad], axis=0)
    #     else:
    #         padded_imgs = imgs
    #
    #     # label = label[:, 0]
    #     # label = np.tile(label, (total_frames, 1)).transpose((1, 0))
    #
    #     return padded_imgs

    # @staticmethod
    # def pad_wrap(imgs, label, total_frames):
    #     if imgs.shape[0] < total_frames:
    #         num_padding = total_frames - imgs.shape[0]
    #
    #         if num_padding:
    #             pad = imgs[:min(num_padding, imgs.shape[0])]
    #             k = num_padding // imgs.shape[0]
    #             tail = num_padding % imgs.shape[0]
    #
    #             pad2 = imgs[:tail]
    #             if k > 0:
    #                 pad1 = np.array(k * [pad])[0]
    #
    #                 padded_imgs = np.concatenate([imgs, pad1, pad2], axis=0)
    #             else:
    #                 padded_imgs = np.concatenate([imgs, pad2], axis=0)
    #     else:
    #         padded_imgs = imgs
    #
    #     label = label[:, 0]
    #     label = np.tile(label, (total_frames, 1)).transpose((1, 0))
    #
    #     return padded_imgs, label
    #
