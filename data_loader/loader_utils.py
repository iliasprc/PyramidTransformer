import glob
import math
import os
import random

import numpy as np
import skvideo.io
import torch
from PIL import Image
from torchvision import transforms



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


def read_autsl_csv(csv_path):
    paths, glosses_list = [], []
    classes = []
    data = open(csv_path, 'r').read().splitlines()
    for item in data:
        # print(item)
        path, gloss = item.split(',')

        paths.append(path)

        glosses_list.append(gloss)
    classes = sorted(list(set(glosses_list)))
    return paths, glosses_list, classes


def read_autsl_labels(csv_path):
    # signers_val = ['signer11', 'signer16', 'signer18', 'signer1', 'signer25', 'signer35']
    signers_train = ['signer0', 'signer10', 'signer12', 'signer13', 'signer15', 'signer17', 'signer19', 'signer20',
                     'signer21', 'signer22', 'signer23', 'signer24', 'signer26', 'signer28', 'signer29', 'signer2',
                     'signer31', 'signer32', 'signer33', 'signer36', 'signer37', 'signer38', 'signer3', 'signer40',
                     'signer41', 'signer42', 'signer4', 'signer5', 'signer7', 'signer8', 'signer9']

    signers_val = signers_train[0:6]

    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    classes = []
    data = open(csv_path, 'r').read().splitlines()
    for item in data:

        # print(item)
        path, gloss = item.split(',')
        signer = path.split('_')[0]
        # print(signer)
        if signer in signers_val:
            # print(signer)
            val_paths.append(path)
            val_labels.append(gloss)
        else:
            train_paths.append(path)
            train_labels.append(gloss)

    classes = sorted(list(set(train_labels + val_labels)))
    # import glob
    # val = sorted(glob.glob('/home/iliask/Desktop/ilias/datasets/challenge/train/*_color.mp4'))
    # print(len(val))
    # signers_val =[]
    # for i in val:
    #     i = i.replace('/home/iliask/Desktop/ilias/datasets/challenge/train/','')
    #     signer = i.split('_')[0]
    #     if signer not in signers_val:
    #         signers_val.append(signer)
    #     print(signer)
    # print(signers_val)

    return train_paths, train_labels, val_paths, val_labels, classes


def read_autsl_labelsv2(csv_path):
    #signers_val = ['signer11', 'signer16', 'signer18', 'signer1', 'signer25', 'signer35']
    signers_train = ['signer0', 'signer10', 'signer12', 'signer13', 'signer15', 'signer17', 'signer19', 'signer20',
                     'signer21', 'signer22', 'signer23', 'signer24', 'signer26', 'signer28', 'signer29', 'signer2',
                     'signer31', 'signer32', 'signer33', 'signer36', 'signer37', 'signer38', 'signer3', 'signer40',
                     'signer41', 'signer42', 'signer4', 'signer5', 'signer7', 'signer8', 'signer9']

    signers_val = signers_train[0:6]

    train_paths, train_labels = [], []
    classes = []
    data = open(csv_path, 'r').read().splitlines()
    for item in data:
        # print(item)
        path, gloss = item.split(',')
        signer = path.split('_')[0]
        # print(signer)

        train_paths.append(path)
        train_labels.append(gloss)

    classes = sorted(list(set(train_labels)))
    # import glob
    # val = sorted(glob.glob('/home/iliask/Desktop/ilias/datasets/challenge/train/*_color.mp4'))
    # print(len(val))
    # signers_val =[]
    # for i in val:
    #     i = i.replace('/home/iliask/Desktop/ilias/datasets/challenge/train/','')
    #     signer = i.split('_')[0]
    #     if signer not in signers_val:
    #         signers_val.append(signer)
    #     print(signer)
    # print(signers_val)

    return train_paths, train_labels, classes


def sampling(clip, size):
    """
    uniform downsampling of video frames
    Args:
        clip (): list of images
        size (): output size

    Returns:

    """
    return_ind = [int(i) for i in np.linspace(1, len(clip), num=size)]
    return [clip[i - 1] for i in return_ind]


def sampling_mode(random_sampling, images, size):
    """
    Random or uniform downsampling of video frames
    Args:
        random_sampling ():
        images (): list of images
        size (): output size

    Returns:

    """
    return sorted(random.sample(images, k=size)) if random_sampling \
        else sorted(sampling(images, size=size))


def multi_label_to_index(classes, target_labels):
    """
    Convert labels to target tensor
    Args:
        classes ():
        target_labels ():

    Returns:

    """
    indexes = []

    for word in target_labels.strip().split(' '):
        indexes.append(classes.index(word))
    return torch.tensor(indexes, dtype=torch.int)


def multi_label_to_index_out_of_vocabulary(classes, target_labels):
    """
    Convert labels to target tensor ignoring OOV labels
    Args:
        classes ():
        target_labels ():

    Returns:

    """
    indexes = []
    for word in target_labels.split(' '):

        if word in classes:
            indexes.append(classes.index(word))
    return torch.tensor(indexes, dtype=torch.int)


def class2indextensor(classes, target_label):
    indexes = classes.index(target_label)
    return torch.tensor([indexes], dtype=torch.long)


def pad_video(x, padding_size=0, padding_type='images'):
    """
    Pad video using zeros or first frame
    Args:
        x ():
        padding_size ():
        padding_type ():

    Returns:

    """
    # print(padding_size ,'pad size')
    assert len(x.shape) == 4
    if padding_size != 0:
        if padding_type == 'images':
            if random.uniform(0, 1) > 0.5:
                pad_img = x[0]
                padx = pad_img.repeat(padding_size, 1, 1, 1)
                X = torch.cat((padx, x))
            else:
                pad_img = x[-1]
                padx = pad_img.repeat(padding_size, 1, 1, 1)
                X = torch.cat(( x,padx))
            return X
        elif padding_type == 'zeros':
            T, C, H, W = x.size()
            padx = torch.zeros((padding_size, C, H, W))
            X = torch.cat((padx, x))
            return X
    return x



def skeleton_augment(x):
    if random.uniform(0, 1) > 0.5:
        # flip skeleton x and y only
        x[...,0:2] = -x[...,0:2]
    if random.uniform(0, 1) > 0.7:
        new_pos = torch.rand(2)/20.0
        x[...,0:2]+=new_pos

    return x



def pad_skeleton(x, padding_size=0):
    assert len(x.shape) == 3
    if padding_size != 0:
        if random.uniform(0, 1) > 0.5:
            pad_img = x[0]
            padx = pad_img.repeat(padding_size, 1, 1)
            X = torch.cat((padx, x),dim=0)
        else:
            pad_img = x[-1]
            padx = pad_img.repeat(padding_size, 1, 1)
            X = torch.cat((x, padx),dim=0)
    return X

def video_tensor_shuffle(x):
    # print(x.size())

    r = x[:, 0, :, :]
    g = x[:, 1, :, :]
    b = x[:, 2, :, :]

    # print(test)
    rgb = [r, g, b]
    # print(r[1,0:10,0:10])#,b[1,0:10,0:10],g[1,0:10,0:10])
    random.shuffle(rgb)

    # print(r[1,50:60,50:60],'\n\n',r1[1, 50:60, 50:60])#, b[1, 0:10, 0:10], g[1, 0:10, 0:10])
    x = torch.stack(rgb, dim=1)

    return x


def video_transforms(img, bright, cont, h, resized_crop=None, augmentation=False, normalize=True, to_flip=False,grayscale= False,angle=0):
    """
    Image augmentation function
    Args:
        img ():
        bright (): Brightness value
        cont (): Contrast value
        h (): Hue value
        resized_crop (): Resize img
        augmentation (): Apply augmentation in training only
        normalize (): Normalize image

    Returns:

    """
    if augmentation:
        t = transforms.ToTensor()
        if angle !=0:
            img = transforms.functional.rotate(img,angle)
        if grayscale:
            img = transforms.functional.to_grayscale(img,3)
        if to_flip:
            img = transforms.functional.hflip(img)
        img = resized_crop(img)
        #print(f'resize {img.size}')
        img = transforms.functional.adjust_brightness(img, bright)
        img = transforms.functional.adjust_contrast(img, cont)
        img = transforms.functional.adjust_hue(img, h)
    else:
        t = transforms.ToTensor()
    if normalize:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    else:
        norm = transforms.Normalize(mean=[0, 0, 0],
                                    std=[1, 1, 1])
    t1 = norm(t(img))
    return t1


def load_video_sequence(path, time_steps, dim=(224, 224), augmentation='test', padding=False, normalize=True,
                        scale=(0.9, 1.0), ratio=(0.9, 1.1), abs_brightness=.1, abs_contrast=.1,
                        img_type='png', training_random_sampling=True, sign_length_check=16, gsl_extras=False):
    """
    Load image sequence
    Args:
        path ():
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
    images = sorted(glob.glob(os.path.join(path, '*' + img_type)))
    img_sequence = []
    if augmentation == 'train':
        temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
        if temporal_augmentation > 15:
            images = sampling_mode(training_random_sampling, images, temporal_augmentation)
        if len(images) > time_steps:
            images = sampling_mode(training_random_sampling, images, time_steps)
    else:
        if len(images) > time_steps:
            images = sampling_mode(False, images, time_steps)

    if not isinstance(abs_brightness, float) or not isinstance(abs_contrast, float):
        abs_brightness = float(abs_brightness)
        abs_contrast = float(abs_contrast)

    brightness = 1 + random.uniform(-abs(abs_brightness), abs(abs_brightness))
    contrast = 1 + random.uniform(-abs(abs_contrast), abs(abs_contrast))
    hue = random.uniform(0, 1) / 20.0
    r_resize = (256, 256)
    t1 = VideoRandomResizedCrop(dim[0], scale, ratio)
    for img_path in images:
        frame = Image.open(img_path)
        frame.convert('RGB')

        if augmentation == 'train':

            frame = frame.resize(r_resize)
            img_tensor = video_transforms(img=frame, bright=brightness, cont=contrast, h=hue,
                                          resized_crop=t1,
                                          augmentation='train',
                                          normalize=normalize)
        else:
            frame = frame.resize(dim)
            img_tensor = video_transforms(img=frame, bright=1, cont=1, h=0, augmentation='test',
                                          normalize=normalize)
        img_sequence.append(img_tensor)
    pad_len = time_steps - len(images)
    tensor_imgs = torch.stack(img_sequence).float().permute(1, 0, 2, 3)

    if padding:
        tensor_imgs = pad_video(tensor_imgs, padding_size=pad_len, padding_type='images')
    elif len(images) < sign_length_check:
        tensor_imgs = pad_video(tensor_imgs, padding_size=sign_length_check - len(images), padding_type='images')

    return tensor_imgs


def load_video(path, time_steps, dim=(224, 224), augmentation='test', padding=True, normalize=True,
               scale=(0.8, 1.2), ratio=(0.8, 1.2), abs_brightness=.15, abs_contrast=.15,
               training_random_sampling=True):
    """
    Load image sequence
    Args:
        path ():
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
    video_array = skvideo.io.vread(path)
    video_array = video_array.astype(np.uint8)
    T, H, W, C = video_array.shape
    img_sequence = []
    num_of_images = list(range(T))
    if augmentation == 'train':
        temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * T)
        # elif temporal_augmentation > 15:
        num_of_images = sampling_mode(training_random_sampling, num_of_images, temporal_augmentation)
        if len(num_of_images) > time_steps:
            num_of_images = sampling_mode(training_random_sampling, num_of_images, time_steps)

        video_array = video_array[num_of_images, :, :, :]

    else:
        if T > time_steps:
            num_of_images = sampling_mode(False, num_of_images, time_steps)
            video_array = video_array[num_of_images, :, :, :]

    if not isinstance(abs_brightness, float) or not isinstance(abs_contrast, float):
        abs_brightness = float(abs_brightness)
        abs_contrast = float(abs_contrast)

    brightness = 1 + random.uniform(-abs(abs_brightness), abs(abs_brightness))
    contrast = 1 + random.uniform(-abs(abs_contrast), abs(abs_contrast))
    hue = random.uniform(0, 1) / 20.0
    r_resize = (dim[0] + 30, dim[0] + 30)

    t1 = VideoRandomResizedCrop(dim[0], scale, ratio)
    crop_h, crop_w = (400, 400)
    for idx in range(video_array.shape[0]):

        frame = Image.fromarray(video_array[idx])
        im_w, im_h = frame.size
        w1 = int(round((im_w - crop_w) / 2.))
        h1 = int(round((im_h - crop_h) / 2.))
        # frame = frame.crop((w1, h1, w1 + crop_w, h1 + crop_h))
        if augmentation == 'train':

            frame = frame.resize(r_resize)
            img_tensor = video_transforms(img=frame, bright=brightness, cont=contrast, h=hue,
                                          resized_crop=t1,
                                          augmentation='train',
                                          normalize=normalize)
        else:
            frame = frame.resize(dim)
            img_tensor = video_transforms(img=frame, bright=1, cont=1, h=0, augmentation='test',
                                          normalize=normalize)
        img_sequence.append(img_tensor)
    pad_len = time_steps - len(num_of_images)
    tensor_imgs = torch.stack(img_sequence).float()

    if padding:
        tensor_imgs = pad_video(tensor_imgs, padding_size=pad_len, padding_type='zeros')
    # elif len(num_of_images) < sign_length_check:
    #     tensor_imgs = pad_video(tensor_imgs, padding_size=sign_length_check - len(num_of_images),
    #     padding_type='zeros')
    # print(tensor_imgs.shape)
    return tensor_imgs.permute(1, 0, 2, 3)


class VideoRandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.parameters = self.get_params(self.scale, self.ratio)

    @staticmethod
    def get_params(scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = 256 * 256

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            # print(aspect_ratio,target_area)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= 256 and h <= 256:
                i = random.randint(0, 256 - h)
                j = random.randint(0, 256 - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = 256 / 256
        if in_ratio < min(ratio):
            w = 256
            h = w / min(ratio)
        elif in_ratio > max(ratio):
            h = 256
            w = h * max(ratio)
        else:  # whole image
            w = 256
            h = 256
        i = (256 - h) // 2
        j = (256 - w) // 2

        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.parameters
        img =  transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)

        return img
