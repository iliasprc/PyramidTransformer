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
from data_loader.loader_utils import multi_label_to_index, pad_video, video_transforms, sampling, \
    VideoRandomResizedCrop, \
    class2indextensor
from data_loader.gsl_iso.dataloader_greek_isolated import read_classes_file


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
        # print(scenario)
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
train_filepath = "data_loader/gsl/files/gsl_split_SI_train.csv"
val_filepath = "data_loader/gsl/files/gsl_split_SI_dev.csv"
test_filepath = "data_loader/gsl/files/gsl_split_SI_test.csv"


def read_gsl_si_isolated(csv_path):
    trainpaths, train_labels = [], []
    train_glosses = []
    val_glosses = []
    valpaths, val_labels = [], []
    classes = []
    gloss_classes = []
    id2w = {'blank': 0}
    c = 0
    data = open(csv_path, 'r').read().splitlines()
    for item in data:
        if (len(item.split(',')) < 3):
            print("\n {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n".format(item, path, csv_path))
        path, prot, gloss = item.split(',')
        prot = prot.strip()
        gloss = gloss.strip()
        for g in gloss.split(' '):
            g = g.strip()
            if g not in gloss_classes:
                gloss_classes.append(g)
                c += 1
                id2w[g] = c
        if prot not in classes:
            classes.append(prot)
        if 'signer3' not in path:
            trainpaths.append(path)
            train_glosses.append(gloss)
            train_labels.append(prot)
        else:
            valpaths.append(path)
            val_glosses.append(gloss)
            val_labels.append(prot)
    classes = sorted(classes)
    gloss_classes = sorted(gloss_classes)
    gloss_classes.insert(0,'blank')
    # for i in classes:
    #      print(i)
    gloss_w2id = {v: k for k, v in id2w.items()}
    return trainpaths, train_labels, train_glosses, valpaths, val_labels, val_glosses, classes, gloss_classes, \
           gloss_w2id


class GSL_SI(Base_dataset):
    def __init__(self, config, mode):
        """

        Args:
            config:
            args:
            mode:
            classes:
        """
        super(GSL_SI, self).__init__(config, mode, None)

        cwd_path = config.cwd

        self.modality = self.config.dataset.modality
        self.mode = mode
        self.dim = self.config.dataset.dim

        self.seq_length = self.config.dataset[self.mode]['seq_length']
        self.normalize = self.config.dataset.normalize
        self.padding = False  # self.config.dataset.padding
        self.augmentation = self.config.dataset[self.mode]['augmentation']
        self.return_context = False

        trainpaths, train_labels, train_glosses, valpaths, val_labels, val_glosses,sclasses, gloss_classes, \
        gloss_w2id = read_gsl_si_isolated(
            os.path.join(config.cwd, 'data_loader/gsl_iso/continuous_protaseis_glossesv2.csv'))

        indices, classes, sentence_id2w = read_classes_file(
            os.path.join(config.cwd, 'data_loader/gsl_iso/prot_classesv2.csv'))
        print(len(classes),len(sclasses))
        for i in range(len(sclasses)):
            #print(sclasses[i],'-',classes[i])
            sentence_chars = [c for c in sclasses[i]]
            prot_classes_chars = [c for c in classes[i]]
            assert len(sentence_chars) == len(prot_classes_chars),print(f'{sclasses[i]}|{classes[i]}' )
            assert sclasses[i]==classes[i],print(f'{sclasses[i]}|{classes[i]}' )
        self.sentence_id2w = sentence_id2w
        self.gloss_to_index = gloss_w2id
        self.bbox = read_bounding_box(os.path.join(config.cwd, 'data_loader/gsl/files/bbox_for_gsl_continuous.txt'))
        self.classes = classes
        self.num_classes = len(classes)
        self.gloss_classes = gloss_classes
        print(f'NUMBER OF CLASSES PROT {len(self.classes)}   GLOSS   {len(gloss_classes)}')
        print(f'GLOSSES {gloss_classes}')
        print(f'PROT {classes}')
        if self.mode == train_prefix:
            self.list_IDs, self.list_sent, self.list_glosses = trainpaths, train_labels, train_glosses

        elif self.mode == val_prefix:
            self.list_IDs, self.list_sent, self.list_glosses = valpaths, val_labels, val_glosses

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

    def feature_loader(self, index):
        folder_path = os.path.join(self.data_path, self.list_IDs[index])
        # print(folder_path)

        y = multi_label_to_index(classes=self.classes, target_labels=self.list_glosses[index])
        y = multi_label_to_index(classes=self.classes, target_labels=self.list_glosses[index])
        x = torch.FloatTensor(np.load(folder_path + '.npy')).squeeze(0)

        return x, y

    def video_loader(self, index):
        # print(self.list_IDs[index],self.list_sent[index],self.list_glosses[index])
        x = self.load_video_sequence(path=self.list_IDs[index],
                                     img_type='jpg')
        y = class2indextensor(classes=self.classes, target_label=self.list_sent[index])
        y_g = multi_label_to_index(classes=self.gloss_classes, target_labels=self.list_glosses[index])
        return x, (y, y_g)

    def load_video_sequence(self, path,
                            img_type='png'):

        if (self.augmentation):
            if random.uniform(0, 1) > 0.5:
                path = path.replace('GSL_NEW', 'GSL_continuous')
        images = sorted(glob.glob(os.path.join(self.data_path, path, ) + '/*' + img_type))

        h_flip = False
        #        print(os.path.join(self.data_path, path))
        img_sequence = []
        # print(images)
        if (len(images) < 1):
            print(os.path.join(self.data_path, path))

        bbox = self.bbox.get(path)
        # print(bbox)
        # print(path)
        if (self.augmentation):
            ## training set temporal  AUGMENTATION
            temporal_augmentation = int((np.random.randint(65, 100) / 100.0) * len(images))
            if (temporal_augmentation > 15):
                images = sorted(random.sample(images, k=temporal_augmentation))
            if (len(images) > self.seq_length):
                # random frame sampling
                images = sorted(random.sample(images, k=self.seq_length))

        else:
            # test uniform sampling
            if (len(images) > self.seq_length):
                images = sorted(sampling(images, self.seq_length))

        brightness = 1 + random.uniform(-0.2, +0.2)
        contrast = 1 + random.uniform(-0.2, +0.2)
        hue = random.uniform(0, 1) / 5.0
        r_resize = ((256, 256))

        to_flip = random.uniform(0, 1) > 0.5
        grayscale = random.uniform(0, 1) > 0.9
        t1 = VideoRandomResizedCrop(self.dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))

        bbox_scalex = random.uniform(0, 1) * 0.5  # + 0.2
        x1 = bbox['x1']
        x2 = bbox['x2']
        y1 = bbox['y1']
        y2 = bbox['y2']
        bbox_scaley = random.uniform(0, 1) * 0.5  # + 0.2
        x1 = int(max(0, x1 - int(bbox_scalex * x1)))
        x2 = int(min(648, x2 + int(bbox_scalex * x2)))
        y1 = int(max(0, y1 - int(bbox_scaley * y1)))
        y2 = int(min(480, y2 + int(bbox_scaley * y2)))

        for img_path in images:

            frame_o = Image.open(img_path)
            frame_o.convert('RGB')

            crop_size = 120
            ## CROP BOUNDING BOX
            ## CROP BOUNDING BOX

            frame1 = np.array(frame_o)
            # print(frame1.shape)
            # print(bbox)
            # print(bbox)
            if bbox != None:

                # print('dfasdfdsf')
                frame1 = frame1[y1:y2, x1:x2]
                # cv2.imshow()
            else:
                frame1 = frame1[:, crop_size:648 - crop_size]
            frame = Image.fromarray(frame1)

            if self.augmentation:

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame, bright=brightness, cont=contrast, h=hue,

                                              resized_crop=t1,
                                              augmentation=True,
                                              normalize=self.normalize, to_flip=to_flip, grayscale=grayscale
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
            X1 = pad_video(X1, padding_size=pad_len, padding_type='images')
        if (len(images) < 25):
            X1 = pad_video(X1, padding_size=25 - len(images), padding_type='images')
        # print(X1.shape)
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

        return X1.permute(1, 0, 2, 3)
