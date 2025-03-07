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
from data_loader.loader_utils import multi_label_to_index, pad_video, video_transforms, sampling, VideoRandomResizedCrop,read_gsl_continuous,read_gsl_continuous_classes
import skvideo.io

feats_path = 'gsl_cont_features/'
train_prefix = "train"
val_prefix = "val"
test_prefix = "test"
train_filepath = "data_loader/gsl/files/gsl_split_SI_train.csv"
val_filepath = "data_loader/gsl/files/gsl_split_SI_dev.csv"
test_filepath = "data_loader/gsl/files/gsl_split_SI_test.csv"


class GSLW(Base_dataset):
    def __init__(self, config,  mode, classes):
        """

        Args:
            config:
            args:
            mode:
            classes:
        """
        super(GSLW, self).__init__(config, mode, classes)

        cwd_path = config.cwd

        self.modality = self.config.dataset.modality
        self.mode = mode
        self.dim = self.config.dataset.dim
        self.num_classes = len(classes)
        self.seq_length = self.config.dataset[self.mode]['seq_length']
        self.normalize = self.config.dataset.normalize
        self.padding = self.config.dataset.padding
        self.augmentation = self.config.dataset[self.mode]['augmentation']
        #self.return_context = False
        
        train,train_labels,val,val_labels = load_gslw(os.path.join(config.cwd,'data_loader/gsl/files/gsl_wild.csv'))
        if self.mode == train_prefix:
            self.list_IDs, self.list_glosses = train,train_labels  # us(os.path.join(config.cwd, train_filepath))

        elif self.mode == val_prefix:
            self.list_IDs, self.list_glosses =val,val_labels  #read_gsl_continuous(os.path.join(config.cwd, val_filepath))


        print(f"{len(self.list_IDs)} {self.mode} instances")


        if (self.modality == 'RGB'):
            self.data_path = os.path.join(self.config.dataset.input_data,'')
            self.get = self.video_loader


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        return self.get(index)

    def feature_loader(self, index):
        folder_path = os.path.join(self.data_path, self.list_IDs[index])
        # print(folder_path)

        y = multi_label_to_index(classes=self.classes, target_labels=self.list_glosses[index])


        x = torch.FloatTensor(np.load(folder_path + '.npy')).squeeze(0)

        return x, y

    def video_loader(self, index):

        x = self.load_video_sequence(path=os.path.join(self.data_path,self.list_IDs[index]),
                                     img_type='jpg')
        y = multi_label_to_index(classes=self.classes, target_labels=self.list_glosses[index])

        return x, y

    def load_video_sequence(self, path,
                            img_type='png'):



        frame_list = []
        saved_name = path

        videogen = skvideo.io.vreader(saved_name)
        videometadata = skvideo.io.ffprobe(saved_name)

        if 'video' not in videometadata.keys():
            print(saved_name)




        h_flip = False
#        print(os.path.join(self.data_path, path))
        img_sequence = []


        #
        # if (len(images) > self.seq_length):
        #     # random frame sampling
        #     images = sorted(random.sample(images, k=self.seq_length))



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
        for i,frame in enumerate(videogen):


            frame = Image.fromarray(frame)

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
        pad_len = self.seq_length - i

        X1 = torch.stack(img_sequence).float()

        if (self.padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='images')
        if (i < 16):
            X1 = pad_video(X1, padding_size=16 - i, padding_type='images')
        #print(X1.shape)
        return X1.permute(1,0,2,3)



def load_gslw(path='./gsl_wild.csv'):
    i = 0
    with open(path, 'r') as f:
        data = f.read().splitlines()
    paths, glosses, scenaria = [], [], []
    signers = []
    starts = []
    ends = []
    flips = []
    train_p = []
    train_glosses = []
    val_paths = []
    val_glosses = []
    train_signers = ['signer5', 'signer7','signer2','boss','signer3', 'signer1']
    for idx,i in enumerate(data):
        if 'ΕΓΩ_ΔΙΝΩ_ΕΣΕΝΑ ΠOΤΕ_' in i:
            continue
        if 'ΣΥΜΠΛΗΡΩΝΩ ΣΥΝ ΓΡΑΦΩ ΒΙΒΛΙΟ ΕΝΣΗΜΑ ΙΚΑ(Δ.Α.) ΧΑΝΩ' in i:
            continue 
        if 'ΕΣΥ ΠΡΕΠΕΙ ΑΙΤΗΣΗ ΒΙΒΛΙΟ ΕΝΣΗΜΑ' in i:
            continue
        if ' ΓΙΑ ΒΙΒΛΙΟ ΕΝΣΗΜΑ ΙΚΑ(Δ.Α.)' in i:
            continue
        if 'wmv' not in i:
            path,  g,signer,start,end,flip = i.split(',')

            if signer in train_signers:
                train_p.append(path.replace('GSL_in_the_wild','GSLW'))
                train_glosses.append(g)
            else:
                val_paths.append(path.replace('GSL_in_the_wild','GSLW'))
                val_glosses.append(g)
    return train_p,train_glosses,val_paths,val_glosses
