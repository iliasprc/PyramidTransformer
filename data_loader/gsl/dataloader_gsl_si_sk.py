import os

import numpy as np
import torch

from base.base_data_loader import Base_dataset
from data_loader.gsl.utils import read_json_sequence
from data_loader.loader_utils import multi_label_to_index, read_gsl_continuous,pad_skeleton,skeleton_augment,sampling_mode


feats_path = 'gsl_cont_features/'
train_prefix = "train"
val_prefix = "val"
test_prefix = "test"
train_filepath = "data_loader/gsl/files/gsl_split_SI_train.csv"
val_filepath = "data_loader/gsl/files/gsl_split_SI_dev.csv"
test_filepath = "data_loader/gsl/files/gsl_split_SI_test.csv"


class GSL_SI_Skeleton(Base_dataset):
    def __init__(self, config, mode, classes):
        """

        Args:
            config:
            args:
            mode:
            classes:
        """
        super(GSL_SI_Skeleton, self).__init__(config, mode, classes)

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

        self.data_path = os.path.join(self.config.dataset.input_data, 'GSL_tf_lite_keypoints')
        # self.mean = 0.2915
        # self.std = 0.3513
        # self.count = 0
        self.mean = torch.tensor([0.498,0.507,-0.129],dtype=torch.float32)
        self.std =torch.tensor( [0.085,0.229,0.1910],dtype=torch.float32)
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        path = os.path.join(self.data_path, self.list_IDs[index])
        pose, lh, rh = read_json_sequence(path)


        y = multi_label_to_index(classes=self.classes, target_labels=self.list_glosses[index])
        # print(pose.shape, lh.shape, rh.shape)
        x = torch.from_numpy(np.concatenate((pose, lh, rh), axis=1)[:, :, :3]).float()
        T = x.shape[0]
        num_of_images = list(range(T))
        if self.mode == 'train':
            if T>self.seq_length:
                num_of_images = sampling_mode(True, num_of_images, self.seq_length)
                x = x[num_of_images,:,:]
            x = skeleton_augment(x)
        else:
            if T > self.seq_length:
                num_of_images = sampling_mode(False, num_of_images, self.seq_length)
                x = x[num_of_images,:,:]
        x = (x- self.mean)/self.std
        # self.mean+= x.mean()
        # self.mean1+=x.view(-1,3).mean(dim=0)
        # self.std1+=x.view(-1,3).std(dim=0)
        # print(self.mean1/self.count,self.std1/self.count)
        # # self.std +=x.std()
        # self.count+=1
        #print(self.mean/self.count,self.std/self.count)
        if x.shape[0] < 16:
            x = pad_skeleton(x,16 - x.shape[0])

        #     zeros = torch.from_numpy(np.zeros((16 - x.shape[0], 65, 3))).float()
        #     # print(x.shape)
        #     x = torch.cat((x, zeros), dim=0)

        return x, y

