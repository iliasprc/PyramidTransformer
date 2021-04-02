from torch.utils.data import Dataset
from data_loader.loader_utils import class2indextensor, load_video_sequence

import json, glob

def select_asl_subset(path, N):
    id2w = dict()
    classes = []
    with open(path, 'r') as jf:
        data = json.load(jf)
        for idx, line in enumerate(data):
            video_url = line['url']
            signer = line['signer_id']
            height = line['height']
            width = line['width']
            label = line['label']
            text_label = line['text']
            fps_json = line['fps']
            start_time = line['start_time']
            end_time = line['end_time']
            start = line['start']
            end = line['end']
            box = line['box']
            vid_name = str(line['url']).split('/')[-1]

            if label < N:
                if text_label not in classes:
                    classes.append(text_label)
                    id2w[text_label] = label
        print(" Training Subset ASL {}".format(N))
        return id2w, classes


def get_subset_paths_labels(path, split, classes):
    gloss_folders = glob.glob(path + split + '/*')
    examples = []
    labels = []
    for fold in gloss_folders:
        label = fold.split('_')[-1]
        if label in classes:
            labels.append(label)
            examples.append(fold)
    c = list(set(labels))
    print('Found {} classes in dataset {} split'.format(len(c), split))
    print("{} examples {} labels".format(len(examples), len(labels)))
    return examples, labels

dataset_path = '/home/papastrat/Desktop/ms_asl_dataset/'
mode = ['train', 'val', 'test']
train_path = '/home/papastrat/PycharmProjects/PyramidTransformer/data_loader/ms_asl/MSASL_train.json'
val_path = '/home/papastrat/PycharmProjects/PyramidTransformer/data_loader/ms_asl/MSASL_val.json'



class MS_ASL_dataset(Dataset):
    def __init__(self, mode, seq_length=64, dim=(224, 224), n_channels=3, subset=100, normalize=True, padding=True):
        """
        Args:
            mode : train or test and path prefix to read frames acordingly
            seq_length : Number of frames to be loaded in a sample
            dim: Dimensions of the frames
            subset : select 100 or 200 or 500 or 1000 classes
            normalize : normalize tensor with imagenet mean and std
            padding : padding of video to size seq_length

        """
        self.seq_length = seq_length
        self.mode = mode
        self.images_path = dataset_path
        self.channels = n_channels
        self.seq_length = 32
        self.dim = dim
        self.normalize = normalize
        self.padding = padding
        classes_number = subset
        """ select subset of N classes """
        _, self.classes = select_asl_subset(train_path, classes_number)
        self.list_IDs, self.labels = get_subset_paths_labels(dataset_path, self.mode, self.classes)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        y = class2indextensor(classes=self.classes, target_label=self.labels[index])
        x = load_video_sequence(path=self.list_IDs[index], time_steps=self.seq_length, dim=self.dim,
                                augmentation=self.mode, padding=self.padding, normalize=self.normalize,
                                img_type='jpg', training_random_sampling=False)
        #print('tensor ',x.shape)

        return x, y
