import skvideo.io
import torch
#import videoaugment.transforms as VA


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
    # # signers_val = ['signer11', 'signer16', 'signer18', 'signer1', 'signer25', 'signer35']
    # signers_train = ['signer0', 'signer10', 'signer12', 'signer13', 'signer15', 'signer17', 'signer19', 'signer20',
    #                  'signer21', 'signer22', 'signer23', 'signer24', 'signer26', 'signer28', 'signer29', 'signer2',
    #                  'signer31', 'signer32', 'signer33', 'signer36', 'signer37', 'signer38', 'signer3', 'signer40',
    #                  'signer41', 'signer42', 'signer4', 'signer5', 'signer7', 'signer8', 'signer9']

    # signers_val = signers_train[0:6]

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
    return torch.tensor(indexes, dtype=torch.long)


def pad_video(x, padding_size=0, padding_type='images'):
    """
    Pad video using zeros or first frame
    Args:
        x ():
        padding_size ():
        padding_type ():

    Returns:

    """
    if padding_size != 0:
        if padding_type == 'images':
            pad_img = x[0]
            padx = pad_img.repeat(padding_size, 1, 1, 1)
            X = torch.cat((padx, x))
            return X
        elif padding_type == 'zeros':
            T, C, H, W = x.size()
            padx = torch.zeros((padding_size, C, H, W))
            X = torch.cat((padx, x))
            return X
    return x


def load_videov2(path, time_steps, dim=(224, 224), augmentation='test', padding=True, normalize=True,
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

    if augmentation == 'train':
        st = VA.ComposeSpatialTransforms(
            transforms=[VA.CenterCrop(400), VA.Resize(256), VA.RandomCrop(crop_size=dim[0], img_size=256),
                        VA.RandomColorAugment(brightness=0.2, contrast=0.2, hue=0.2,
                                              saturation=0.2),
                        VA.RandomRotation(20), VA.intensity.Rescale(1. / 255.0),
                        VA.PILToTensor(),
                        VA.RearrangeTensor(),
                        VA.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        tt = VA.ComposeTemporalTransforms(
            transforms=[VA.RandomTemporalDownsample(0.8),
                        VA.TemporalScale(num_of_frames=time_steps), VA.VideoToTensor()])
        videogen = skvideo.io.vreader(path)
        video = []
        for frame in videogen:
            augmented_frame = st(frame)
            video.append(augmented_frame)
        video_tensor = tt(video)




    else:

        st = VA.ComposeSpatialTransforms(
            transforms=[VA.CenterCrop(400), VA.Resize(dim[0]), VA.intensity.Rescale(1. / 255.0),
                        VA.PILToTensor(),
                        VA.RearrangeTensor(),
                        VA.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        tt = VA.ComposeTemporalTransforms(
            transforms=[VA.TemporalScale(num_of_frames=time_steps), VA.VideoToTensor()])
        videogen = skvideo.io.vreader(path)
        video = []
        for frame in videogen:
            augmented_frame = st(frame)
            video.append(augmented_frame)

        video_tensor = tt(video)

    return video_tensor.float()
