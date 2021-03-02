import csv
import json
import logging
import os
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch
import yaml

logging.captureWarnings(True)


def read_json(fname):
    """

    Args:
        fname ():

    Returns:

    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def get_logger(name):
    """Logs a message
    Args:
    name(str): name of logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)
    return logger


def make_dirs_if_not_present(path):
    """
    creates new directory if not present
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(config_file):
    """

    Args:
        config_file ():

    Returns:

    """
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def ensure_dir(dirname):
    """
    Args:
        dirname ():
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    """

    Args:
        fname ():

    Returns:

    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    """

    Args:
        content ():
        fname ():
    """
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader.

    Args:
        data_loader ():
    '''
    for loader in repeat(data_loader):
        yield from loader


def check_dir(path):
    if not os.path.exists(path):
        print("Checkpoint Directory does not exist! Making directory {}".format(path))
        os.makedirs(path)


def get_lr(optimizer):
    """

    Args:
        optimizer ():

    Returns:

    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MetricTracker:
    def __init__(self, *keys, writer=None, mode='/'):
        """

        Args:
            *keys ():
            writer ():
            mode ():
        """
        self.writer = writer
        self.mode = mode + '/'
        self.keys = keys
        # print(self.keys)
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, writer_step=1):
        if self.writer is not None:
            self.writer.add_scalar(self.mode + key, value, writer_step)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_all_metrics(self, values_dict, n=1, writer_step=1):
        for key in values_dict:
            self.update(key, values_dict[key], n, writer_step)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def calc_all_metrics(self):
        """
        Calculates string with all the metrics
        Returns:

        """
        s = ''
        d = dict(self._data.average)
        for key in dict(self._data.average):
            s += f'{key} {d[key]:7.4f}\t'

        return s

    def wer(self):
        wer_keys = ['S', 'D', 'I', 'C']
        if wer_keys in self.keys:
            return (self._data.total['S'] + self._data.total['D'] + self._data.total['I']) / (
                    self._data.total['S'] + self._data.total['D'] + self._data.total['C'])
        else:
            return self._data.average['wer']


def one_hot(target, num_classes):
    """

    Args:
        target ():
        num_classes ():

    Returns:

    """
    labels = target.reshape(-1, 1).cpu()
    one_hot_target = ((labels == torch.arange(num_classes).reshape(1, num_classes)).float())
    return one_hot_target


def smooth_one_hot(true_labels, classes, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    Args:
        true_labels ():
        classes ():
        smoothing ():

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist


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


def txt_logger(txtname, log):
    """

    Args:
        txtname ():
        log ():
    """
    with open(txtname, 'a') as f:
        for item in log:
            f.write(item)
            f.write(',')

        f.write('\n')


def write_csv(data, name):
    """

    Args:
        data ():
        name ():
    """
    with open(name, 'w') as fout:
        for item in data:
            # print(item)
            fout.write(item)
            fout.write('\n')
