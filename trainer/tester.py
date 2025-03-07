import json
import os

import numpy as np
import torch
import torch.nn as nn

from base import BaseTrainer
from models.model_utils import save_checkpoint_slr
from utils.metrics import accuracy
from utils.util import MetricTracker
from utils.util import write_csv, check_dir


class Tester(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, config, model, data_loader, writer, checkpoint_dir, logger,
                 valid_data_loader=None, test_data_loader=None, metric_ftns=None):
        super(Tester, self).__init__(config, data_loader, writer, checkpoint_dir, logger,
                                     valid_data_loader=valid_data_loader,
                                     test_data_loader=test_data_loader, metric_ftns=metric_ftns)

        if (self.config.cuda):
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")
        self.start_epoch = 1

        self.epochs = self.config.epochs
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None

        self.log_step = self.config.log_interval
        self.model = model

        self.mnt_best = np.inf

        self.checkpoint_dir = checkpoint_dir
        self.gradient_accumulation = config.gradient_accumulation
        self.metric_ftns = ['loss', 'acc']
        self.valid_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='validation')
        self.logger = logger
    def _valid_epoch(self, epoch, mode, loader):
        """

        Args:
            epoch (int): current epoch
            mode (string): 'validation' or 'test'
            loader (dataloader):

        Returns: validation loss

        """
        self.model.eval()
        self.valid_sentences = []
        self.valid_metrics.reset()
        predictions = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data = data.to(self.device)

                #target = target.long().to(self.device)

                output, loss = self.model(data)#, target)
                loss = loss.mean()
                writer_step = (epoch - 1) * len(loader) + batch_idx

                prediction = torch.max(output, 1)
                for i in range(data.shape[0]):

                    predictions.append(f"{target[i]},{prediction[1].cpu().numpy()[i]}")
                    #print(f"{target[i]},{prediction.cpu().numpy()[i]}")
                acc = np.sum(prediction[1].cpu().numpy() == target.cpu().numpy()) / target.size(0)

                self.valid_metrics.update(key='loss',value=loss.item(),n=1,writer_step=writer_step)
                self.valid_metrics.update(key='acc', value=np.sum(prediction[1].cpu().numpy() == target.cpu().numpy()), n=target.size(0), writer_step=writer_step)

        self._progress(batch_idx, epoch, metrics=self.valid_metrics, mode=mode, print_summary=True)

        time = str(batch_idx)
        pred_name = os.path.join(self.checkpoint_dir, f'predictions_{time}.csv')
        write_csv(predictions, pred_name)
        val_loss = self.valid_metrics.avg('loss')


        return val_loss


    def predict(self,loader):
        """
        Inference
        Args:
            epoch ():

        Returns:

        """
        self.model.eval()

        predictions = []
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, path,target) in enumerate(loader):
                data = data.to(self.device)

                logits = self.model(data, None)
             #   print(data.shape)
                logits = logits.cpu()
                maxes, prediction = torch.max(logits, 1)  # get the index of the max log-probability
                print(prediction.shape,target.shape)
                print(prediction.cpu().numpy() == target.squeeze(-1).cpu().numpy())
                s = np.sum(prediction.cpu().numpy() == target.cpu().numpy())
                if s>4:
                    print(prediction.cpu().numpy() == target.cpu().numpy())
                    print(s)
                    print('GAMW TH PANAGIA ')
               # print(target,'\n')
                l = list(zip(target.cpu().numpy()[0],prediction.cpu().numpy()))
                #print(l)
                #print(np.sum(prediction.cpu().numpy() == target.cpu().numpy()))
                correct +=float(np.sum(prediction.cpu().numpy() == target.cpu().numpy()))
                total+= float(target.size(0))
                self.logger.info(f"{batch_idx}  {correct/total}  {correct} {total} ")
                for i in range(data.shape[0]):

                    predictions.append(f"{path[i]},{prediction.cpu().numpy()[i]}")
                    #print(f"{path[i]},{prediction.cpu().numpy()[i]}")
        self.logger.info('Inference done')
        time = str(batch_idx)
        pred_name = os.path.join(self.checkpoint_dir, f'predictions_{time}.csv')
        write_csv(predictions, pred_name)
        return predictions

    def _progress(self, batch_idx, epoch, metrics, mode='', print_summary=False):
        metrics_string = metrics.calc_all_metrics()
        if ((batch_idx * self.config.dataloader.train.batch_size) % self.log_step == 0):

            if metrics_string == None:
                self.logger.warning(f" No metrics")
            else:
                self.logger.info(
                    f"{mode} Epoch: [{epoch:2d}/{self.epochs:2d}]\t Video [{batch_idx * self.config.dataloader.train.batch_size:5d}/{self.len_epoch:5d}]\t {metrics_string}")
        elif print_summary:
            self.logger.info(
                f'{mode} summary  Epoch: [{epoch}/{self.epochs}]\t {metrics_string}')

