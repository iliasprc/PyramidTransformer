import os

import numpy as np
import torch

from base import BaseTrainer
from models.model_utils import save_checkpoint_slr
from utils.util import MetricTracker
from utils.util import write_csv


class TesterRGBD(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, config, model, data_loader, writer, checkpoint_dir, logger,
                 valid_data_loader=None, test_data_loader=None, metric_ftns=None):
        super(TesterRGBD, self).__init__(config, data_loader, writer, checkpoint_dir, logger,
                                         valid_data_loader=valid_data_loader,
                                         test_data_loader=test_data_loader, metric_ftns=metric_ftns)
        if (self.config.cuda):
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")
        self.start_epoch = 1
        self.train_data_loader = data_loader

        self.len_epoch = self.config.dataloader.train.batch_size * len(self.train_data_loader)
        self.epochs = self.config.epochs
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.log_step = self.config.log_interval
        self.model = model

        self.checkpoint_dir = checkpoint_dir

        self.writer = writer
        self.metric_ftns = ['loss', 'acc']
        self.train_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='train')
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

        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                rgb_tensor, depth_tensor, y = data
                rgb_tensor = rgb_tensor.to(self.device)
                depth_tensor = depth_tensor.to(self.device)
                y = y.to(self.device)

                output, loss, target = self.model((rgb_tensor, depth_tensor, y))

                loss = loss.mean()
                writer_step = (epoch - 1) * len(loader) + batch_idx

                prediction = torch.max(output, 1)
                # acc = np.sum(prediction[1].cpu().numpy() == target.cpu().numpy()) / target.size(0)
                # self.valid_metrics.update_all_metrics(
                #     {'loss': loss.item(), 'acc': acc}, writer_step=writer_step)
                self.valid_metrics.update(key='loss', value=loss.item(), n=1, writer_step=writer_step)
                self.valid_metrics.update(key='acc', value=np.sum(prediction[1].cpu().numpy() == target.cpu().numpy()),
                                          n=target.size(0), writer_step=writer_step)
                # {'loss': loss.item(), 'acc': acc}, writer_step=writer_step)

        self._progress(batch_idx, epoch, metrics=self.valid_metrics, mode=mode, print_summary=True)

        val_loss = self.valid_metrics.avg('loss')

        return val_loss

    def predict(self,loader):
        """
         Inference


         Returns: predictions (list)

         """
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                rgb_tensor, depth_tensor, y = data
                rgb_tensor = rgb_tensor.to(self.device)
                depth_tensor = depth_tensor.to(self.device)
                target = y

                output, _ = self.model((rgb_tensor, depth_tensor, y), False)

                maxes, prediction = torch.max(output, 1)  # get the index of the max log-probability

                for i in range(data.shape[0]):

                    predictions.append(f"{target[i]},{prediction.cpu().numpy()[i]}")

        pred_name = os.path.join(self.checkpoint_dir, f'predictions_.csv')
        write_csv(predictions, pred_name)
        return predictions

    def checkpointer(self, epoch, metric):

        is_best = metric < self.mnt_best
        if (is_best):
            self.mnt_best = metric

            self.logger.info(f"Best val loss {self.mnt_best} so far ")

        save_checkpoint_slr(self.model, self.optimizer, epoch, self.valid_metrics.avg('loss'),
                            self.checkpoint_dir, '_model',
                            save_seperate_layers=True, is_best=is_best)

    def _progress(self, batch_idx, epoch, metrics, mode='', print_summary=False):
        metrics_string = metrics.calc_all_metrics()
        if (batch_idx % self.log_step == 0):

            if metrics_string == None:
                self.logger.warning(f" No metrics")
            else:
                self.logger.info(
                    f"{mode} Epoch: [{epoch:2d}/{self.epochs:2d}]\t Video [{batch_idx:5d}/{self.len_epoch:5d}]\t {metrics_string}")
        elif print_summary:
            self.logger.info(
                f'{mode} summary  Epoch: [{epoch}/{self.epochs}]\t {metrics_string}')
