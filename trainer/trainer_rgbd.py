import json
import os

import numpy as np
import torch

from base import BaseTrainer
from models.model_utils import save_checkpoint_slr
from utils.util import MetricTracker
from utils.util import write_csv, check_dir


class TrainerRGBD(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, config, model, optimizer, data_loader, writer, checkpoint_dir, logger,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, metric_ftns=None):
        super(TrainerRGBD, self).__init__(config, data_loader, writer, checkpoint_dir, logger,
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
        self.lr_scheduler = lr_scheduler
        self.log_step = self.config.log_interval
        self.model = model

        self.optimizer = optimizer

        self.mnt_best = np.inf

        self.checkpoint_dir = checkpoint_dir
        self.gradient_accumulation = config.gradient_accumulation
        self.writer = writer
        self.metric_ftns = ['loss', 'acc']
        self.train_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='train')
        self.metric_ftns = ['loss', 'acc']
        self.valid_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='validation')
        self.logger = logger
        # if torch.cuda.device_count()>1:
        #     self.logger.info(f" Data parallel {torch.cuda.device_count()} GPUS")
        #     model = nn.DataParallel(model)
        #     model.cuda()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Args:
            epoch (int): current training epoch.
        """

        self.model.train()

        self.train_metrics.reset()
        gradient_accumulation = self.gradient_accumulation
        for batch_idx, data in enumerate(self.train_data_loader):
            rgb_tensor, depth_tensor, y = data
            rgb_tensor = rgb_tensor.to(self.device)
            depth_tensor = depth_tensor.to(self.device)
            y = y.to(self.device)

            output, loss, target = self.model((rgb_tensor, depth_tensor, y))

            loss = loss.mean()


            (loss / gradient_accumulation).backward()
            if (batch_idx % gradient_accumulation == 0):
                self.optimizer.step()  # Now we can do an optimizer step
                self.optimizer.zero_grad()  # Reset gradients tensors

            prediction = torch.max(output, 1)
            acc = np.sum(prediction[1].cpu().numpy() == target.cpu().numpy()) / target.size(0)
            writer_step = (epoch - 1) * self.len_epoch + batch_idx
            self.train_metrics.update_all_metrics(
                {
                    'loss': loss.item(), 'acc': acc,
                }, writer_step=writer_step)

            self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train')

        self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train', print_summary=True)

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
                acc = np.sum(prediction[1].cpu().numpy() == target.cpu().numpy()) / target.size(0)
                self.valid_metrics.update_all_metrics(
                    {'loss': loss.item(), 'acc': acc}, writer_step=writer_step)
        self._progress(batch_idx, epoch, metrics=self.valid_metrics, mode=mode, print_summary=True)

        val_loss = self.valid_metrics.avg('loss')

        return val_loss

    def train(self):
        """
        Train the model
        """
        for epoch in range(self.start_epoch, self.epochs):
            self._train_epoch(epoch)

            self.logger.info(f"{'!' * 10}    VALIDATION   , {'!' * 10}")
            validation_loss = self._valid_epoch(epoch, 'validation', self.valid_data_loader)
            check_dir(self.checkpoint_dir)
            self.checkpointer(epoch, validation_loss)
            self.lr_scheduler.step(validation_loss)
            if self.do_test:
                self.logger.info("!" * 10, "   TESTING   ", "!" * 10)
                self.predict(epoch)

    def predict(self, epoch):
        """
         Inference
         Args:
             epoch ():

         Returns:

         """
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_data_loader):
                rgb_tensor, depth_tensor, y = data
                rgb_tensor = rgb_tensor.to(self.device)
                depth_tensor = depth_tensor.to(self.device)
                target = y

                output, _ = self.model((rgb_tensor, depth_tensor, y), False)

                maxes, prediction = torch.max(output, 1)  # get the index of the max log-probability

                predictions.append(f"{target[0]},{prediction.cpu().numpy()[0]}")

        pred_name = os.path.join(self.checkpoint_dir, f'validation_predictions_epoch_{epoch:d}_.csv')
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
