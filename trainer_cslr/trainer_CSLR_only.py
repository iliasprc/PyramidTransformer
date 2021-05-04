import json
import os
from base import BaseTrainer
import numpy as np
import torch

from utils.util import write_csv, check_dir
from models.model_utils import save_checkpoint_slr
from utils.util import MetricTracker
from utils.metrics import word_error_rate_generic

class Trainer_CSLR_method(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, config, model, optimizer, data_loader, writer, id2w, checkpoint_dir, logger,
                 valid_data_loader=None, test_data_loader=None, lr_schedulers=None, metric_ftns=None):
        super(Trainer_CSLR_method, self).__init__(config, data_loader, writer, checkpoint_dir, logger,
                                      valid_data_loader=valid_data_loader,
                                      test_data_loader=test_data_loader, metric_ftns=metric_ftns)
        if (self.config.cuda):
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")
        self.start_epoch = 1
        self.train_data_loader = data_loader

        self.len_epoch = len(self.train_data_loader)
        self.epochs = self.config.epochs
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_schedulers = lr_schedulers
        self.log_step = self.config.log_interval
        self.model = model
        self.gradient_accumulation = config.gradient_accumulation
        self.optimizer = optimizer

        self.mnt_best = np.inf

        self.metric_ftns = metric_ftns
        self.checkpoint_dir = checkpoint_dir
        self.id2w = id2w

        self.writer = writer
        self.metric_ftns = ['ctc_loss', 'wer']
        self.train_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='train')
        self.metric_ftns = ['ctc_loss', 'wer']
        self.valid_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='dev')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()

        self.train_metrics.reset()
        n_critic = 1
        for batch_idx, (data, target) in enumerate(self.train_data_loader):

            data = data.to(self.device)

            target = target.long().to(self.device)

            output,loss_ctc = self.model(data,target)



            (loss_ctc / n_critic).backward()
            if (batch_idx % n_critic == 0):
                self.optimizer.step()  # Now we can do an optimizer step
                self.optimizer.zero_grad()  # Reset gradients tensors

            temp_wer, s, C, S, I, D = word_error_rate_generic(output, target, self.id2w)

            writer_step = (epoch - 1) * self.len_epoch + batch_idx
            self.train_metrics.update_all_metrics(
                {
                    'ctc_loss': loss_ctc.item(), 'wer': 100.0 * temp_wer,
                }, writer_step=writer_step)

            self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train')

        self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train', print_summary=True)

    def _train_epoch_with_context(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()

        real_label = 1
        fake_label = 0
        self.train_metrics.reset()
        n_critic = 1
        for batch_idx, (data, target) in enumerate(self.train_data_loader):

            data = data.to(self.device)
            if (len(target) > 1):
                target, context = target

            target = target.long().to(self.device)

            output = self.model(data)

            #### GENERATOR #######

            loss_ctc = self.criterion(output, target).to(self.device)

            (loss_ctc / n_critic).backward()
            if (batch_idx % n_critic == 0):
                self.optimizer.step()  # Now we can do an optimizer step
                self.optimizer.zero_grad()  # Reset gradients tensors

            temp_wer, s, C, S, I, D = word_error_rate_generic(output, target, self.id2w)

            # self.logger.info(true_score,fake_score)
            writer_step = (epoch - 1) * self.len_epoch + batch_idx
            self.train_metrics.update_all_metrics(
                {
                    'ctc_loss': loss_ctc.item(), 'wer': 100.0 * temp_wer,
                }, writer_step=writer_step)

            self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train')

        self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train', print_summary=True)

        self.logger.info('self.hyperparam ', self.model_Discriminator.hyperparam.item())

    def _valid_epoch(self, epoch, mode, loader):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_sentences = []
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data = data.to(self.device)

                target = target.long().to(self.device)

                output,loss_ctc = self.model(data,target)  # , lm_inputs)



                temp_wer, s, C, S, I, D = word_error_rate_generic(output, target, self.id2w)
                # self.logger.info(temp_wer)
                self.valid_sentences.append(s)

                writer_step = (epoch - 1) * len(loader) + batch_idx

                self.valid_metrics.update_all_metrics(
                    {'ctc_loss': loss_ctc.item(), 'wer': 100.0 * temp_wer}, writer_step=writer_step)
        self._progress(batch_idx, epoch, metrics=self.valid_metrics, mode=mode, print_summary=True)

        check_dir(self.checkpoint_dir)
        # pred_name = self.checkpoint_dir.joinpath(mode + '_epoch_{:d}_WER_{:.2f}_.csv'.format(epoch,
        #                                                                                      self.valid_metrics.avg(
        #                                                                                          'wer')))
        # write_csv(self.valid_sentences, pred_name)
        werr = self.valid_metrics.avg('wer')
        # if (self.args.dataset == 'phoenix2014' or self.args.dataset == 'phoenix2014_feats'):
        #     werr = evaluate_phoenix(pred_name, mode)
        #     self.logger.info('PHOENIX EVALUATION {}'.format(werr))
        #     os.rename(pred_name, self.checkpoint_dir.joinpath( mode + '_epoch_{:d}_WER_{:.2f}_.csv'.format(epoch,
        #                                                                                                  werr)))

        return werr

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self._train_epoch(epoch)

            #self.logger.info("!" * 10, "   VALIDATION   ", "!" * 10)
            werr = self._valid_epoch(epoch, 'dev', self.valid_data_loader)
            #werr = 0
            check_dir(self.checkpoint_dir)
            self.checkpointer(epoch, werr)
            # TODO
            self.lr_schedulers.step(float(werr))
            # if self.do_test:
            #     self.logger.info("!" * 10, "   TESTING   ", "!" * 10)
            #     self._valid_epoch(epoch, 'test', self.test_data_loader)

    def test(self):
        self.logger.info(f"   VALIDATION   ")
        werr = self._valid_epoch(0, 'dev', self.valid_data_loader)
        self.logger.info("!" * 10, "   TESTING   ", "!" * 10)
        self._valid_epoch(0, 'test', self.test_data_loader)
        check_dir(self.checkpoint_dir)
        pred_name = self.checkpoint_dir.joinpath('test_predictions_epoch_{:d}_WER_{:.2f}_.csv'.format(0,
                                                                                                           self.valid_metrics.avg(
                                                                                                               'wer')))
        write_csv(self.valid_sentences, pred_name)

    def checkpointer(self, epoch, werr):

        is_best = werr < self.mnt_best
        if (is_best):
            self.mnt_best = werr

            self.logger.info("Best wer {} so far ".format(self.mnt_best))

        save_checkpoint_slr(self.model, self.optimizer, epoch, self.valid_metrics.avg('wer'),
                            self.checkpoint_dir, 'generator',
                            save_seperate_layers=True, is_best=is_best)

    def _progress(self, batch_idx, epoch, metrics, mode='', print_summary=False):
        metrics_string = metrics.calc_all_metrics()
        if (batch_idx % self.log_step == 0):

            if metrics_string == None:
                self.logger.info(" No metrics")
            else:
                self.logger.info(
                    '{} Epoch: [{:2d}/{:2d}]\t Video [{:5d}/{:5d}]\t {}'.format(
                        mode, epoch, self.epochs, batch_idx, self.len_epoch, metrics_string))
        elif print_summary:
            self.logger.info(
                '{} summary  Epoch: [{}/{}]\t {}'.format(
                    mode, epoch, self.epochs, metrics_string))

    def _extract_features(self, save_folder='/home/papastrat/Desktop/ilias/datasets/csl_features/'):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        import os
        import numpy as np
        self.model.eval()
        self.valid_sentences = []
        self.valid_metrics.reset()
        dataloaders = [self.train_data_loader, self.valid_data_loader, self.test_data_loader]
        modes = ['train', 'val', 'test']

        for i, loader in enumerate(dataloaders):
            self.valid_metrics.reset()
            self.logger.info('Extract fetures mode {} samples {} '.format(modes[i], len(loader)))
            with torch.no_grad():
                for batch_idx, (data, target, path) in enumerate(loader):
                    data = data.to(self.device)

                    target = target.long().to(self.device)

                    output, feats = self.model.extract_feats(data)  # , lm_inputs)
                    feats = feats.cpu().numpy()
                    # self.logger.info(feats.shape)
                    # self.logger.info(path[0].split('/')[-1])
                    sent = path[0].split('/')[-1]
                    scenario = path[0].split('/')[-2]
                    path = scenario + '-' + sent + '.npy'
                    save_name = os.path.join(save_folder, scenario, sent)
                    # self.logger.info(scenario,sent)
                    # self.logger.info(save_folder+modes[i]+'/'+path[0]+'.npy')
                    if not os.path.exists(save_name):
                        os.makedirs(save_name)
                    np.save(save_name + '/feats.npy', feats)
                    # loss_ctc = self.criterion(output, target).to(self.device)

                    temp_wer, s, C, S, I, D = word_error_rate_generic(output, target, self.id2w)
                    # self.logger.info('wer : {}'.format(temp_wer))
                    self.valid_sentences.append(s)

                    self.valid_metrics.update_all_metrics(
                        {'ctc_loss': 0.0, 'wer': 100.0 * temp_wer}, writer_step=batch_idx)
            self._progress(batch_idx, 0, metrics=self.valid_metrics, mode=modes[i], print_summary=True)

        return self.valid_metrics.result()
