import numpy as np
import torch

from base import BaseTrainer


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

        self.logger = logger

    def predict(self):
        """
        Inference
        Args:
            epoch ():

        Returns:

        """
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                data = data.to(self.device)

                logits = self.model(data, None)

                maxes, prediction = torch.max(logits, 1)  # get the index of the max log-probability

                predictions.append(f"{target[0]},{prediction.cpu().numpy()[0]}")
        self.logger.info('Inference done')
        return predictions

