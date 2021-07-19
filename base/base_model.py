import pytorch_lightning as pl
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseModel(pl.LightningModule):
    """
    Base class for all models
    """

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config['model']


# def forward(self, *inputs):
#     """
#
#     Args:
#         *inputs (torch.Tensor): input data to model
#     """
#     raise NotImplementedError
#
#
# def training_step(self, train_batch, batch_idx=None):
#     """
#     training function of vodel
#     Args:
#         train_batch (tuple): (data, target)
#         batch_idx (int):
#     """
#     pass
#
#
# def validation_step(self, train_batch, batch_idx=None):
#     """
#     validation function of vodel
#     Args:
#         train_batch (tuple): (data, target)
#         batch_idx (int):
#     """
#     pass


    def optimizer(self):
        opt =self.config['optimizer']['type']
        lr = self.config['optimizer']['lr']
        if (opt == 'Adam'):
            print(" use optimizer Adam lr ", lr)
            optimizer = optim.Adam(self.parameters(), lr=float(self.config['optimizer']['lr']),
                                   weight_decay=float(self.config['optimizer']['weight_decay']))
        elif (opt == 'SGD'):
            print(" use optimizer SGD lr ", lr)
            optimizer = optim.SGD(self.parameters(), lr=float(self.config['optimizer']['lr']), momentum=0.9, nesterov=False,
                                  weight_decay=float(self.config['optimizer']['weight_decay']))
        elif (opt == 'RMSprop'):
            print(" use RMS  lr", lr)
            optimizer = optim.RMSprop(self.parameters(), lr=float(self.config['optimizer']['lr']))

        if self.config['scheduler']['type'] == 'ReduceLRonPlateau':
            scheduler = ReduceLROnPlateau(optimizer, factor=self.config['scheduler']['scheduler_factor'],
                                          patience=self.config['scheduler']['scheduler_patience'],
                                          min_lr=self.config['scheduler']['scheduler_min_lr'],
                                          verbose=self.config['scheduler']['scheduler_verbose'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler,'monitor': 'metric'}


    def loss(self, *inputs):
        """
        Loss calculation
        """
        raise NotImplementedError

    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)
