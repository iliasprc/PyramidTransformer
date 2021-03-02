import warnings

import torch.hub
import torch.nn as nn
import torch.nn.functional as F

from models.vmz.layers import Conv3DDepthwise, BasicStem_Pool, IPConv3DDepthwise, BasicStem, Bottleneck
from models.vmz.utils import _generic_resnet, _pyramid_resnet


class IP_CSN_152(nn.Module):
    def __init__(self, pretraining="ig_ft_kinetics_32frms", n_classes=400, use_pool1=True, progress=False, **kwargs):
        super(IP_CSN_152, self).__init__()
        avail_pretrainings = [
            "ig65m_32frms",
            "ig_ft_kinetics_32frms",
            "sports1m_32frms",
            "sports1m_ft_kinetics_32frms",
        ]

        if pretraining in avail_pretrainings:
            arch = "ip_csn_152_" + pretraining
            pretrained = True
        else:
            warnings.warn(
                f"Unrecognized pretraining dataset, continuing with randomly initialized network."
                " Available pretrainings: {avail_pretrainings}",
                UserWarning,
            )
            arch = "ip_csn_152"
            pretrained = False

        self.model = _generic_resnet(
            arch,
            pretrained,
            progress,
            block=Bottleneck,
            conv_makers=[IPConv3DDepthwise] * 4,
            layers=[3, 8, 36, 3],
            stem=BasicStem_Pool if use_pool1 else BasicStem,
            **kwargs,
        )
        if pretraining == "ig65m_kinetics":
            print("load ig65m_kinetics")
            self.model.load_state_dict(torch.load(
                "/home/papastrat/PycharmProjects/VMZ/checkpoint/ipCSN_152_ft_kinetics_from_ig65m_f133090949.pth"),
                strict=True)
        self.model.fc = torch.nn.Linear(2048, n_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch):
        x, y = train_batch
        y_hat = self.forward(x)
        # print(y_hat.shape)
        loss = F.cross_entropy(y_hat, y.squeeze(-1))
        return y_hat, loss

    def validation_step(self, train_batch):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return y_hat, loss

    def freeze_param(self):
        count = 0
        for param in self.model.parameters():
            count += 1

            param.requires_grad = False
        for param in self.model.fc.parameters():
            count += 1

            param.requires_grad = True


class PyramidIR_CSN_152(nn.Module):
    def __init__(self, pretraining="ig_ft_kinetics_32frms", n_classes=400, use_pool1=True, progress=False, **kwargs):
        super(PyramidIR_CSN_152, self).__init__()
        avail_pretrainings = [
            "ig65m_32frms",
            "ig_ft_kinetics_32frms",
            "sports1m_32frms",
            "sports1m_ft_kinetics_32frms",
        ]

        if pretraining in avail_pretrainings:
            arch = "ir_csn_152_" + pretraining
            pretrained = True
        else:
            warnings.warn(
                f"Unrecognized pretraining dataset, continuing with randomly initialized network."
                " Available pretrainings: {avail_pretrainings}",
                UserWarning,
            )
            arch = "ir_csn_152"
            pretrained = False

        self.model = _pyramid_resnet(
            arch,
            pretrained,
            progress,
            block=Bottleneck,
            conv_makers=[Conv3DDepthwise] * 4,
            layers=[3, 8, 36, 3],
            stem=BasicStem_Pool if use_pool1 else BasicStem,
            **kwargs,
        )

        self.model.fc = torch.nn.Linear(2048, n_classes)
        # self.freeze_param()

    def forward(self, x, y=None):
        if y == None:
            return self.model(x)
        y_hat, auxloss = self.model(x, y)
        return y_hat, auxloss

    def training_step(self, train_batch):
        x, y = train_batch
        y_hat, auxloss = self.forward(x, y)
        # print(y_hat.shape)
        loss = F.cross_entropy(y_hat, y.squeeze(-1))
        return y_hat, loss  # +auxloss

    def validation_step(self, train_batch):
        x, y = train_batch
        y_hat, auxloss = self.forward(x, y)
        # print(y_hat.shape)
        loss = F.cross_entropy(y_hat, y.squeeze(-1))
        return y_hat, loss  # +auxloss

    # def freeze_param5(self):
    #     count = 0
    #     for param in self.model.parameters():
    #         count += 1
    #
    #         param.requires_grad = False
    #     for param in self.model.fc.parameters():
    #         count += 1
    #
    #         param.requires_grad = True

    def freeze_param(self):
        count = 0
        for name, param in self.model.named_parameters():
            count += 1
            # print(name)'layer3' in name  or 'layer4' in name or
            if 'stem' in name or 'tpn' in name or 'aux' in name or 'fc' in name or 'classifier' in name:
                # print(name)
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.model.fc.parameters():
            count += 1

            param.requires_grad = True
