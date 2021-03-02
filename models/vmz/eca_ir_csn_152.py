import warnings

import torch
import torch.nn as nn

from models.attentions.attentions import ECA_3D
from models.vmz.layers import Bottleneck, IPConv3DDepthwise, BasicStem, BasicStem_Pool

from models.vmz.utils import model_urls

class ECAVideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(ECAVideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        # self.eca1 = ECA_3D(1,k_size=3)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        # self.eca2 = ECA_3D(1, k_size=5)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.eca3 = ECA_3D(1, k_size=7)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)
        self.eca4 = ECA_3D(1, k_size=9)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.eca5 = ECA_3D(1, k_size=11)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)  # self.eca1(self.layer1(x))
        x = self.layer2(x)  # self.eca2(self.layer2(x))
        x = self.eca3(self.layer3(x))
        x = self.eca4(self.layer4(x))

        x = self.eca5(self.avgpool(x))
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def eca_ir_csn_152(pretraining="ig65m_32frms", pretrained=False, progress=False, num_classes=226, **kwargs):
    avail_pretrainings = [
        "ig65m_32frms",
        "ig_ft_kinetics_32frms",
        "sports1m_32frms",
        "sports1m_ft_kinetics_32frms",
    ]
    if pretrained:
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
    use_pool1 = True
    model = ECAVideoResNet(block=Bottleneck, conv_makers=[IPConv3DDepthwise] * 4, layers=[3, 8, 36, 3],
                           stem=BasicStem_Pool if use_pool1 else BasicStem, num_classes=num_classes, **kwargs)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict, strict=False)

    return model
