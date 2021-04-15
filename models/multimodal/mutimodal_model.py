import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from omegaconf import OmegaConf

from base.base_model import BaseModel
from models.vmz.layers import BasicStem_Pool, SpatialModulation, Conv3DDepthwise, ECA_3D,PositionalEncoding1D
from models.vmz.resnet import Bottleneck

model_urls = {
    "r2plus1d_34_8_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth",
    # noqa: E501
    "r2plus1d_34_32_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth",
    # noqa: E501
    "r2plus1d_34_8_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth",
    # noqa: E501
    "r2plus1d_34_32_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth",
    # noqa: E501
    "r2plus1d_152_ig65m_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/r2plus1d_152_ig65m_from_scratch_f106380637.pth",
    "r2plus1d_152_ig_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/r2plus1d_152_ft_kinetics_from_ig65m_f107107466.pth",
    "r2plus1d_152_sports1m_32frms": "",
    "r2plus1d_152_sports1m_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/r2plus1d_152_ft_kinetics_from_sports1m_f128957437.pth",
    "ir_csn_152_ig65m_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/irCSN_152_ig65m_from_scratch_f125286141.pth",
    "ir_csn_152_ig_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/irCSN_152_ft_kinetics_from_ig65m_f126851907.pth",
    "ir_csn_152_sports1m_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/irCSN_152_Sports1M_from_scratch_f99918785.pth",
    "ir_csn_152_sports1m_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/irCSN_152_ft_kinetics_from_Sports1M_f101599884.pth",
    "ip_csn_152_ig65m_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/ipCSN_152_ig65m_from_scratch_f130601052.pth",
    "ip_csn_152_ig_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/ipCSN_152_ft_kinetics_from_ig65m_f133090949.pth",
    "ip_csn_152_sports1m_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/ipCSN_152_Sports1M_from_scratch_f111018543.pth",
    "ip_csn_152_sports1m_ft_kinetics_32frms": "https://github.com/bjuncek/VMZ/releases/download/test_models/ipCSN_152_ft_kinetics_from_Sports1M_f111279053.pth",
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class TransformerResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False, late_fusion=True):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(TransformerResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)

        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.tpn3 = SpatialModulation(256 * block.expansion, downsample_scale=16, k=3, s=1, d=2)

        self.tpn4 = SpatialModulation(512 * block.expansion, downsample_scale=8, k=1, s=1, d=1)
        #
        # self.eca = ECA_3D(k_size=9)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x, target=None):
        with torch.no_grad():
            x = self.stem(x)

            x = self.layer1(x)

            x = self.layer2(x)

        x = self.layer3(x)
        tpn3 = self.tpn3(x)  # + tpn2

        x = self.layer4(x)
        tpn4 = self.tpn4(x)  # + tpn3

        x_new = torch.cat((tpn3, tpn4), dim=-1)
        # x_new = self.eca(x_new.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1).squeeze(-1)

        return x_new

    def training_step(self, train_batch):
        x, y = train_batch
        y_hat = self.forward(x)
        # print(y_hat.shape)
        loss = F.cross_entropy(y_hat, y.squeeze(-1))
        return y_hat, loss

    def validation_step(self, train_batch):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y.squeeze(-1))
        return y_hat, loss

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def ir_csn_152_transformer(pretraining="ig_ft_kinetics_32frms", pretrained=False, progress=False, num_classes=226,
                           **kwargs):
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
    model = TransformerResNet(block=Bottleneck, conv_makers=[Conv3DDepthwise] * 4, layers=[3, 8, 36, 3],
                              stem=BasicStem_Pool, num_classes=num_classes, **kwargs)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    #model.fc = nn.Linear(2048, 400)
    state_dict = torch.hub.load_state_dict_from_url(
        model_urls[arch], progress=progress
    )
    model.load_state_dict(state_dict, strict=False)
    #model.fc = nn.Linear(3072, 226)

    return model




class SkeletonTR(nn.Module):
    def __init__(self, mode='isolated', planes=512, N_classes=226):
        super(SkeletonTR, self).__init__()
        self.embed = nn.Linear(27 * 3, planes)
        # self.embed = nn.Sequential(nn.Linear(133*3,planes),nn.ReLU(),nn.Dropout(0.2),nn.Linear(planes,planes))

        self.pe = PositionalEncoding1D(planes, max_tokens=300)
        encoder_layer = nn.TransformerEncoderLayer(d_model=planes, dim_feedforward=planes, nhead=8, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=32, dim_feedforward=planes, nhead=8, dropout=0.2)
        # self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.use_cls_token = True
        self.mode = mode
        if mode == 'isolated':
            self.loss = nn.CrossEntropyLoss()

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, planes))
            self.to_out = nn.Sequential(
                nn.LayerNorm(planes),
                nn.Linear(planes, N_classes)
            )
        else:
            self.classifier = nn.Linear(planes, N_classes)

    def forward(self, x, y=None):
        b = x.shape[0]
        #  #print(x.shape)
        x = rearrange(x, 'b t k a -> b t (k a)')
        # x = self.eca1(x)
        # x = self.pool(x)  # self.relu(self.bn(self.conv(x))))
        # #print(x.shape)
        x = self.pe(self.embed(x))
        ##print(x.shape)
        # x = rearrange(x, 'b c t h w -> (t h w) b c')

        if self.use_cls_token:
            cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
            ##print(cls_token.shape,x.shape)
            x = torch.cat((cls_token, x), dim=1)
            ##print(x.shape)
            x = rearrange(x, 'b t d -> t b d')

            x = self.transformer_encoder(x)
            ##print(x.shape,x[0].shape)
            # cls_token = x
            # if self.mode == 'isolated':
            #print(x.shape)
            cls_token = x[0]
            return cls_token
            y_hat = self.to_out(cls_token)
        else:
            # #print(x.shape)
            x = rearrange(x, 'b t d -> t b d')

            x = self.transformer_encoder(x)
            # x = self.transformer_encoder(rearrange(x, 't b d -> d b t'))
            x = torch.mean(x, dim=0)
            y_hat = self.classifier(x)

        if y != None:
            loss = self.loss(y_hat, y)
            return y_hat, loss
        return y_hat


class RGBD_Transformer(BaseModel):
    def __init__(self, config, N_classes):
        """

        Args:
            args (): argparse arguments
            N_classes (int): number of classes
        """
        super().__init__()
        config = OmegaConf.load(os.path.join(config.cwd, 'models/multimodal/model.yml'))['model']
        self.rgb_encoder = ir_csn_152_transformer(pretraining="ig_ft_kinetics_32frms", pretrained=True, progress=False,
                                                  num_classes=N_classes)
        self.depth_encoder = ir_csn_152_transformer(pretraining="ig_ft_kinetics_32frms", pretrained=True,
                                                    progress=False,
                                                    num_classes=N_classes)

        self.classifier = nn.Linear(6144, N_classes)
        self.device0 = torch.device('cuda:0')
        self.device1 = torch.device('cuda:1')

    def forward(self, train_batch, return_loss=True):
        rgb_tensor, depth_tensor, y = train_batch

        features_rgb = self.rgb_encoder(rgb_tensor)
        features_depth = self.depth_encoder(depth_tensor)

        concatenated = torch.cat((features_rgb, features_depth), dim=1)
        concatenated = self.eca(concatenated.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1).squeeze(
            -1)
        logits = self.classifier(concatenated)
        if return_loss:
            loss = F.cross_entropy(logits, y.squeeze(-1))  # .mean()

            return logits, loss, y
        return logits, y

    def training_step(self, train_batch, batch_idx=None):
        rgb_tensor, depth_tensor, y = train_batch

        features_rgb = self.rgb_encoder(rgb_tensor.to(self.device0))
        features_depth = self.depth_encoder(depth_tensor.to(self.device1))

        concatenated = torch.cat((features_rgb, features_depth.to(self.device0)), dim=1).to(self.device0)

        logits = self.classifier(concatenated)
        loss = F.cross_entropy(logits, y.squeeze(-1).to(self.device0))
        return logits, loss, y

    def validation_step(self, train_batch, batch_idx=None):
        rgb_tensor, depth_tensor, y = train_batch
        # (len(train_batch))
        features_rgb = self.rgb_encoder(rgb_tensor.to(self.device0))
        features_depth = self.depth_encoder(depth_tensor.to(self.device1))
        # y = y.to(self.device0)
        concatenated = torch.cat((features_rgb, features_depth.to(self.device0)), dim=1).to(self.device0)

        logits = self.classifier(concatenated)
        loss = F.cross_entropy(logits, y.squeeze(-1).to(self.device0))
        return logits, loss, y


class RGB_SK_Transformer(BaseModel):
    def __init__(self, config, N_classes):
        """

        Args:
            args (): argparse arguments
            N_classes (int): number of classes
        """
        super().__init__()
        config = OmegaConf.load(os.path.join(config.cwd, 'models/multimodal/model.yml'))['model']
        self.rgb_encoder = ir_csn_152_transformer(pretraining="ig_ft_kinetics_32frms", pretrained=True, progress=False,
                                                  num_classes=N_classes)
        self.sk_encoder = SkeletonTR(planes=512,N_classes=226)
        #self.eca = ECA_3D(k_size=11)
        self.classifier = nn.Linear(3072+512, N_classes)
        self.device0 = torch.device('cuda:0')
        self.device1 = torch.device('cuda:1')

    def forward(self, train_batch, return_loss=True):
        rgb_tensor, sk_tensor, y = train_batch

        features_rgb = self.rgb_encoder(rgb_tensor)
        features_sk = self.sk_encoder(sk_tensor)
        #print(features_rgb.shape,features_sk.shape)
        concatenated = torch.cat((features_rgb, features_sk), dim=1)

        logits = self.classifier(concatenated)
        if return_loss:
            loss = F.cross_entropy(logits, y.squeeze(-1))  # .mean()

            return logits, loss, y
        return logits, y

    def training_step(self, train_batch, batch_idx=None):
        rgb_tensor, depth_tensor, y = train_batch

        features_rgb = self.rgb_encoder(rgb_tensor.to(self.device0))
        features_depth = self.depth_encoder(depth_tensor.to(self.device1))

        concatenated = torch.cat((features_rgb, features_depth.to(self.device0)), dim=1).to(self.device0)

        logits = self.classifier(concatenated)
        loss = F.cross_entropy(logits, y.squeeze(-1).to(self.device0))
        return logits, loss, y

    def validation_step(self, train_batch, batch_idx=None):
        rgb_tensor, depth_tensor, y = train_batch
        # (len(train_batch))
        features_rgb = self.rgb_encoder(rgb_tensor.to(self.device0))
        features_depth = self.depth_encoder(depth_tensor.to(self.device1))
        # y = y.to(self.device0)
        concatenated = torch.cat((features_rgb, features_depth.to(self.device0)), dim=1).to(self.device0)

        logits = self.classifier(concatenated)
        loss = F.cross_entropy(logits, y.squeeze(-1).to(self.device0))
        return logits, loss, y
