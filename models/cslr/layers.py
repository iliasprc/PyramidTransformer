import numpy as np

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}

import torch
import torch.nn as nn
from einops import repeat, rearrange
import torch.nn.functional as F


def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)


class PositionalEncoding1D(nn.Module):

    def __init__(self, dim, dropout=0.1, max_tokens=64):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_tokens, dim)
        position = torch.arange(0, max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.Tensor([10000.0])) / dim))
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = pe.cuda()

    def forward(self, x):
        batch, seq_tokens, _ = x.size()
        x = x + expand_to_batch(self.pe[:, :seq_tokens, :], desired_size=batch)
        return self.dropout(x)


class ECA_3D(nn.Module):
    """Constructs a channel attention module.
    Args:
        channel (int): Number of channels of the input feature map
        k_size (int): Adaptive selection of kernel size
    """

    def __init__(self, channel=1, k_size=3, name='eca'):
        super(ECA_3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.name = name

    def forward(self, x):
        # x: input features with shape [b, c,d, h, w]
        # b, c,d, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # print(f'avg pool {y.shape}')
        # Two different branches of ECA module
        # print(y.squeeze(-1).squeeze(-1).transpose(-1, -2).shape)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        # print(y.shape)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):
        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):
        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """

    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class BasicStem_Pool(nn.Sequential):
    def __init__(self):
        super(BasicStem_Pool, self).__init__(
            nn.Conv3d(
                3,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )


class R2Plus1dStem_Pool(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self):
        super(R2Plus1dStem_Pool, self).__init__(
            nn.Conv3d(
                3,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45,
                64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )


class Conv3DDepthwise(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        assert in_planes == out_planes
        super(Conv3DDepthwise, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            groups=in_planes,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class IPConv3DDepthwise(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        assert in_planes == out_planes
        super(IPConv3DDepthwise, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_planes),
            # nn.ReLU(inplace=True),
            Conv3DDepthwise(out_planes, out_planes, None, stride),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class TemporalModulation(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 downsample_scale=8,
                 groups=32,
                 ):
        super(TemporalModulation, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False, groups=groups)
        self.pool = nn.MaxPool3d((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class SpatialModulation(nn.Module):
    def __init__(self,
                 planes,
                 downsample_scale=8,
                 k=1,
                 s=1,
                 d=1
                 ):
        super(SpatialModulation, self).__init__()
        k1 = int(np.log2(planes))
        if k1 % 2 == 0:
            k1 -= 5
        else:
            k1 -= 4
        self.eca1 = ECA_3D(k_size=k1)
        # self.conv = nn.Conv3d(planes, planes//2, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
        #                       ,dilation=(1 ,1,1), bias=False ,groups=planes//2)
        #
        # self.bn = nn.BatchNorm3d(planes//2)
        # self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool3d((downsample_scale, 7, 7))
        encoder_layer = nn.TransformerEncoderLayer(d_model=planes, dim_feedforward=planes, nhead=8, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_cls_token = False
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, planes // 2))
        #     self.to_out = nn.Sequential(
        #         nn.LayerNorm(planes//2),
        #         nn.Linear(planes//2, 226)
        #     )
        # else:
        #     self.classifier = nn.Linear(planes//2,226)

    def forward(self, x):
        b = x.shape[0]
        x = self.eca1(x)
        x = self.pool(x)  # self.relu(self.bn(self.conv(x))))
        # print(x.shape)
        x = rearrange(x, 'b c t h w -> (t h w) b c')

        if self.use_cls_token:
            cls_token = repeat(self.cls_token, 'n d -> n b d', b=b)
            # print(cls_token.shape,x.shape)
            x = torch.cat((cls_token, x), dim=0)
            # print(x.shape)

            x = self.transformer_encoder(x)
            # print(x.shape,x[0].shape)
            cls_token = x[0]
            return cls_token  # self.to_out(cls_token)
        else:
            # print(x.shape)
            x = self.transformer_encoder(x)
            x = torch.mean(x, dim=1, keepdim=True)

            # x = self.classifier(x)

            return x


class ConvModule(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            k=1,
            s=1,
            d=1,
            pool_size=1,
            padding=0,
            downsample_scale=1,
            bias=False,
            groups=1,
    ):
        super(ConvModule, self).__init__()
        k1 = int(np.log2(planes))
        print(k1)
        if k1 % 2 == 0:
            k1 -= 1
        self.eca1 = ECA_3D(k_size=k1)
        self.conv1 = nn.Conv3d(planes, planes // 2, kernel_size=(k, k, k), stride=(s, s, s), padding=(0, 0, 0)
                               , dilation=(1, d, d), bias=False, groups=planes // 2)
        self.bn = nn.BatchNorm3d(planes // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropt = nn.Dropout(0.2)
        self.eca2 = ECA_3D(k_size=k1)
        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))

    def forward(self, x):
        #  print(x.shape)
        out = self.eca2(self.dropt(self.relu(self.bn(self.conv1(self.eca1(x))))))
        # print(out.shape)
        out = (self.pool(out)).squeeze(-1).squeeze(-1).squeeze(-1)

        return out


class AttConvModule(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            k=1,
            s=1,
            d=1,
            pool_size=1,
            padding=0,
            downsample_scale=1,
            bias=False,
            groups=1,
    ):
        super(AttConvModule, self).__init__()

        self.conv1 = nn.Conv3d(planes, planes, kernel_size=(1, k, k), stride=(1, s, s), padding=(0, 0, 0),
                               dilation=(1, d, d), bias=True, groups=planes)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool3d((downsample_scale, 7, 7))

        self.g = nn.Conv3d(planes, planes // 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                           dilation=(1, 1, 1), bias=False, groups=planes // 2)

        self.f = nn.Conv3d(planes, planes // 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                           dilation=(1, 1, 1), bias=False, groups=planes // 2)

        self.h = nn.Conv3d(planes, planes // 2, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                           dilation=(1, 1, 1), bias=False, groups=planes // 2)

    def forward(self, x):
        out = (self.pool(self.relu(self.bn(self.conv1(x)))))
        g = self.g(out)
        f = self.f(out)
        h = self.h(out)
        B, C, D, H, W = f.shape
        g = g.view(B, -1, C)
        f = f.view(B, C, -1)
        soft = torch.softmax(g @ f, dim=-1)
        h = h.view(B, C, -1)
        # print(soft.shape,h.shape)
        out = h @ soft
        out = out.view(B, -1, D, H, W)
        # print(out.shape)
        # print(out.shape)

        return out


class AuxHead(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            loss_weight=0.25
    ):
        super(AuxHead, self).__init__()
        # self.convs = \
        #     ConvModule(inplanes, inplanes * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.loss_weight = loss_weight
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(inplanes, planes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

    def forward(self, x, target=None):
        if target is None:
            return None

        # x = self.convs(x)
        x = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        # print(x.shape)
        x = self.fc(x)

        loss_aux = self.loss_weight * F.cross_entropy(x, target.squeeze(-1))
        return x, loss_aux

#
# class Conv2Plus1D(nn.Sequential):
#     def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
#         midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
#                 in_planes * 3 * 3 + 3 * out_planes
#         )
#         super(Conv2Plus1D, self).__init__(
#             nn.Conv3d(
#                 in_planes,
#                 midplanes,
#                 kernel_size=(1, 3, 3),
#                 stride=(1, stride, stride),
#                 padding=(0, padding, padding),
#                 bias=False,
#             ),
#             nn.BatchNorm3d(midplanes),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(
#                 midplanes,
#                 out_planes,
#                 kernel_size=(3, 1, 1),
#                 stride=(stride, 1, 1),
#                 padding=(padding, 0, 0),
#                 bias=False,
#             ),
#         )
#
#     @staticmethod
#     def get_downsample_stride(stride):
#         return (stride, stride, stride)
