import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from einops import repeat, rearrange
from utils.ctcl import CTCL
from omegaconf import OmegaConf
from utils.loss import LabelSmoothingCrossEntropy
from base.base_model import BaseModel
from models.vmz.layers import BasicStem_Pool, SpatialModulation3D, Conv3DDepthwise, PositionalEncoding1D
from models.vmz.layers1d import Conv1DDepthwise, BasicStem_Pool1D, Bottleneck1D
from models.vmz.resnet import Bottleneck
from models.gcn.model.decouple_gcn_attn import STGCN
from models.transformers.transformer import TransformerEncoder

from einops import rearrange
class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, temporal_resolution=16, mode='isolated', spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          temporal_resolution : output every temporal_resolution frames, changes last avg pool kernel
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self.mode = mode

        print("Mode i3d ", self.mode)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self.temp_resolution = temporal_resolution
        last_duration = int(math.ceil(16/ 8))
        #if (self.mode == 'unfold' or self.mode == 'features'):
        self.window_size = 16
        self.stride = 8
        print("Train with sliding window size {} stride {}".format(self.window_size, self.stride))

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[last_duration, 7, 7],
                                     stride=(1, 1, 1))
        #self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)

        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        self.loss =  nn.CrossEntropyLoss()

        self.build()
        self.use_tr = False
        planes = 1024
        max_tokens = 8+1
        #self.pe = PositionalEncoding1D(dim = planes,max_tokens=max_tokens)


        if self.use_tr:
            self.transformer_encoder = TransformerEncoder(dim=planes, blocks=2, heads=8, dim_head=64,
                                                          dim_linear_block=planes, dropout=0.1)
            self.use_cls_token = True
            self.cls_token = nn.Parameter(torch.randn(1, planes ))
            self.classifier = nn.Sequential(
                nn.LayerNorm(planes),
                nn.Linear(planes, self._num_classes)
            )
        else:
            self.rnn = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=2,
                dropout=0.5,
                bidirectional=True,batch_first=True)
            self.fc1 = nn.Sequential(nn.Linear(1024,1024),nn.LeakyReLU(0.2))
            self.fc2 = nn.Sequential(nn.Linear(1024,512),nn.LeakyReLU(0.2),nn.Linear(512,256),nn.LeakyReLU(0.2),nn.Dropout(0.1))
            self.fc = nn.Linear(256,self._num_classes)
    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x , y=None):

        if (self.mode == 'isolated'):

            for end_point in self.VALID_ENDPOINTS:
                if end_point in self.end_points:
                    # print(x.size())
                    x = self._modules[end_point](x)  # use _modules to work with dataparallel
            if False:
                y_hat = self.logits(self.dropout(self.avg_pool(x))).squeeze(-1).squeeze(-1).squeeze(-1)
                #print(x.shape)
                if y != None:
                    loss_ctc = self.loss(y_hat, y.squeeze(-1))
                    return y_hat, loss_ctc
                return y_hat
            if self.use_tr:
                x = self.avg_pool(self.dropout(x)).squeeze(-1).squeeze(-1)
                #print(x.shape)
                b = x.shape[0]
                cls_token = repeat(self.cls_token, 'n c -> b n c', b=b)
                #print(x.shape,cls_token.shape)
                x = rearrange(x, 'b c t -> b t c')
                x = torch.cat((cls_token, x), dim=1)
                x = self.transformer_encoder(x)
                #print(x.shape)
                # x = rearrange(x, 'b t d -> t b d')

                y_hat = self.classifier(x[:, 0, :])
            else:
                x = self.dropout(self.avg_pool(x)).squeeze(-1).squeeze(-1)
                x = rearrange(x, 'b c t -> b t c')
                r_out, (h_n, h_c) = self.rnn(x)
                x = r_out.mean(dim=1)
                x1 = self.fc2(x+ self.fc1(x))

                y_hat = self.fc(x1)
                #print(y_hat.shape,y.shape)
            if y != None:
                loss_ctc = self.loss(y_hat, y.squeeze(-1))
                return y_hat, loss_ctc
            return y_hat
            return y_hat
        elif (self.mode == 'continuous'):
            #print(x.shape)
            self.window_size = random.randint(16,18)
            self.stride = random.randint(self.window_size//2 +1,self.window_size-1)
            x = x.unfold(2, self.window_size, self.stride).squeeze(0)
            #print(x.shape)
            x = rearrange(x, 'c n h w t -> n c t h w')
            #print(x.shape)
            with torch.no_grad():
                for end_point in self.VALID_ENDPOINTS:
                    if end_point in self.end_points:
                        # print(x.size())
                        x = self._modules[end_point](x)  # use _modules to work with dataparallel

            x = self.dropout(self.avg_pool(x))
            #print(f"feats {x.shape}")
            return x
            # logits = self.logits(x)
            # print(x.size())
            final_time, dim, _, _, _ = x.size()


            y_hat = self.logits(x)
            y_hat = rearrange(y_hat,'t classes d h w -> t (d h w) classes')
            #print(y_hat.size())
            if y != None:
                loss_ctc = self.loss(y_hat, y)
                return y_hat, loss_ctc
            return y_hat


        elif (self.mode == 'features'):

            batch_size, T, _, _, _ = x.size()

            x = x.unfold(1, self.window_size, self.stride).squeeze(0).permute(0, 1, 4, 2, 3)

            for end_point in self.VALID_ENDPOINTS:
                if end_point in self.end_points:
                    # print(x.size())
                    x = self._modules[end_point](x)

            return x

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.dropout(self.avg_pool(x))
    def freeze_param(self):
        count = 0
        for name,param in self.named_parameters():

            count += 1

            if 'logits' in  name :
                print(name)
                param.requires_grad = True
            else:
                param.requires_grad=False

class InceptionI3d_Sentence(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, temporal_resolution=16, mode='isolated', spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          temporal_resolution : output every temporal_resolution frames, changes last avg pool kernel
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d_Sentence, self).__init__()
        self.mode = mode

        print("Mode i3d ", self.mode)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self.temp_resolution = temporal_resolution
        last_duration = int(math.ceil(16/ 8))
        #if (self.mode == 'unfold' or self.mode == 'features'):
        self.window_size = 16
        self.stride = 8
        print("Train with sliding window size {} stride {}".format(self.window_size, self.stride))

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[last_duration, 7, 7],
                                     stride=(1, 1, 1))
        #self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)

        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        self.loss =  nn.CrossEntropyLoss()
        self.ctc_loss = CTCL()

        self.build()
        self.use_tr = False
        planes = 1024
        max_tokens = 8+1
        #self.pe = PositionalEncoding1D(dim = planes,max_tokens=max_tokens)


        if self.use_tr:
            self.transformer_encoder = TransformerEncoder(dim=planes, blocks=2, heads=8, dim_head=64,
                                                          dim_linear_block=planes, dropout=0.1)
            self.use_cls_token = True
            self.cls_token = nn.Parameter(torch.randn(1, planes ))
            self.classifier = nn.Sequential(
                nn.LayerNorm(planes),
                nn.Linear(planes, self._num_classes)
            )
        else:
            self.rnn = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=2,
                dropout=0.5,
                bidirectional=True,batch_first=True)
            # self.fc1 = nn.Sequential(nn.Linear(1024,1024),nn.LeakyReLU(0.2))
            # self.fc2 = nn.Sequential(nn.Linear(1024,512),nn.LeakyReLU(0.2),nn.Linear(512,256),nn.LeakyReLU(0.2),nn.Dropout(0.1))
            # self.fc = nn.Linear(256,self._num_classes)
            # self.fc1 = nn.Sequential(nn.Linear(1024,1024),nn.LeakyReLU(0.2))
            # self.fc2 = nn.Sequential(nn.Linear(1024,512),nn.LeakyReLU(0.2),nn.Linear(512,256),nn.LeakyReLU(0.2),nn.Dropout(0.1))
            self.fc_gloss = nn.Linear(1024,311)
            self.fc_sentence = nn.Linear(1024, 228)
    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x , y=None):
        y , y_g = y

        with torch.no_grad():
            for end_point in self.VALID_ENDPOINTS:
                if end_point in self.end_points:
                    # print(x.size())
                    x = self._modules[end_point](x)  # use _modules to work with dataparallel

        if self.use_tr:
            x = self.avg_pool(self.dropout(x)).squeeze(-1).squeeze(-1)
            #print(x.shape)
            b = x.shape[0]
            cls_token = repeat(self.cls_token, 'n c -> b n c', b=b)
            #print(x.shape,cls_token.shape)
            x = rearrange(x, 'b c t -> b t c')
            x = torch.cat((cls_token, x), dim=1)
            x = self.transformer_encoder(x)
            #print(x.shape)
            # x = rearrange(x, 'b t d -> t b d')

            y_hat = self.classifier(x[:, 0, :])
        else:
            #with torch.no_grad():
            x = self.dropout(self.avg_pool(x)).squeeze(-1).squeeze(-1)
            x = rearrange(x, 'b c t -> b t c')
            r_out, (h_n, h_c) = self.rnn(x)
            y_gloss = rearrange(self.fc_gloss(r_out), 'b t c -> t b c')
            x = r_out.mean(dim=1)
            y_hat = self.fc_sentence(x)

           # y_hat = self.fc(x1)
        #print(y_g.shape,y_gloss.shape)

        if y != None:
            loss_ = self.loss(y_hat, y.squeeze(-1))
            loss_ctc = self.ctc_loss(y_gloss,y_g)
            return y_hat,y_gloss, loss_+loss_ctc
        return y_hat
        return y_hat

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.dropout(self.avg_pool(x))
    def freeze_param(self):
        count = 0
        for name,param in self.named_parameters():

            count += 1


