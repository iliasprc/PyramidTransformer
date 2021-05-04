import math

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops import repeat

from base.base_model import BaseModel
from models.cslr.i3d import InceptionModule, MaxPool3dSamePadding, Unit3D
from models.skeleton.skeleton_transformer import PositionalEncoding1D
from utils.ctc_loss import CTC_Loss




class SK_TCL(nn.Module):
    def __init__(self,channels=256,N_classes=311):
        super(SK_TCL, self).__init__()

        self.tc_kernel_size = 5
        self.tc_pool_size = 2
        self.padding = 0
        self.window_size = 16
        self.stride = 8
        planes = channels
        self.embed = nn.Linear(33*3,planes)
        self.tcl = torch.nn.Sequential(

            nn.Conv1d(planes, planes, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
            nn.Conv1d(planes, planes, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.InstanceNorm1d(planes),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size))
        self.pe = PositionalEncoding1D(planes,max_tokens=300)
        encoder_layer = nn.TransformerEncoderLayer(d_model=planes, dim_feedforward=2*planes, nhead=8, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(planes,N_classes)
        self.loss = CTC_Loss()

    def forward(self,x,y=None):
        x1 = x.unfold(1, self.window_size, self.stride).squeeze(0)



        #print(x1.shape)
        x1 = self.embed(rearrange(x1,'w k a t -> w t (k a)'))
        #print(x1.shape)
        x1 = rearrange(x1,'w t c -> w c t')
        x = self.tcl(x1)


        x = rearrange(x,'w c t -> t w c')
        x = self.pe(x)
        #print(x.shape)
        x = rearrange(x,'b t c -> t b c')
        x = self.transformer_encoder(x)
        y_hat = self.fc(x)
        if y != None:
            loss = self.loss(y_hat, y)
            return x, loss
        return x,torch.tensor(0)


class CSLRSkeletonTR(nn.Module):
    def __init__(self, planes=512, N_classes=226):
        super(CSLRSkeletonTR, self).__init__()
        self.embed = nn.Linear(65 * 3, planes)
        #self.embed = nn.Sequential(nn.Linear(65 * 3, planes), nn.Dropout(0.2), nn.LayerNorm(planes))
        self.temp_channels = planes

        self.tc_kernel_size = 5
        self.tc_pool_size = 2
        self.padding = 1
        channels = planes

        self.pe = PositionalEncoding1D(channels, max_tokens=300)
        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, dim_feedforward=2*channels, nhead=8, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=32, dim_feedforward=planes, nhead=8, dropout=0.2)
        # self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=2)

        #self.use_cls_token = True
        self.window_size = 16
        self.stride = 8

        #if self.use_cls_token:
        self.cls_token = nn.Parameter(torch.randn(1, channels))
        # self.to_out = nn.Sequential(
        #     nn.LayerNorm(channels),
        #     nn.Linear(channels, N_classes)
        # )


    def forward(self, x, y=None):

        # print(x.shape)
        x1 = x.unfold(1, self.window_size, self.stride).squeeze(0)
        # print(x1.shape)
        x1 = rearrange(x1, 'w k a t -> w t (k a)')
        x = x1  # rearrange(x,'b t k a -> b t (k a)')
        # print(x1.shape)
        # x = self.eca1(x)
        # x = self.pool(x)  # self.relu(self.bn(self.conv(x))))
        # #print(x.shape)
        x = self.embed(x)
        b = x.shape[0]
        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        #print('SK CLS x ',cls_token.shape,x.shape)
        x = torch.cat((cls_token, x), dim=1)
        #print(x.shape)
        x = rearrange(x, 'b t c -> b c t')
        x = self.pe(rearrange(x, 'b c t -> b t c'))
        #print('SK after position ',x.shape)

        x = rearrange(x, 'b t d -> t b d')

        x = self.transformer_encoder(x)
        #print('SK features x x[0] ',x.shape,x[0].shape)
        return x[0].unsqueeze(1)



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
        last_duration = int(math.ceil(16 / 8))
        # if (self.mode == 'unfold' or self.mode == 'features'):
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
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        # self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
        #                      kernel_shape=[1, 1, 1],
        #                      padding=0,
        #                      activation_fn=None,
        #                      use_batch_norm=False,
        #                      use_bias=True,
        #                      name='logits')
        self.pe = PositionalEncoding1D(1024, dropout=0.1, max_tokens=300)
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, dim_feedforward=2048, nhead=8, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(1024,num_classes)
        self.loss= CTC_Loss()


        self.build()

        self.freeze_param()

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

    def forward(self, x,y=None):


        x = x.unfold(2, self.window_size, self.stride).squeeze(0)
        # print(x.shape)
        x = rearrange(x, 'c n h w t -> n c t h w')
        #print('after window ',x.shape)
        # with torch.no_grad():
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                # print(x.size())
                x = self._modules[end_point](x)  # use _modules to work with dataparallel

        x = self.dropout(self.avg_pool(x))

        # logits = self.logits(x)
        # print(x.size())
        final_time, dim, _, _, _ = x.size()

        x= rearrange(x, 'n c d h w -> n (d h w) c')


        x = rearrange(x, 't b c -> b t c')
        #print('RGB before pe ', x.shape)
        x = self.pe(x)

        x = rearrange(x, 'b t c -> t b c')

        #print('RGB befor tr,',x.shape)
        #

        x = self.transformer_encoder(x)
        y_hat = self.fc(x)

        if y!=None:
            loss = self.loss(y_hat, y)
            return x, loss
        return x,torch.tensor(0)




    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)

    def freeze_param(self):
        count = 0
        for name, param in self.named_parameters():
            count += 1
            print(name)
            if 'Conv3d_' in name or 'Mixed_3' in name:
                param.requires_grad = False

            else:
                param.requires_grad = True



class RGBSK_model(nn.Module):
    def __init__(self, N_classes):
        super().__init__()

        self.cnn = InceptionI3d(num_classes=N_classes, temporal_resolution=25, mode='continuous',
                                in_channels=3)
        self.sk = SK_TCL(N_classes=N_classes)
        channels = 1024 + 256
        self.pe = PositionalEncoding1D(channels, max_tokens=300)
        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, dim_feedforward=2*channels, nhead=8, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=32, dim_feedforward=planes, nhead=8, dropout=0.2)
        # self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=2)

        #self.use_cls_token = True
        self.window_size = 16
        self.stride = 8

        #if self.use_cls_token:
        #self.cls_token = nn.Parameter(torch.randn(1, channels))
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, N_classes)
        )

        self.loss = CTC_Loss()
    def forward(self, vid,skeleton,y=None):

        with torch.no_grad():
            rgb_feats,loss_rgb = self.cnn(vid,y)
        sk_feats,loss_sk = self.sk(skeleton,y)
        #print(f"rgb {rgb_feats.shape} sk {sk_feats.shape}")
        x = torch.cat((rgb_feats,sk_feats),dim=-1)
        #print('cocnat x ',x.shape)
        b = x.shape[0]
        x = rearrange(x, 't b c -> b t c')
        x = self.pe(x)
        ##print(x.shape)
        # x = rearrange(x, 'b c t h w -> (t h w) b c')

        #if self.use_cls_token:
        # cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        # # print(cls_token.shape,x.shape)
        # x = torch.cat((cls_token, x), dim=1)
        # ##print(x.shape)
        x = rearrange(x, 'b t d -> t b d')

        x = self.transformer_encoder(x)
        # print(x.shape,x[0].shape)


        y_hat = self.classifier(x)
        if y!=None:
            loss = self.loss(y_hat, y)
            return y_hat, 0.5*loss+0.3*loss_rgb+0.2*loss_sk
        return y_hat