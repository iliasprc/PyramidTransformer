import math

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from base.base_model import BaseModel

from base.base_model import BaseModel
from models.cslr.i3d import InceptionModule, MaxPool3dSamePadding, Unit3D
from models.skeleton.skeleton_transformer import PositionalEncoding1D
from utils.ctcl import CTCL




class SK_TCL(BaseModel):
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
        self.loss = CTCL()

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



class CSLRSkeletonTR(BaseModel):
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



