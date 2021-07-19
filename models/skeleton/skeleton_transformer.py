import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from utils.ctcl import CTCL

from base.base_model import BaseModel
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


class SkeletonTR(BaseModel):
    def __init__(self, mode='isolated', planes=512, N_classes=226):
        super(SkeletonTR, self).__init__()
        self.embed = nn.Linear(65 * 3, planes)
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
        else:
            self.loss = CTCL()
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
            print(x.shape, x[0].shape)
            cls_token = x
            if self.mode == 'isolated':
                cls_token = x[0]

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


# class CSLRSkeletonTR(nn.Module):
#     def __init__(self, planes = 512,N_classes = 226):
#         super(CSLRSkeletonTR, self).__init__()
#         self.embed = nn.Linear(65*3,planes)
#         #self.embed = nn.Sequential(nn.Linear(133*3,planes),nn.ReLU(),nn.Dropout(0.2),nn.Linear(planes,planes))
#         self.temp_channels = planes
#
#         self.tc_kernel_size = 5
#         self.tc_pool_size = 2
#         self.padding = 1
#         self.temporal = torch.nn.Sequential(
#             nn.InstanceNorm1d(planes),
#             nn.Conv1d(planes, planes, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
#             nn.ReLU(),
#             nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
#             nn.Conv1d(planes, planes, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
#             nn.ReLU(),
#             nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size))
#         self.pe = PositionalEncoding1D(planes,max_tokens=300)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=planes, dim_feedforward=planes, nhead=8, dropout=0.1)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
#        # encoder_layer = nn.TransformerEncoderLayer(d_model=32, dim_feedforward=planes, nhead=8, dropout=0.2)
#        # self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=2)
#
#         self.use_cls_token = True
#
#         self.loss = CTC_Loss()
#         if self.use_cls_token:
#             self.cls_token = nn.Parameter(torch.randn(1, planes))
#             self.to_out = nn.Sequential(
#                 nn.LayerNorm(planes),
#                 nn.Linear(planes, N_classes)
#             )
#         else:
#             self.classifier = nn.Linear(planes,N_classes)
#
#     def forward(self, x,y=None):
#         b = x.shape[0]
#       #  #print(x.shape)
#         x= rearrange(x,'b t k a -> b t (k a)')
#         #x = self.eca1(x)
#         #x = self.pool(x)  # self.relu(self.bn(self.conv(x))))
#         # #print(x.shape)
#         x = self.embed(x)
#
#         x = self.temporal(rearrange(x,'b t c -> b c t'))
#         x = self.pe(rearrange(x,'b c t -> b t c'))
#         ##print(x.shape)
#         #x = rearrange(x, 'b c t h w -> (t h w) b c')
#
#         if self.use_cls_token:
#             cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
#             ##print(cls_token.shape,x.shape)
#             x = torch.cat((cls_token, x), dim=1)
#             ##print(x.shape)
#             x= rearrange(x,'b t d -> t b d')
#
#             x = self.transformer_encoder(x)
#             ##print(x.shape,x[0].shape)
#             cls_token = x
#
#
#             y_hat = self.to_out(cls_token)
#         else:
#             # #print(x.shape)
#             x = rearrange(x, 'b t d -> t b d')
#
#             x = self.transformer_encoder(x)
#             #x = self.transformer_encoder(rearrange(x, 't b d -> d b t'))
#             x = torch.mean(x, dim=0)
#             y_hat = self.classifier(x)
#
#         if y!=None:
#             loss = self.loss(y_hat, y)
#             return y_hat, loss
#         return y_hat
#


class SK_TCL(nn.Module):
    def __init__(self, channels=128, N_classes=311):
        super(SK_TCL, self).__init__()

        self.tc_kernel_size = 5
        self.tc_pool_size = 2
        self.padding = 0
        self.window_size = 16
        self.stride = 8
        planes = channels
        self.embed = nn.Linear(33 * 3, planes)
        self.tcl = torch.nn.Sequential(

            nn.Conv1d(planes, planes, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
            nn.Conv1d(planes, planes, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size)

        )
        self.pe = PositionalEncoding1D(planes, max_tokens=300)
        encoder_layer = nn.TransformerEncoderLayer(d_model=planes, dim_feedforward=2 * planes, nhead=8, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(planes, N_classes)
        self.loss = CTCL()

    def forward(self, x, y=None):
        x1 = x.unfold(1, self.window_size, self.stride).squeeze(0)

        # print(x1.shape)
        x1 = self.embed(rearrange(x1, 'w k a t -> w t (k a)'))
        # print(x1.shape)
        x1 = rearrange(x1, 'w t c -> w c t')
        x = self.tcl(x1)

        x = rearrange(x, 'w c t -> t w c')
        x = self.pe(x)
        # print(x.shape)
        x = rearrange(x, 'b t c -> t b c')
        x = self.transformer_encoder(x)
        y_hat = self.fc(x)
        if y != None:
            loss = self.loss(y_hat, y)
            return y_hat, loss
        return y_hat

    def training_step(self, train_batch):
        x, y = train_batch
        y_hat = self.forward(x)
        # print(y_hat.shape)
        loss = self.loss(y_hat, y)
        return y_hat, loss

    def validation_step(self, train_batch):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return y_hat, loss


class CSLRSkeletonTR(nn.Module):
    def __init__(self, planes=256, N_classes=226):
        super(CSLRSkeletonTR, self).__init__()
        planes = 128
        self.embed = nn.Linear(33 * 3, planes)
        # self.embed = nn.Sequential(nn.Linear(65*3,planes),nn.LayerNorm(planes))
        self.temp_channels = planes

        # self.tc_kernel_size = 3
        # self.tc_pool_size = 2
        # self.padding = 1
        # self.temporal = torch.nn.Sequential(
        #
        #     nn.Conv1d(65*3, planes, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
        #     nn.ReLU(),
        # #     nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
        #     nn.Conv1d(planes, planes, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
        #     nn.ReLU(),
        #     nn.InstanceNorm1d(planes))
        #     nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size))
        self.pe = PositionalEncoding1D(planes, max_tokens=300)
        encoder_layer = nn.TransformerEncoderLayer(d_model=planes, dim_feedforward=2 * planes, nhead=8, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=32, dim_feedforward=planes, nhead=8, dropout=0.2)
        # self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.use_cls_token = True
        self.window_size = 16
        self.stride = 8
        self.loss = CTCL()
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, planes))
            self.to_out = nn.Sequential(
                nn.LayerNorm(planes),
                nn.Linear(planes, N_classes)
            )
        else:
            self.classifier = nn.Linear(planes, N_classes)

    def forward(self, x, y=None):

        # print(x.shape)
        x1 = x.unfold(1, self.window_size, self.stride).squeeze(0)

        # print(x1.shape)
        x1 = rearrange(x1, 'w k a t -> w t (k a)')
        x = self.embed(x1)
        # x=  rearrange(x1,'b t c -> b t c')

        #
        b = x.shape[0]
        x = rearrange(x, 'b t c -> b c t')
        x = self.pe(rearrange(x, 'b c t -> b t c'))
        ##print(x.shape)
        # x = rearrange(x, 'b c t h w -> (t h w) b c')

        if self.use_cls_token:
            cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
            # print(cls_token.shape,x.shape)
            x = torch.cat((cls_token, x), dim=1)
            ##print(x.shape)
            x = rearrange(x, 'b t d -> t b d')

            x = self.transformer_encoder(x)
            # print(x.shape,x[0].shape)
            cls_token = x[0]

            y_hat = self.to_out(cls_token).unsqueeze(1)
            # print('yaht',y_hat.shape)
        else:
            # #print(x.shape)
            x = rearrange(x, 'b t d -> t b d')

            x = self.transformer_encoder(x)
            # x = self.transformer_encoder(rearrange(x, 't b d -> d b t'))
            x = torch.mean(x, dim=0, keepdim=True)
            print(x.shape)
            print(b)
            y_hat = self.classifier(x)

        if y != None:
            loss = self.loss(y_hat, y)
            return y_hat, loss
        return y_hat

    def training_step(self, train_batch):
        x, y = train_batch
        y_hat = self.forward(x)
        # print(y_hat.shape)
        loss = self.loss(y_hat, y)
        return y_hat, loss

    def validation_step(self, train_batch):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return y_hat, loss

# m = SkeletonTR()
# from torchsummary import summary
# summary(m,(64,133*3),device='cpu')
