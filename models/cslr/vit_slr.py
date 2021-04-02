
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from einops import rearrange
# model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=311)
# print(model)
#
#
# m = timm.create_model('vit_base_patch16_224', pretrained=True)
# o = m(torch.randn(2, 3, 224, 224))
# print(f'Original shape: {o.shape}')
# m.reset_classifier(0, '')
# o = m(torch.randn(2, 3, 224, 224))
# print(f'Unpooled shape: {o.shape}')

class VitSLR(nn.Module):
    def __init__(self):

        super(VitSLR, self).__init__()
        self.cnn = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.cnn.reset_classifier(0, '')
        self.temp_channels = 768
        self.tc_kernel_size = 5
        self.tc_stride = 1
        self.tc_pool_size = 2
        self.tc_padding = 0
        # print(self.tc_padding)
        self.temporal = torch.nn.Sequential(
        nn.Conv1d(768, self.temp_channels, kernel_size=self.tc_kernel_size, stride=self.tc_stride,
        padding=0),
        nn.ReLU(),
        nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
        nn.Conv1d(self.temp_channels, self.temp_channels, kernel_size=self.tc_kernel_size, stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size))
        cslr = True
        if cslr:
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, dim_feedforward=2*768, nhead=8, dropout=0.2)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(768,311)

    def forward(self, x,y=None):
        batch_size, C, T, H, W = x.size()
        c_in = x.view(batch_size * T, C, H, W)
        c_outputs = self.cnn(c_in)
        c_out = c_outputs.contiguous().view(batch_size, -1, 768)
        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)
        # print('temp out ', temp.size())
        # last linear input must be timesteps x batch_size x dim_feats

        fc_input = temp.squeeze(-1)

        y_hat = self.fc(fc_input)
        if y!=None:
            loss = F.cross_entropy(y_hat, y.squeeze(-1))
            return y_hat, loss
        return y_hat
    def cslr_forward(self, x,y=None):
        batch_size, C, T, H, W = x.size()
        c_in = x.view(batch_size * T, C, H, W)
        c_outputs = self.cnn(c_in)
        c_out = c_outputs.contiguous().view(batch_size, -1, 768)
        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)
        # print('temp out ', temp.size())
        # last linear input must be timesteps x batch_size x dim_feats

        fc_input = temp.squeeze(-1)

