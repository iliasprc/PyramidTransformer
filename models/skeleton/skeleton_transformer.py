

import torch
import torch.nn as nn
from einops import repeat,rearrange
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
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = pe.cuda()

    def forward(self, x):
        batch, seq_tokens, _ = x.size()
        x = x + expand_to_batch( self.pe[:, :seq_tokens, :], desired_size=batch)
        return self.dropout(x)


class SkeletonTR(nn.Module):
    def __init__(self,planes = 512,N_classes = 226):
        super(SkeletonTR, self).__init__()
        self.embed = nn.Linear(27*3,planes)
        #self.embed = nn.Sequential(nn.Linear(133*3,planes),nn.ReLU(),nn.Dropout(0.2),nn.Linear(planes,planes))

        self.pe = PositionalEncoding1D(planes)
        encoder_layer = nn.TransformerEncoderLayer(d_model=planes, dim_feedforward=planes, nhead=8, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
       # encoder_layer = nn.TransformerEncoderLayer(d_model=32, dim_feedforward=planes, nhead=8, dropout=0.2)
       # self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.use_cls_token = True
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, planes))
            self.to_out = nn.Sequential(
                nn.LayerNorm(planes),
                nn.Linear(planes, N_classes)
            )
        else:
            self.classifier = nn.Linear(planes,N_classes)

    def forward(self, x,y=None):
        b = x.shape[0]
        #print(x.shape)
        x= rearrange(x,'b t k a -> b t (k a)')
        #x = self.eca1(x)
        #x = self.pool(x)  # self.relu(self.bn(self.conv(x))))
        # print(x.shape)
        x = self.pe(self.embed(x))
        #print(x.shape)
        #x = rearrange(x, 'b c t h w -> (t h w) b c')

        if self.use_cls_token:
            cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
            #print(cls_token.shape,x.shape)
            x = torch.cat((cls_token, x), dim=1)
            #print(x.shape)
            x= rearrange(x,'b t d -> t b d')

            x = self.transformer_encoder(x)
            #print(x.shape,x[0].shape)
            cls_token = x[0]
            y_hat = cls_token  # self.to_out(cls_token)
        else:
            # print(x.shape)
            x = rearrange(x, 'b t d -> t b d')

            x = self.transformer_encoder(x)
            #x = self.transformer_encoder(rearrange(x, 't b d -> d b t'))
            x = torch.mean(x, dim=0)
            y_hat = self.classifier(x)

        if y!=None:
            loss = F.cross_entropy(y_hat, y.squeeze(-1))
            return y_hat, loss
        return y_hat

# m = SkeletonTR()
# from torchsummary import summary
# summary(m,(64,133*3),device='cpu')