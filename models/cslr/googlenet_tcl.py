import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from einops import repeat, rearrange
from torchvision import models

from base.base_model import BaseModel
from models.transformers.transformer import TransformerEncoder
from utils.ctcl import CTCL
from utils.metrics import WER


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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                             stride=stride, padding=padding, dilation=dilation, groups=n_outputs // 4),
                                   nn.BatchNorm1d(n_outputs))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Sequential(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                             stride=stride, padding=padding, dilation=dilation, groups=n_outputs),
                                   nn.BatchNorm1d(n_outputs))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool1d(2, 2)
        # self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # print(out.shape,res.shape)
        return self.pool(self.relu(out + res))


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                # print(m)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


# m = TemporalConvNet(128,[128,128,128,128,128])
# a = torch.randn(4,128,32)
# o = m(a)
# print(o.shape)
class MixTConv(nn.Module):

    def __init__(self, num_channels, split=4):
        super(MixTConv, self).__init__()

        self.split = split
        self.conv1 = nn.Conv1d(in_channels=num_channels // self.split, out_channels=num_channels // self.split,
                               kernel_size=1, groups=num_channels // self.split)
        self.conv2 = nn.Conv1d(in_channels=num_channels // self.split, out_channels=num_channels // self.split,
                               kernel_size=3, padding=1,
                               groups=num_channels // self.split)
        self.conv3 = nn.Conv1d(in_channels=num_channels // self.split, out_channels=num_channels // self.split,
                               kernel_size=5, padding=2,
                               groups=num_channels // self.split)
        self.conv4 = nn.Conv1d(in_channels=num_channels // self.split, out_channels=num_channels // self.split,
                               kernel_size=7, padding=3,
                               groups=num_channels // self.split)

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, self.split, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x3 = self.conv1(x3)
        x4 = self.conv1(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1) + x
        return x


class MixTConvHead(nn.Module):

    def __init__(self, num_channels, split=1):
        super(MixTConvHead, self).__init__()

        self.split = split
        self.conv1 = nn.Conv1d(in_channels=num_channels // self.split, out_channels=num_channels // self.split,
                               kernel_size=1, groups=num_channels // self.split)
        self.conv2 = nn.Conv1d(in_channels=num_channels // self.split, out_channels=num_channels // self.split,
                               kernel_size=3, padding=1,
                               groups=num_channels // self.split)
        self.conv3 = nn.Conv1d(in_channels=num_channels // self.split, out_channels=num_channels // self.split,
                               kernel_size=5, padding=2,
                               groups=num_channels // self.split)
        self.conv4 = nn.Conv1d(in_channels=num_channels // self.split, out_channels=num_channels // self.split,
                               kernel_size=7, padding=3,
                               groups=num_channels // self.split)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1(x)
        x3 = self.conv1(x)
        x4 = self.conv1(x)

        x = x1 + x2 + x3 + x4 + x
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GoogLeNet_TConvs(BaseModel):
    def __init__(self, config, hidden_size=512, n_layers=2, dropt=0.5, bi=True, N_classes=1232, mode='isolated',
                 backbone='googlenet'):
        """

        :param hidden_size: Hidden size of BLSTM
        :param n_layers: Number of layers of BLSTM
        :param dropt: Dropout ratio of BLSTM
        :param bi: Bidirectional
        :param N_classes:
        :param mode:
        :param backbone:
        """
        super(GoogLeNet_TConvs, self).__init__(config)

        self.name = backbone
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.n_cl = N_classes
        self.mode = mode

        if (self.mode == 'continuous'):
            self.loss = CTCL()
            # for end-to-end
            self.padding = 1
        else:
            # for feature extractor
            self.padding = 0

        if bi:
            hidden_size = 2 * hidden_size
        self.select_backbone(backbone)
        self.temp_channels = 1024
        self.tc_kernel_size = 5
        self.tc_pool_size = 2
        self.temporal = torch.nn.Sequential(
            nn.Conv1d(self.dim_feats, 1024, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
            nn.Conv1d(1024, 1024, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size))

        cslr = True
        if cslr:
            self.pe = PositionalEncoding1D(1024)
            encoder_layer = nn.TransformerEncoderLayer(d_model=1024, dim_feedforward=2 * 1024, nhead=8, dropout=0.3)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.fc = nn.Linear(1024, self.n_cl)
        else:
            self.rnn = nn.LSTM(
                input_size=self.temp_channels,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=dropt,
                bidirectional=True)

            if bi:
                self.fc = nn.Linear(2 * self.hidden_size, self.n_cl)
            else:
                self.fc = nn.Linear(self.hidden_size, self.n_cl)

        # if self.mode == 'continuous':
        #     self.init_param()

    def forward(self, x, y=None):
        # select continous or isolated
        if (self.mode == 'continuous'):
            return self.continuous_forwardtr(x, y)
        elif (self.mode == 'isolated'):
            return self.isolated_forward(x, y)

        return None

    def continuous_forward(self, x, y):
        with torch.no_grad():
            batch_size, C, timesteps, H, W = x.size()
            c_in = x.view(batch_size * timesteps, C, H, W)
            c_outputs = self.cnn(c_in)
            c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)
        # c_out has size timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)
        # temporal layers output size batch_size x dim_feats x timesteps
        # rnn input must be timesteps x batch_size x dim_feats
        rnn_input = temp.permute(2, 0, 1)
        rnn_out, (h_n, h_c) = self.rnn(rnn_input)
        fc_input = rnn_out.squeeze(1)
        y_hat = self.fc(fc_input).unsqueeze(1)
        if y != None:
            loss_ctc = self.loss(y_hat, y)
            return y_hat, loss_ctc
        return y_hat

    def continuous_forwardtr(self, x, y):
        with torch.no_grad():
            batch_size, C, timesteps, H, W = x.size()
            c_in = x.view(batch_size * timesteps, C, H, W)
            c_outputs = self.cnn(c_in)
            c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)
        # c_out has size timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)
        # temporal layers output size batch_size x dim_feats x timesteps
        # rnn input must be timesteps x batch_size x dim_feats
        rnn_input = rearrange(temp, 'b c t -> b t c')

        x = self.pe(rnn_input)
        x = rearrange(x, 'b t d -> t b d')
        x = self.transformer_encoder(x)

        y_hat = self.fc(x)
        if y != None:
            loss_ctc = self.loss(y_hat, y)
            return y_hat, loss_ctc
        return y_hat

    def __inference__(self, x):
        batch_size, timesteps, C, H, W = x.size()

        c_in = x.view(batch_size * timesteps, C, H, W)
        c_outputs = []
        for i in range(batch_size * timesteps):
            out = self.cnn(c_in[i, ...].unsqueeze(0))
            c_outputs.append(out)

        c_out = torch.stack(c_outputs)

        c_out = c_out.contiguous().view(batch_size, timesteps, -1)
        # c_out has size timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)
        # temporal layers output size batch_size x dim_feats x timesteps
        # rnn input must be timesteps x batch_size x dim_feats
        rnn_input = temp.permute(2, 0, 1)
        rnn_out, (h_n, h_c) = self.rnn(rnn_input)
        fc_input = rnn_out.squeeze(1)
        r_out = self.last_linear(fc_input).unsqueeze(1)

        return r_out

    def isolated_forward(self, x, y=None):

        batch_size, C, timesteps, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_outputs = self.cnn(c_in)
        c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)
        # train only feature extractor
        # c_out has size batch_size x timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)
        # last linear input must be timesteps x batch_size x dim_feats
        fc_input = temp.permute(2, 0, 1).squeeze(0)
        y_hat = self.fc(fc_input)
        print(y_hat.shape)
        if y != None:
            loss = F.cross_entropy(y_hat, y.squeeze(-1))
            return y_hat, loss
        return y_hat

    def init_param(self):
        count = 0
        for param in self.cnn.parameters():
            count += 1
            param.requires_grad = False

    def select_backbone(self, backbone):

        if (backbone == 'alexnet'):
            self.aux_logits = False
            self.cnn = models.alexnet(pretrained=True)

            self.dim_feats = 4096

            self.cnn.classifier[-1] = Identity()

        elif (backbone == 'googlenet'):
            self.aux_logits = False
            from torchvision.models import googlenet

            self.cnn = googlenet(pretrained=True, transform_input=False, aux_logits=self.aux_logits)
            self.cnn.fc = Identity()
            self.dim_feats = 1024
            count = 0

        elif (backbone == 'mobilenet'):
            self.aux_logits = False
            from torchvision.models import mobilenet_v2

            self.cnn = mobilenet_v2(pretrained=True)
            self.cnn.fc = Identity()
            self.cnn.classifier = Identity()
            self.dim_feats = 1280
            count = 0
        elif backbone == 'pruned_mobilenet':
            self.cnn = load_pruned_mobilenet()
            self.cnn.fc = Identity()
            self.cnn.classifier = Identity()
            self.dim_feats = 1280


class ISL_cnn(BaseModel):
    def __init__(self, config, hidden_size=512, n_layers=2, dropt=0.5, bi=True, N_classes=1232, mode='isolated',
                 backbone=''):
        """

        :param hidden_size: Hidden size of BLSTM
        :param n_layers: Number of layers of BLSTM
        :param dropt: Dropout ratio of BLSTM
        :param bi: Bidirectional
        :param N_classes:
        :param mode:
        :param backbone:
        """
        super(ISL_cnn, self).__init__(config)

        self.name = backbone
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.n_cl = N_classes
        self.mode = mode

        if (self.mode == 'continuous'):
            self.loss = CTCL()
            # for end-to-end
            self.padding = 1
        else:
            # for feature extractor
            self.padding = 0

        if bi:
            hidden_size = 2 * hidden_size
        # self.mix_t_conv_0 = MixTConvHead(3)
        self.select_backbone(backbone)

        self.use_temporal = True
        planes = self.dim_feats
        if self.use_temporal:
            self.temp_channels = 1024
            self.tc_kernel_size = 5
            self.tc_pool_size = 2
            # self.mix_t_conv_1 = MixTConv(self.dim_feats)
            from models.layers.temporal import MS_TCN2, TCL
            self.ms_tcn = MS_TCN2(num_layers_PG=5, num_layers_R=5, num_R=3, num_f_maps=1024, dim=1024,average=True)
            self.temporal = TCL(num_input_feats=1024, dim=1024, n_layers=2, kernel_size=self.tc_kernel_size, stride=1,
                                pooling_size=self.tc_pool_size, dropout=0.1, groupwise=True)

            # self.temporal= TemporalConvNet(self.dim_feats, [1024,1024,1024,1024])
            # self.temporal = torch.nn.Sequential(
            #     nn.Conv1d(self.dim_feats, 1024, kernel_size=self.tc_kernel_size, stride=2, padding=self.padding),
            #     nn.LeakyReLU(0.1),
            #     nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size),
            #     nn.Conv1d(1024, 1024, kernel_size=self.tc_kernel_size, stride=1, padding=self.padding),
            #     nn.LeakyReLU(0.1),
            #     nn.MaxPool1d(self.tc_pool_size, self.tc_pool_size))

            planes = 1024

        else:
            max_tokens = 32 + 1
            self.pe = PositionalEncoding1D(dim=planes, max_tokens=max_tokens)
            self.transformer_encoder = TransformerEncoder(dim=planes, blocks=3, heads=8, dim_head=64,
                                                          dim_linear_block=planes * 2, dropout=0.2)
            self.use_cls_token = True

            self.cls_token = nn.Parameter(torch.randn(1, planes))
        self.fc = nn.Sequential(
            # nn.LayerNorm(planes),
            nn.Linear(planes, self.n_cl)
        )

        # if self.mode == 'continuous':
        #     self.init_param()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, x, y=None):

        return self.isolated_forward(x, y)




    def __inference__(self, x):
        batch_size, timesteps, C, H, W = x.size()

        c_in = x.view(batch_size * timesteps, C, H, W)
        c_outputs = []
        for i in range(batch_size * timesteps):
            out = self.cnn(c_in[i, ...].unsqueeze(0))
            c_outputs.append(out)

        c_out = torch.stack(c_outputs)

        c_out = c_out.contiguous().view(batch_size, timesteps, -1)
        # c_out has size timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)
        # temporal layers output size batch_size x dim_feats x timesteps
        # rnn input must be timesteps x batch_size x dim_feats
        rnn_input = temp.permute(2, 0, 1)
        rnn_out, (h_n, h_c) = self.rnn(rnn_input)
        fc_input = rnn_out.squeeze(1)
        r_out = self.last_linear(fc_input).unsqueeze(1)

        return r_out

    def isolated_forward(self, x, y=None):

        batch_size, C, timesteps, H, W = x.size()

        # c_in = self.mix_t_conv_0( rearrange(x,'b c t h w -> (b h w) c t'))
        c_in = x.view(batch_size * timesteps, C, H, W)  # +  c_in.view(batch_size * timesteps, C, H, W)

        c_outputs = self.cnn(c_in)
        c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)
        # train only feature extractor
        # c_out has size batch_size x timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        if self.use_temporal:
            # temp_input = self.mix_t_conv_1(rearrange(c_out, 'b t d -> b d t'))
            # print(c_out.shape)
            out = self.temporal(self.ms_tcn(rearrange(c_out, 'b t d -> b d t'))).squeeze(-1)

            # print(out.shape)
            y_hat = self.fc(out)
        else:
            cls_token = repeat(self.cls_token, 'n d -> b n d', b=batch_size)
            # print(cls_token.shape,c_out.shape)
            x = torch.cat((cls_token, c_out), dim=1)
            # print(x.shape)
            x = self.pe(x)
            x = self.transformer_encoder(x)
            # print(x.shape,x[0].shape)
            cls_token = x[:, 0, :]

            y_hat = self.fc(cls_token)
        # print(y_hat.shape)
        if y != None:
            loss = F.cross_entropy(y_hat, y.squeeze(-1))
            return y_hat, loss
        return y_hat

    def training_step(self, train_batch, batch_idx=None):
        x, y = train_batch
        y_hat, loss = self.forward(x, y)
        acc = self.train_acc(y_hat, y.squeeze(-1))
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        # dict_for_progress_bar = {'train_acc': acc}
        return {'loss': loss, 'out': y_hat}

    def validation_step(self, train_batch, batch_idx=None):
        x, y = train_batch
        y_hat, loss = self.forward(x, y)
        acc = self.valid_acc(y_hat, y.squeeze(-1))
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        dict_for_progress_bar = {'train_acc': acc}
        return {'loss': loss, 'out': y_hat}

    def configure_optimizers(self):
        return self.optimizer()

    def init_param(self):
        count = 0
        for param in self.cnn.parameters():
            count += 1
            param.requires_grad = False

    def select_backbone(self, backbone):

        if (backbone == 'alexnet'):
            self.aux_logits = False
            self.cnn = models.alexnet(pretrained=True)

            self.dim_feats = 4096

            self.cnn.classifier[-1] = Identity()

        elif (backbone == 'googlenet'):
            self.aux_logits = False
            from torchvision.models import googlenet

            self.cnn = googlenet(pretrained=True, transform_input=False, aux_logits=self.aux_logits)
            self.cnn.fc = Identity()
            self.dim_feats = 1024
            count = 0

        elif (backbone == 'mobilenet2'):
            self.aux_logits = False
            from torchvision.models import mobilenet_v2

            self.cnn = mobilenet_v2(pretrained=True)
            self.cnn.fc = Identity()
            self.cnn.classifier = Identity()
            self.dim_feats = 1280
            count = 0
        elif (backbone == 'mobilenet2'):
            self.aux_logits = False
            from torchvision.models import mobilenet_v2

            self.cnn = mobilenet_v2(pretrained=True)
            self.cnn.fc = Identity()
            self.cnn.classifier = Identity()
            self.dim_feats = 1280
            count = 0
        elif backbone == 'pruned_mobilenet':
            self.cnn = load_pruned_mobilenet()
            self.cnn.fc = Identity()
            self.cnn.classifier = Identity()
            self.dim_feats = 1280
        elif backbone == 'xcit_tiny_12_p16_224':
            import timm
            self.cnn = timm.create_model('xcit_tiny_12_p16_224')
            # self.cnn.fc = Identity()
            self.cnn.head.fc = Identity()
            self.dim_feats = 1280

        else:
            import timm
            backbone = 'mobilenetv3_large_100_miil_in21k'
            self.cnn = timm.create_model(backbone, pretrained=True)
            self.cnn.classifier = Identity()
            self.cnn.head = Identity()
            self.dim_feats = 1280


class CSLR_2dcnntcl(BaseModel):
    def __init__(self, config, hidden_size=512, n_layers=2, dropt=0.5, bi=True, N_classes=1232,
                 backbone=''):
        """

        :param hidden_size: Hidden size of BLSTM
        :param n_layers: Number of layers of BLSTM
        :param dropt: Dropout ratio of BLSTM
        :param bi: Bidirectional
        :param N_classes:
        :param mode:
        :param backbone:
        """
        super(CSLR_2dcnntcl, self).__init__(config)

        self.name = backbone
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.n_cl = N_classes

        self.loss = CTCL()
        # for end-to-end
        self.padding = 1

        if bi:
            hidden_size = 2 * hidden_size
        # self.mix_t_conv_0 = MixTConvHead(3)
        self.select_backbone(backbone)

        self.use_temporal = True
        planes = self.dim_feats
        if self.use_temporal:
            self.temp_channels = 1024
            self.tc_kernel_size = 5
            self.tc_pool_size = 2
            # self.mix_t_conv_1 = MixTConv(self.dim_feats)
            from models.detection.mstcn2 import MS_TCN2
            self.temporal = MS_TCN2(num_layers_PG=5, num_layers_R=5, num_R=3, num_f_maps=1024, dim=1024)
            self.pool1 = nn.MaxPool1d(3, 3)
            self.pool2 = nn.MaxPool1d(3, 3)

            planes = 1024

        else:
            max_tokens = 32 + 1
            self.pe = PositionalEncoding1D(dim=planes, max_tokens=max_tokens)
            self.transformer_encoder = TransformerEncoder(dim=planes, blocks=3, heads=8, dim_head=64,
                                                          dim_linear_block=planes * 2, dropout=0.2)
            self.use_cls_token = True

            self.cls_token = nn.Parameter(torch.randn(1, planes))
        self.fc = nn.Sequential(
            # nn.LayerNorm(planes),
            nn.Linear(planes, self.n_cl)
        )

        # if self.mode == 'continuous':
        #     self.init_param()

        self.train_wer = WER()
        self.valid_wer = WER()

    def forward(self, x, y=None):
        # select continous or isolated

        return self.continuous_forward(x, y)

    def continuous_forward(self, x, y):
        with torch.no_grad():
            batch_size, C, timesteps, H, W = x.size()

            # c_in = self.mix_t_conv_0( rearrange(x,'b c t h w -> (b h w) c t'))
            c_in = x.view(batch_size * timesteps, C, H, W)  # +  c_in.view(batch_size * timesteps, C, H, W)

            c_outputs = self.cnn(c_in)
            c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)
            # train only feature extractor
            # c_out has size batch_size x timesteps x dim feats
            # temporal layers gets input size batch_size x dim_feats x timesteps

        # temp_input = self.mix_t_conv_1(rearrange(c_out, 'b t d -> b d t'))
        # print(c_out.shape)
        out = self.pool2(self.temporal(self.pool1(rearrange(c_out, 'b t d -> b d t')))).squeeze(-1)

        # print(out.shape)
        y_hat = self.fc(rearrange(out, 'b d t -> b t d'))
        y_hat = rearrange(y_hat, 'b t d -> t b d')
        # print(y_hat.shape)
        if y != None:
            loss_ctc = self.loss(y_hat, y)
            return y_hat, loss_ctc
        return y_hat

    def isolated_forward(self, x, y=None):

        batch_size, C, timesteps, H, W = x.size()

        # c_in = self.mix_t_conv_0( rearrange(x,'b c t h w -> (b h w) c t'))
        c_in = x.view(batch_size * timesteps, C, H, W)  # +  c_in.view(batch_size * timesteps, C, H, W)

        c_outputs = self.cnn(c_in)
        c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)
        # train only feature extractor
        # c_out has size batch_size x timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        if self.use_temporal:
            # temp_input = self.mix_t_conv_1(rearrange(c_out, 'b t d -> b d t'))
            # print(c_out.shape)
            out = self.temporal(rearrange(c_out, 'b t d -> b d t')).squeeze(-1)

            # print(out.shape)
            y_hat = self.fc(out)
        else:
            cls_token = repeat(self.cls_token, 'n d -> b n d', b=batch_size)
            # print(cls_token.shape,c_out.shape)
            x = torch.cat((cls_token, c_out), dim=1)
            # print(x.shape)
            x = self.pe(x)
            x = self.transformer_encoder(x)
            # print(x.shape,x[0].shape)
            cls_token = x[:, 0, :]

            y_hat = self.fc(cls_token)
        # print(y_hat.shape)
        if y != None:
            loss = F.cross_entropy(y_hat, y.squeeze(-1))
            return y_hat, loss
        return y_hat

    def training_step(self, train_batch, batch_idx=None):
        x, y = train_batch
        y_hat, loss = self.forward(x, y)
        trainwer = self.train_wer(y_hat, y)
        self.log('train_wer', self.train_wer, on_epoch=True, prog_bar=True)

        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'out': y_hat}

    def validation_step(self, train_batch, batch_idx=None):
        x, y = train_batch
        y_hat, loss = self.forward(x, y)
        acc = self.valid_wer(y_hat, y)
        self.log('valid_wer', self.valid_wer, on_step=True, on_epoch=True, prog_bar=True)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'out': y_hat}

    def configure_optimizers(self):
        return self.optimizer()

    def init_param(self):
        count = 0
        for param in self.cnn.parameters():
            count += 1
            param.requires_grad = False

    def select_backbone(self, backbone):

        if (backbone == 'alexnet'):
            self.aux_logits = False
            self.cnn = models.alexnet(pretrained=True)

            self.dim_feats = 4096

            self.cnn.classifier[-1] = Identity()

        elif (backbone == 'googlenet'):
            self.aux_logits = False
            from torchvision.models import googlenet

            self.cnn = googlenet(pretrained=True, transform_input=False, aux_logits=self.aux_logits)
            self.cnn.fc = Identity()
            self.dim_feats = 1024
            count = 0

        elif (backbone == 'mobilenet2'):
            self.aux_logits = False
            from torchvision.models import mobilenet_v2

            self.cnn = mobilenet_v2(pretrained=True)
            self.cnn.fc = Identity()
            self.cnn.classifier = Identity()
            self.dim_feats = 1280
            count = 0
        elif (backbone == 'mobilenet2'):
            self.aux_logits = False
            from torchvision.models import mobilenet_v2

            self.cnn = mobilenet_v2(pretrained=True)
            self.cnn.fc = Identity()
            self.cnn.classifier = Identity()
            self.dim_feats = 1280
            count = 0
        elif backbone == 'pruned_mobilenet':
            self.cnn = load_pruned_mobilenet()
            self.cnn.fc = Identity()
            self.cnn.classifier = Identity()
            self.dim_feats = 1280
        elif backbone == 'xcit_tiny_12_p16_224':
            import timm
            self.cnn = timm.create_model('xcit_tiny_12_p16_224')
            # self.cnn.fc = Identity()
            self.cnn.head.fc = Identity()
            self.dim_feats = 1280

        else:
            import timm
            backbone = 'mobilenetv3_large_100_miil_in21k'
            self.cnn = timm.create_model(backbone, pretrained=True)
            self.cnn.classifier = Identity()
            self.cnn.head = Identity()
            self.dim_feats = 1280


def keypoint_detector(detector_path=None):
    # load an instance segmentation model pre-trained on COCO
    '''
        During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,  with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
        - keypoints (``FloatTensor[N, K, 3]``): the locations of the predicted keypoints, in ``[x, y, v]`` format.

    Keypoint R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
    '''
    model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

    return model


def load_pruned_mobilenet():
    path_refine = '/home/iliask/PycharmProjects/epikoinwnw_SLR/models/mobilenet_prune/mobilenev2_nodec_w_1.996_par_27' \
                  '.13_FLOPs_43.65.tar'
    path_resume = '/home/iliask/PycharmProjects/epikoinwnw_SLR/models/mobilenet_prune' \
                  '/checkpoint_mobilenetv2_top1_66_top5_87.21.pth.tar'

    checkpoint = torch.load(path_refine, map_location='cpu')
    model = mobilenet_v2(inverted_residual_setting=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])

    print("=> loading checkpoint '{}'".format(path_resume))

    checkpoint = torch.load(path_resume, map_location='cpu')
    best_prec1 = checkpoint['best_prec1']
    print(best_prec1)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(path_resume, checkpoint['epoch']))
    # print(model)
    return model
