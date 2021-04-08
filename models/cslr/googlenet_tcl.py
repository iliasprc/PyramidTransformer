import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ctc_loss import CTC_Loss
from torchvision import models
from einops import rearrange


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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GoogLeNet_TConvs(nn.Module):
    def __init__(self, hidden_size=512, n_layers=2, dropt=0.5, bi=True, N_classes=1232, mode='isolated',
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
        super(GoogLeNet_TConvs, self).__init__()

        self.name = backbone
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.n_cl = N_classes
        self.mode = mode

        if (self.mode == 'continuous'):
            self.loss = CTC_Loss()
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
            encoder_layer = nn.TransformerEncoderLayer(d_model=1024, dim_feedforward=2*1024, nhead=8, dropout=0.3)
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

    def forward(self, x,y=None):
        # select continous or isolated
        if (self.mode == 'continuous'):
            return self.continuous_forwardtr(x,y)
        elif (self.mode == 'isolated'):
            return self.isolated_forward(x,y)

        return None

    def continuous_forward(self, x,y):
        with torch.no_grad():
            batch_size,  C,timesteps, H, W = x.size()
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
        if y!=None:
            loss_ctc = self.loss(y_hat,y)
            return y_hat,loss_ctc
        return y_hat


    def continuous_forwardtr(self, x,y):
        with torch.no_grad():
            batch_size,  C,timesteps, H, W = x.size()
            c_in = x.view(batch_size * timesteps, C, H, W)
            c_outputs = self.cnn(c_in)
            c_out = c_outputs.contiguous().view(batch_size, timesteps, -1)
        # c_out has size timesteps x dim feats
        # temporal layers gets input size batch_size x dim_feats x timesteps
        temp_input = c_out.permute(0, 2, 1)
        temp = self.temporal(temp_input)
        # temporal layers output size batch_size x dim_feats x timesteps
        # rnn input must be timesteps x batch_size x dim_feats
        rnn_input = rearrange(temp,'b c t -> b t c')

        x = self.pe(rnn_input)
        x = rearrange(x, 'b t d -> t b d')
        x = self.transformer_encoder(x)

        y_hat = self.fc(x)
        if y!=None:
            loss_ctc = self.loss(y_hat,y)
            return y_hat,loss_ctc
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

    def isolated_forward(self, x,y=None):

        batch_size, C, timesteps,H, W = x.size()
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
        #print(y_hat.shape)
        if y!=None:
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
