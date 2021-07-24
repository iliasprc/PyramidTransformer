import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torchvision import models
import torchmetrics
from base.base_model import BaseModel
from models.transformers.transformer import TransformerEncoder
from utils.ctcl import CTCL
from utils.metrics import WER,BAcc

import copy




class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MS_TCN2(nn.Module):
   def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes=1):
        super(MS_TCN2, self).__init__()
        self.reduce = nn.Conv1d(1280,num_f_maps,1)
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])


   def forward(self, x):
        out = self.PG(self.reduce(x))
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(out)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes=1):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes=1):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out
class SLDetection_2dcnntcl(BaseModel):
    def __init__(self,config, hidden_size=512, n_layers=2, dropt=0.5, bi=True, N_classes=1232,
                 backbone='repvgg_b0'):
        """

        :param hidden_size: Hidden size of BLSTM
        :param n_layers: Number of layers of BLSTM
        :param dropt: Dropout ratio of BLSTM
        :param bi: Bidirectional
        :param N_classes:
        :param mode:
        :param backbone:
        """
        super(SLDetection_2dcnntcl, self).__init__(config)

        self.name = backbone
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.n_cl = N_classes



        self.loss = nn.BCEWithLogitsLoss()
        # for end-to-end
        self.padding = 1


        if bi:
            hidden_size = 2 * hidden_size
        #self.mix_t_conv_0 = MixTConvHead(3)
        self.select_backbone(backbone)

        self.use_temporal = True
        planes = self.dim_feats
        if self.use_temporal:
            self.temp_channels = 1024
            self.tc_kernel_size = 5
            self.tc_pool_size = 2
            #self.mix_t_conv_1 = MixTConv(self.dim_feats)

            self.temporal = MS_TCN2(num_layers_PG=5,num_layers_R=5,num_R=3,num_f_maps=1024,dim=1024)



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

        self.train_acc = BAcc()
        self.valid_acc = BAcc()
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
        predictions = self.temporal(rearrange(c_out, 'b t d -> b d t')).squeeze(-1)

        #print(out.shape)
        #y_hat = self.fc(rearrange(out, 'b d t -> (b t) d'))

        #print(y_hat.shape,y.float().permute(1,0).shape)
        loss = 0
        for p in predictions:
            #print(p.transpose(2, 1).shape,p.transpose(2, 1).contiguous().view(-1, 1).shape)
            loss += self.loss(p.transpose(2, 1).contiguous().view(-1, 1), y.view(-1,1).float())
            loss+=0.2*F.mse_loss(p.contiguous().view(-1),y.view(-1).float())

        return p.transpose(2, 1).contiguous().view(-1, 1),loss


    def training_step(self, train_batch, batch_idx=None):
        x, y = train_batch
        y_hat, loss = self.forward(x, y)
        trainwer = self.train_acc(y_hat,y)
        self.log('train_acc', self.train_acc, on_step=True,on_epoch=True,prog_bar=True)

        self.log('train_loss',loss.item(), on_step=True, on_epoch=True,prog_bar=True)
        return {'loss':loss,'out':y_hat}

    def validation_step(self, train_batch, batch_idx=None):
        x, y = train_batch
        y_hat, loss = self.forward(x, y)
        acc = self.valid_acc( y_hat, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True,prog_bar=True)
        self.log('valid_loss',loss.item(), on_step=True, on_epoch=True,prog_bar=True)

        return {'loss':loss,'out':y_hat}

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
            self.cnn = timm.create_model(backbone,pretrained=True)
            self.cnn.classifier = Identity()
            self.cnn.head = Identity()
            self.dim_feats = 1280

class CSLDR(BaseModel):
    def __init__(self,config, hidden_size=512, n_layers=2, dropt=0.5, bi=True, N_classes=1232,
                 backbone='mobilenetv3_large_100_miil_in21k'):
        """

        :param hidden_size: Hidden size of BLSTM
        :param n_layers: Number of layers of BLSTM
        :param dropt: Dropout ratio of BLSTM
        :param bi: Bidirectional
        :param N_classes:
        :param mode:
        :param backbone:
        """
        super(CSLDR, self).__init__(config)

        self.name = backbone
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.n_cl = N_classes



        self.loss = nn.BCEWithLogitsLoss()
        # for end-to-end
        self.padding = 1


        if bi:
            hidden_size = 2 * hidden_size
        #self.mix_t_conv_0 = MixTConvHead(3)
        self.select_backbone(backbone)

        self.use_temporal = True
        planes = self.dim_feats
        if self.use_temporal:
            self.temp_channels = 1024
            self.tc_kernel_size = 5
            self.tc_pool_size = 2
            #self.mix_t_conv_1 = MixTConv(self.dim_feats)

            self.temporal = MS_TCN2(num_layers_PG=5,num_layers_R=5,num_R=3,num_f_maps=1024,dim=1024)



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

        self.train_acc = BAcc()
        self.valid_acc = BAcc()
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
        predictions = self.temporal(rearrange(c_out, 'b t d -> b d t')).squeeze(-1)

        #print(out.shape)
        #y_hat = self.fc(rearrange(out, 'b d t -> (b t) d'))

        #print(y_hat.shape,y.float().permute(1,0).shape)
        loss = 0
        for p in predictions:
            #print(p.transpose(2, 1).shape,p.transpose(2, 1).contiguous().view(-1, 1).shape)
            loss += self.loss(p.transpose(2, 1).contiguous().view(-1, 1), y.view(-1,1).float())
            loss+=0.2*F.mse_loss(p.contiguous().view(-1),y.view(-1).float())

        return p.transpose(2, 1).contiguous().view(-1, 1),loss


    def training_step(self, train_batch, batch_idx=None):
        x, y = train_batch
        y_hat, loss = self.forward(x, y)
        trainwer = self.train_acc(y_hat,y)
        self.log('train_acc', self.train_acc, on_step=True,on_epoch=True,prog_bar=True)

        self.log('train_loss',loss.item(), on_step=True, on_epoch=True,prog_bar=True)
        return {'loss':loss,'out':y_hat}

    def validation_step(self, train_batch, batch_idx=None):
        x, y = train_batch
        y_hat, loss = self.forward(x, y)
        acc = self.valid_acc( y_hat, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True,prog_bar=True)
        self.log('valid_loss',loss.item(), on_step=True, on_epoch=True,prog_bar=True)

        return {'loss':loss,'out':y_hat}

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
            self.cnn = timm.create_model(backbone,pretrained=True)
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
