import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from models.vmz.layers import ECA_3D
from .resnet import Bottleneck


class SelfAttention(nn.Module):
    """
    Implementation of plain self attention mechanism with einsum operations
    Paper: https://arxiv.org/abs/1706.03762
    Blog: https://theaisummer.com/transformer/
    """

    def __init__(self, dim):
        """
        Args:
            dim: for NLP it is the dimension of the embedding vector
            the last dimension size that will be provided in forward(x)
            where x is a 3D tensor
        """
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim * 3, bias=False)
        self.scale_factor = dim ** -0.5  # 1/np.sqrt(dim)

    def forward(self, x, mask=None):
        assert x.dim_head() == 3, '3D tensor must be provided'
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3 ]

        # decomposition to q,v,k
        # rearrange tensor to [3, batch, tokens, dim] and cast to tuple
        q, k, v = tuple(rearrange(qkv, 'b t (d k) -> k b t d ', k=3))

        # Resulting shape: [batch, tokens, tokens]
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        return torch.einsum('b i j , b j d -> b i d', attention, v)


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

        self.conv = nn.Conv3d(planes, planes, kernel_size=(1, k, k), stride=(1, s, s), padding=(0, 0, 0),
                              dilation=(1, d, d), bias=False, groups=planes)

        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool3d((downsample_scale, 1, 1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=planes, dim_feedforward=planes, nhead=8, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        # print(x.shape)

        x = self.pool(x).squeeze(-1).squeeze(-1).permute(2, 0, 1)
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        x = torch.mean(x, dim=0)

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
        self.conv1 = nn.Conv3d(planes, planes // 2, kernel_size=(k, k, k), stride=(s, s, s), padding=(0, 0, 0),
                               dilation=(1, d, d), bias=False, groups=planes // 2)
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


class PyramidResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False, late_fusion=True):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(PyramidResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)

        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.tpn5 = ConvModule(512 * block.expansion, 512 * block.expansion, 1, 1,pool_size=)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.same_size = False
        self.information_sum = False
        if self.same_size:
            self.tpn_dim = 128
            self.tpn1 = ConvModule(64 * block.expansion, self.tpn_dim * block.expansion, 1, 1, pool_size=8, groups=64)
            self.aux1 = AuxHead(self.tpn_dim * block.expansion, 226, loss_weight=0.1)
            self.tpn2 = ConvModule(128 * block.expansion, self.tpn_dim * block.expansion, 1, 1, pool_size=4,
                                   downsample_scale=2, groups=128)
            self.aux2 = AuxHead(self.tpn_dim * block.expansion, 226, loss_weight=0.1)
            self.tpn3 = ConvModule(256 * block.expansion, self.tpn_dim * block.expansion, 1, 1, pool_size=2,
                                   downsample_scale=4, groups=256)
            self.aux3 = AuxHead(self.tpn_dim * block.expansion, 226, loss_weight=0.2)
            self.tpn4 = ConvModule(512 * block.expansion, self.tpn_dim * block.expansion, 1, 1, downsample_scale=8,
                                   groups=512)
            self.aux4 = AuxHead(self.tpn_dim * block.expansion, 226, loss_weight=0.5)
        else:
            self.tpn1 = ConvModule(64 * block.expansion, 64 * block.expansion, k=7, s=3, d=5, downsample_scale=64)
            # self.aux1 = AuxHead(64 * block.expansion, 226, loss_weight=0.01)
            self.tpn2 = ConvModule(128 * block.expansion, 128 * block.expansion, k=5, s=2, d=4,
                                   downsample_scale=32)
            # self.aux2 = AuxHead(128 * block.expansion, 226, loss_weight=0.01)
            self.tpn3 = ConvModule(256 * block.expansion, 256 * block.expansion, k=3, s=1, d=2,
                                   downsample_scale=16)
            # self.aux3 = AuxHead(256 * block.expansion, 226, loss_weight=0.05)
            self.tpn4 = ConvModule(512 * block.expansion, 512 * block.expansion, k=1, s=1, d=1, downsample_scale=8)
            # self.aux4 = AuxHead(512 * block.expansion, 226, loss_weight=0.2)

        # self.late_fusion = late_fusion
        # if self.late_fusion:
        #
        #     self.classifier = nn.Linear(5*226, 226)
        self.mha = False
        if self.mha:
            self.att = nn.MultiheadAttention(self.tpn_dim * block.expansion, 8, 0.2)
            self.classifier = nn.Linear(self.tpn_dim * block.expansion, 226)
        else:
            #     k = int(np.log2((64+256+512+128) * block.expansion))
            #     if k % 2 == 0:
            #         k -= 1
            #     self.att = ECA_3D(k_size=k)

            self.att = nn.Linear((64 + 256 + 512 + 128) * 2, (64 + 256 + 512 + 128) * 2)

            self.classifier = nn.Linear((64 + 256 + 512 + 128) * 2, 226)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x, target=None):
        with torch.no_grad():
            x = self.stem(x)

            x = self.layer1(x)
        # print(f'layer 1 {x.shape}')
        tpn1 = self.tpn1(x)
        # print(f' tpn1 {tpn1.shape}')
        with torch.no_grad():
            x = self.layer2(x)
        tpn2 = self.tpn2(x)  # + tpn1
        # print(f'layer 2 {x.shape}{ tpn2.shape}')

        # with torch.no_grad():
        x = self.layer3(x)
        tpn3 = self.tpn3(x)  # + tpn2
        # print(f'layer 3 {x.shape} {tpn3.shape}')
        # with torch.no_grad():
        x = self.layer4(x)
        tpn4 = self.tpn4(x)  # + tpn3
        # print(f'layer 4 {x.shape}  {tpn4.shape}')

        if self.information_sum:
            tpn3 += F.interpolate(tpn4, size=(2, 7, 7))
            tpn2 += F.interpolate(tpn3, size=(4, 7, 7))
            tpn1 += F.interpolate(tpn2, size=(8, 7, 7))

            tpn4 += F.interpolate(tpn3, size=(1, 7, 7))
            tpn3 += F.interpolate(tpn2, size=(2, 7, 7))
            tpn2 += F.interpolate(tpn1, size=(4, 7, 7))
        if target != None:
            # y1,aux1loss = self.aux1(tpn1,target)
            # y2,aux2loss = self.aux2(tpn2,target)
            # y3,aux3loss = self.aux3(tpn3,target)
            # y4,aux4loss = self.aux4(tpn4,target)

            # print(x_new.shape)
            if self.mha:
                tpn1 = F.adaptive_avg_pool3d(tpn1, 1).squeeze(-1).squeeze(-1)
                tpn2 = F.adaptive_avg_pool3d(tpn2, 1).squeeze(-1).squeeze(-1)
                tpn3 = F.adaptive_avg_pool3d(tpn3, 1).squeeze(-1).squeeze(-1)
                tpn4 = F.adaptive_avg_pool3d(tpn4, 1).squeeze(-1).squeeze(-1)
                # print(tpn1.shape)
                x_new = torch.cat((tpn1, tpn2, tpn3, tpn4), dim=-1).permute(2, 0, 1)
                out, weights = self.att(x_new, x_new, x_new)
                x = self.classifier(out.mean(dim=0))
            else:
                ##print(tpn1.shape)
                # tpn1 = F.adaptive_avg_pool3d(tpn1, 1).squeeze(-1).squeeze(-1).squeeze(-1)
                # tpn2 = F.adaptive_avg_pool3d(tpn2, 1).squeeze(-1).squeeze(-1).squeeze(-1)
                # tpn3 = F.adaptive_avg_pool3d(tpn3, 1).squeeze(-1).squeeze(-1).squeeze(-1)
                # tpn4 = F.adaptive_avg_pool3d(tpn4, 1).squeeze(-1).squeeze(-1).squeeze(-1)
                x_new = torch.cat((tpn1, tpn2, tpn3, tpn4), dim=1)
                # print(x_new.shape)
                # x_att = self.att(x_new)
                # x_A = F.adaptive_avg_pool3d(x_att, 1).squeeze(-1).squeeze(-1).squeeze(-1)
                # #print(x_A.shape)
                # 3print(x_new.shape)
                w = torch.softmax(self.att(x_new), dim=-1)
                # print(w.shape)
                x_new = w * x_new
                x = self.classifier(x_new)
            # loss = F.cross_entropy(x, target.squeeze(-1))

        if target != None:
            return x, None
        return x
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
    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
                else:
                    print('no bias')


class PyramidResNetVAE(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(PyramidResNetVAE, self).__init__()
        self.inplanes = 64

        self.stem = stem()
        self.tpn_dim = 128

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)

        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # self.tpn5 = ConvModule(512 * block.expansion, 512 * block.expansion, 1, 1,pool_size=)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.classifier = nn.Linear(2048 + 4*self.tpn_dim* block.expansion, 226)
        # init weights

        latent_dim = 256
        self.m1 = nn.Linear(self.tpn_dim * block.expansion, latent_dim)
        self.s1 = nn.Linear(self.tpn_dim * block.expansion, latent_dim)
        self.m2 = nn.Linear(self.tpn_dim * block.expansion, latent_dim)
        self.s2 = nn.Linear(self.tpn_dim * block.expansion, latent_dim)
        self.m3 = nn.Linear(self.tpn_dim * block.expansion, latent_dim)
        self.s3 = nn.Linear(self.tpn_dim * block.expansion, latent_dim)
        self.m4 = nn.Linear(self.tpn_dim * block.expansion, latent_dim)
        self.s4 = nn.Linear(self.tpn_dim * block.expansion, latent_dim)
        self.m5 = nn.Linear(2048, latent_dim)
        self.s5 = nn.Linear(2048, latent_dim)
        self.classifier = nn.Linear(5 * latent_dim, 226)

        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def losskl(self, mu, logvar):
        return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

    def forward(self, x, target=None):
        x = self.stem(x)

        x = self.layer1(x)
        # print(f'layer 1 {x.shape}')
        tpn1 = self.tpn1(x)
        # print(f' tpn1 {tpn1.shape}')
        x = self.layer2(x)
        tpn2 = self.tpn2(x)  # + tpn1
        # print(f'layer 2 {x.shape}{ tpn2.shape}')
        x = self.layer3(x)
        tpn3 = self.tpn3(x)  # + tpn2
        # print(f'layer 3 {x.shape} {tpn3.shape}')
        x = self.layer4(x)
        tpn4 = self.tpn4(x)  # + tpn3
        # print(f'layer 4 {x.shape}  {tpn4.shape}')

        tpn3 += F.interpolate(tpn4, size=(2, 7, 7))
        tpn2 += F.interpolate(tpn3, size=(4, 7, 7))
        tpn1 += F.interpolate(tpn2, size=(8, 7, 7))

        tpn1 = F.adaptive_avg_pool3d(tpn1, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        tpn2 = F.adaptive_avg_pool3d(tpn2, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        tpn3 = F.adaptive_avg_pool3d(tpn3, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        tpn4 = F.adaptive_avg_pool3d(tpn4, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        # print(tpn4.shape, tpn3.shape, tpn2.shape, tpn1.shape)
        x = self.avgpool(x)
        x = x.flatten(1)
        m1, s1 = self.m1(tpn1), self.s1(tpn1)
        m2, s2 = self.m2(tpn2), self.s2(tpn2)
        m3, s3 = self.m3(tpn3), self.s3(tpn3)
        m4, s4 = self.m4(tpn4), self.s4(tpn4)
        m5, s5 = self.m5(x), self.s5(x)
        z1 = self.reparameterize(m1, s1)
        z2 = self.reparameterize(m2, s2)
        z3 = self.reparameterize(m3, s3)
        z4 = self.reparameterize(m4, s4)
        z5 = self.reparameterize(m5, s5)

        # Flatten the layer to fc
        if target != None:
            aux1loss = self.aux1(tpn1, target)
            aux2loss = self.aux2(tpn2, target)
            aux3loss = self.aux3(tpn3, target)
            aux4loss = self.aux4(tpn4, target)

        loss_kl = 0.001 * (
                0.1 * self.losskl(m1, s1) + 0.1 * self.losskl(m2, s2) + 0.3 * self.losskl(m3, s3) + 0.5 * self.losskl(
            m4, s4) + 0.8 * self.losskl(
            m5, s5))
        y_hat = self.classifier(torch.cat((z1, z2, z3, z4, z5), dim=-1))

        if target != None:
            return y_hat, aux4loss + aux3loss + aux2loss + aux1loss + loss_kl
        return y_hat, loss_kl

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
