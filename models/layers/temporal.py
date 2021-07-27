import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCL(nn.Module):
    def __init__(self, num_input_feats, dim, n_layers, kernel_size, stride, pooling_size=2, dropout=0.1,
                 groupwise=True):
        super(TCL,self).__init__()
        modules = []
        for i in range(n_layers):
            if groupwise:
                groups = dim
                modules.append(nn.Conv1d(num_input_feats, dim, kernel_size=kernel_size, stride=stride, groups=groups))
            else:
                modules.append(nn.Conv1d(num_input_feats, dim, kernel_size=kernel_size, stride=stride))
            modules.append(nn.GELU())
            modules.append(nn.MaxPool1d(pooling_size, pooling_size))
            modules.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MS_TCN2(nn.Module):

    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes=1, average=True):

        super(MS_TCN2, self).__init__()
        self.reduce = nn.Conv1d(1280, num_f_maps, kernel_size=1)
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes, groupwise=True)
        self.Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, dim, num_classes, groupwise=True)) for s in
             range(num_R)])
        self.average = average

    def forward(self, x):
        out = self.PG(self.reduce(x))
        outputs = out
        outputs = out.unsqueeze(0)
        # print('outputs',outputs.shape)
        for R in self.Rs:
            out = R(out)
            # print(out.shape)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        if self.average:
            return outputs.mean(dim=0)  # .mean(dim=-1)
        return outputs


class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, groupwise=True):
        super(Prediction_Generation, self).__init__()
        if groupwise:
            groups = num_f_maps
        else:
            groups = 1
        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1, groups=groups)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers - 1 - i), dilation=2 ** (num_layers - 1 - i),
                      groups=groups)
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i, groups=groups)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1, groups=groups)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        return f


class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes=-1, groupwise=True):

        super(Refinement, self).__init__()
        if groupwise:
            groups = num_f_maps
        else:
            groups = 1
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1, groups=groups)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        # out = self.conv_out(out)
        return out


class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages - 1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, groupwise=True):
        super(DilatedResidualLayer, self).__init__()
        if groupwise:
            groups = out_channels
        else:
            groups = 1
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, groups=groups)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1, groups=groups)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out
