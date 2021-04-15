import torch.nn as nn


class Conv1DDepthwise(nn.Conv1d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        assert in_planes == out_planes

        #print(in_planes,out_planes,stride,padding)
        super(Conv1DDepthwise, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=in_planes,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Bottleneck1D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm1d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem_Pool1D(nn.Sequential):
    def __init__(self, input_channels=27 * 3):
        super(BasicStem_Pool1D, self).__init__(
            nn.Conv1d(
                input_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

        )
