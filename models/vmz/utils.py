import torch
import torch.nn as nn

from models.vmz.pyramid_resnet import PyramidResNet
from models.vmz.pyramid_transformer import  PyramidTransformerResNet
from models.vmz.resnet import VideoResNet, ECAVideoResNet



def _generic_resnet(arch, pretrained=False, progress=False, **kwargs):
    model = VideoResNet(**kwargs)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict)

    return model


def _eca_resnet(arch, pretrained=False, progress=False, **kwargs):
    model = ECAVideoResNet(**kwargs)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict, strict=False)

    return model


def _pyramid_resnet(arch, pretrained=False, progress=False, **kwargs):
    model = PyramidResNet(**kwargs)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict, strict=False)

    return model


def _pyramid_transformer_resnet(arch, pretrained=False, progress=False, **kwargs):
    model = PyramidTransformerResNet(**kwargs)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict, strict=False)

    return model


def _pyramid_multi_transformer_resnet(arch, pretrained=False, progress=False, **kwargs):
    model = PyramidMultiTransformerResNet(**kwargs)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    # if pretrained:
    #     state_dict = torch.hub.load_state_dict_from_url(
    #         model_urls[arch], progress=progress
    #     )
    #     model.load_state_dict(state_dict,strict=False)

    return model
