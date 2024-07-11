import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torchvision
import torch
from mmpretrain import get_model
from mmpretrain.models.backbones import EfficientFormer


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'mobilenetv3_l']

from models.registry import BACKBONE


@BACKBONE.register("convnext")
def resnet34_8xb32_in1k(pretrained=True, progress=True, **kwargs):
    backbone =  get_model('convnext-v2-atto_fcmae-pre_3rdparty_in1k', pretrained=True)
    return backbone
    