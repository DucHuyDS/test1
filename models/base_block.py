import math

import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from models.registry import CLASSIFIER


class BaseClassifier(nn.Module):

    def fresh_params(self, bn_wd):
        if bn_wd:
            return self.parameters()
        else:
            return self.named_parameters()

@CLASSIFIER.register("linear")
class LinearClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=1):
        super().__init__()

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)

        self.logits = nn.Sequential(
            nn.Linear(c_in, nattr),
            nn.BatchNorm1d(nattr) if bn else nn.Identity()
        )


    def forward(self, feature, label=None):
        if len(feature.shape) == 3:  # for vit (bt, nattr, c)
            bt, hw, c = feature.shape
            # NOTE ONLY USED FOR INPUT SIZE (256, 192)
            h = 16
            w = 12
            feature = feature.reshape(bt, h, w, c).permute(0, 3, 1, 2)

        if len(feature.shape) == 2:
            feat = feature
        else:
            feat = self.pool(feature).view(feature.size(0), -1)
        x = self.logits(feat)
        return [x], feature


def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)


class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier, bn_wd=True):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.bn_wd = bn_wd

    def fresh_params(self):
        return self.classifier.fresh_params(self.bn_wd)

    def finetune_params(self):

        if self.bn_wd:
            return self.backbone.parameters()
        else:
            return self.backbone.named_parameters()

    def forward(self, x, label=None):
        try:
            feat_map = self.backbone.extract_feat(x, stage='backbone')[-1]
        except:
            feat_map = self.backbone(x)
        logits, feat = self.classifier(feat_map, label)
        
        if torch.onnx.is_in_onnx_export():
            logits = logits[0].sigmoid()
            return logits
        return logits, feat
