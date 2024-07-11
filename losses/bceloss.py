import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight


@LOSSES.register("bceloss")
class BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None

    def forward(self, logits, targets):
        logits = logits[0]

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)

            loss_m = (loss_m * sample_weight.cuda())

        distance = self.pairwise_distance(logits, targets)
        
        loss = loss_m.sum(1).mean() + distance

        return [loss], [loss_m]
    
    def pairwise_distance(self, logits, targets):
        
        distances_minkowski_p3 = torch.sum(torch.abs(logits[:, None, :] - targets[None, :, :]) ** 3, axis=2) ** (1 / 3)
        
        distances_minkowski_p2 = torch.sqrt(torch.sum((logits - targets) ** 2, axis=1))
        
        return distances_minkowski_p3.mean() + distances_minkowski_p2.mean()