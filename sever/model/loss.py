"""
https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/asanakoy/losses.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


class SoftDiceLoss(nn.Module):

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.soft_dice = SoftDiceLoss()

    def forward(self, logits, labels):
        pass


class CombinedBCEDice(CombinedLoss):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        bce_loss = self.bce(logits, labels)
        dice_loss = self.soft_dice(logits, labels)
        return bce_loss + dice_loss


class CombinedBCELogDice(CombinedLoss):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        bce_loss = self.bce(logits, labels)
        dice_loss = self.soft_dice(logits, labels)
        return bce_loss - (1 - dice_loss).log()
