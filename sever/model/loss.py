"""
https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/asanakoy/losses.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)



class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        w = weights.view(num, -1)
        w2 = w * w
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * ((w2 * intersection).sum(1) + 1) / (
            (w2 * m1).sum(1) + (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return score


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
    def __init__(self, is_log_dice=False):
        super().__init__()
        self.is_log_dice = is_log_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.soft_dice = SoftDiceLoss()

    def forward(self, logits, labels):
        bce_loss = self.bce(logits, labels)
        dice_loss = self.soft_dice(logits, labels)

        if self.is_log_dice:
            loss = bce_loss - (1 - dice_loss).log()
        else:
            loss = bce_loss + dice_loss
        # return loss, bce_loss, dice_loss
        return loss
