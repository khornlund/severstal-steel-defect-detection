"""
https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/asanakoy/losses.py
https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/utils/criterion/dice.py
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


class SoftDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None
):
    """
    Computes the dice metric
    Args:
        outputs (list):  A list of predicted elements
        targets (list): A list of elements that are to be predicted
        eps (float): epsilon
        threshold (float): threshold for outputs binarization
    Returns:
        double:  Dice score
    """
    outputs = F.sigmoid(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7, threshold: float = None):
        super().__init__()

        self.loss_fn = partial(
            dice,
            eps=eps,
            threshold=threshold,
        )

    def forward(self, logits, targets):
        dice = self.loss_fn(logits, targets)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(
            self,
            eps: float = 1e-7,
            threshold: float = None,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
    ):
        super().__init__()

        if bce_weight == 0 and dice_weight == 0:
            raise ValueError(
                "Both bce_wight and dice_weight cannot be "
                "equal to 0 at the same time."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.bce_weight != 0:
            self.bce_loss = nn.BCEWithLogitsLoss()

        if self.dice_weight != 0:
            self.dice_loss = DiceLoss(eps=eps, threshold=threshold)

    def forward(self, outputs, targets):
        if self.bce_weight == 0:
            return self.dice_weight * self.dice_loss(outputs, targets)
        if self.dice_weight == 0:
            return self.bce_weight * self.bce_loss(outputs, targets)

        bce = self.bce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)

        return (self.bce_weight * bce + self.dice_weight * dice), bce, dice


class SmoothBCELoss(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.smoother = LabelSmoother(eps)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        return self.loss(
            self.smoother(outputs),
            self.smoother(targets)
        )


class SmoothBCEDiceLoss(BCEDiceLoss):

    def __init__(
            self,
            eps: float = 1e-7,
            threshold: float = None,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
    ):
        super().__init__(eps, threshold, bce_weight, dice_weight)
        self.bce_loss = SmoothBCELoss(eps)


# -- utils ----------------------------------------------------------------------------------------

class LabelSmoother:
    """
    Maps binary labels (0, 1) to (eps, 1 - eps)
    """
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.scale = 1 - 2 * self.eps
        self.bias = self.eps / self.scale

    def __call__(self, t):
        return (t + self.bias) * self.scale
