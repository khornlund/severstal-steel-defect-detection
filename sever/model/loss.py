"""
https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/asanakoy/losses.py
https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/utils/criterion/dice.py
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


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
        loss = self.bce_weight * bce + self.dice_weight * dice
        return {
            'loss': loss,
            'bce': bce,
            'dice': dice
        }


class SmoothBCELoss(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.smoother = LabelSmoother(eps)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        return self.loss(
            self.smoother(outputs),  # this was a mistake, but better results?
            self.smoother(targets)
        )


class ClasSmoothBCELoss(nn.Module):

    def __init__(self, eps=1e-8, pos_weight=[1, 1, 1, 1]):
        super().__init__()
        self.smoother = LabelSmoother(eps)
        pos_weight = torch.tensor(pos_weight)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, outputs, targets):
        return {'loss': self.loss(outputs, self.smoother(targets))}


class SmoothBCEDiceLoss(BCEDiceLoss):

    def __init__(
            self,
            eps: float = 1e-7,
            smooth: float = 1e-6,
            threshold: float = None,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
    ):
        super().__init__(eps, threshold, bce_weight, dice_weight)
        self.bce_loss = SmoothBCELoss(smooth)


class SmoothSoftmaxBCEDiceLoss(nn.Module):

    def __init__(
            self,
            eps: float = 1e-7,
            smooth: float = 1e-6,
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
            self.bce_loss = SmoothSoftmaxBCELoss(smooth=smooth)

        if self.dice_weight != 0:
            self.dice_loss = DiceLoss(eps=eps, threshold=threshold)

    def forward(self, outputs, targets):
        # add background to targets
        background = (torch.sum(targets, dim=1, keepdim=True) == 0).float()
        targets = torch.cat([targets, background], dim=1)
        bce = self.bce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        loss = self.bce_weight * bce + self.dice_weight * dice
        return {
            'loss': loss,
            'bce': bce,
            'dice': dice
        }


class SmoothSoftmaxBCELoss(nn.Module):

    def __init__(self, smooth):
        super().__init__()
        self.act = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.smooth = LabelSmoother(eps=smooth)

    def forward(self, outputs, targets):
        return self.bce(
            self.act(outputs),
            self.smooth(targets)
        )


class CrossEntropyLossWrapper(nn.Module):

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        targets[:, 1, :, :] *= 2
        targets[:, 2, :, :] *= 3
        targets[:, 3, :, :] *= 4
        targets = torch.max(targets, dim=1, keepdim=True)[0].long()
        print(targets)
        return self.ce(outputs, targets)


class DeepSmoothBCEDiceLoss(nn.Module):

    def __init__(
        self,
        eps=1e-7,
        smooth=1e-6,
        threshold=None,
        bce_weight=0.5,
        dice_weight=0.5,
        depth_weights=[1, 0.5, 0.25, 0.1]
    ):
        super().__init__()

        if bce_weight == 0 and dice_weight == 0:
            raise ValueError(
                "Both bce_wight and dice_weight cannot be "
                "equal to 0 at the same time."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.depth_weights = depth_weights

        if self.bce_weight != 0:
            self.bce_loss = SmoothBCELoss(smooth)
        if self.dice_weight != 0:
            self.dice_loss = DiceLoss(eps=eps, threshold=threshold)

    def forward_single(self, outputs, targets):
        if self.bce_weight == 0:
            return self.dice_weight * self.dice_loss(outputs, targets)
        if self.dice_weight == 0:
            return self.bce_weight * self.bce_loss(outputs, targets)

        bce = self.bce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss

    def forward(self, outputs, targets):
        assert len(outputs) == len(self.depth_weights)
        losses = []
        weight_total = 0
        for idx, o in enumerate(outputs):
            weight = self.depth_weights[idx]
            if weight == 0:
                continue
            losses.append(self.forward_single(o, targets) * weight)
            weight_total += weight
        return {'loss': torch.stack(losses, dim=0).sum(dim=0) / weight_total}


class IoULoss(nn.Module):
    """
    Intersection over union (Jaccard) loss
    Args:
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'Sigmoid', 'Softmax2d']
    """

    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None
    ):
        super().__init__()
        self.metric_fn = partial(iou, eps=eps, threshold=threshold)

    def forward(self, outputs, targets):
        iou = self.metric_fn(outputs, targets)
        return 1 - iou


class SmoothBCEDiceIoULoss(nn.Module):
    def __init__(
            self,
            eps: float = 1e-7,
            smooth: float = 1e-6,
            bce_weight: float = 1,
            dice_weight: float = 1,
            iou_weight: float = 1
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

        self.bce_loss = SmoothBCELoss(eps=smooth)
        self.dice_loss = DiceLoss(eps=eps)
        self.iou_loss = IoULoss(eps=eps)

    def forward(self, outputs, targets):
        bce = self.bce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        iou = self.iou_loss(outputs, targets)

        total = self.bce_weight * bce + self.dice_weight * dice + self.iou_weight * iou
        return {'loss': total, 'bce': bce, 'dice': dice, 'iou': iou}


class BinaryFocalLoss(_Loss):
    def __init__(
        self,
        alpha=0.5,
        gamma=2,
        ignore_index=None,
        reduction="mean",
        reduced=False,
        threshold=0.5,
    ):
        """
        :param alpha:
        :param gamma:
        :param ignore_index:
        :param reduced:
        :param threshold:
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if reduced:
            self.focal_loss = partial(
                focal_loss_with_logits,
                alpha=None,
                gamma=gamma,
                threshold=threshold,
                reduction=reduction,
            )
        else:
            self.focal_loss = partial(
                focal_loss_with_logits, alpha=alpha, gamma=gamma, reduction=reduction
            )

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem.
        """
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = label_target != self.ignore_index
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

        loss = self.focal_loss(label_input, label_target)
        return loss


class FocalBCEDiceLoss(BCEDiceLoss):

    def __init__(
            self,
            alpha=0.5,
            gamma=2,
            ignore_index=None,
            reduction="mean",
            reduced=False,
            eps: float = 1e-7,
            threshold: float = None,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
    ):
        super().__init__(eps, threshold, bce_weight, dice_weight)
        self.bce_loss = BinaryFocalLoss(alpha, gamma, ignore_index, reduction, reduced, threshold)


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


def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid"
):
    """
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/utils/criterion/iou.py
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: IoU (Jaccard) score
    """
    outputs = F.sigmoid(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    iou = intersection / (union - intersection + eps)

    return iou


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


def focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma=2.0,
    alpha: float = 0.25,
    reduction="mean",
    normalized=False,
    threshold: float = None,
) -> torch.Tensor:
    """
    https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/functional.py
    Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References::
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction="none")
    pt = torch.exp(logpt)

    # compute the loss
    if threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / threshold).pow(gamma)
        focal_term[pt < threshold] = 1

    loss = -focal_term * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    if normalized:
        norm_factor = focal_term.sum()
        loss = loss / norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss
