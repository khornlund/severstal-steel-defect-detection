import torch.nn.functional as F


def bce_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)
