def dice_mean(output, targets, threshold=0.5):
    d0 = dice_0(output, targets, threshold)
    d1 = dice_1(output, targets, threshold)
    d2 = dice_2(output, targets, threshold)
    d3 = dice_3(output, targets, threshold)
    return (d0 + d1 + d2 + d3) / 4


def dice_0(output, targets, threshold=0.5):
    return dice_c(0, output, targets, threshold)


def dice_1(output, targets, threshold=0.5):
    return dice_c(1, output, targets, threshold)


def dice_2(output, targets, threshold=0.5):
    return dice_c(2, output, targets, threshold)


def dice_3(output, targets, threshold=0.5):
    return dice_c(3, output, targets, threshold)


def dice_c(c, output, targets, threshold=0.5):
    B, C, H, W = targets.size()
    total = 0.
    for b in range(B):
        total += dice_single_channel(
            output[b, c, :, :],
            targets[b, c, :, :],
            threshold
        )
    return total / B


def dice_single_channel(probability, truth, threshold, eps=1e-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return dice


def accuracy_c0(output, targets, threshold=0.5):
    return accuracy(output, targets, 0)


def accuracy_c1(output, targets, threshold=0.5):
    return accuracy(output, targets, 1)


def accuracy_c2(output, targets, threshold=0.5):
    return accuracy(output, targets, 2)


def accuracy_c3(output, targets, threshold=0.5):
    return accuracy(output, targets, 3)


def accuracy(output, targets, class_, threshold=0.5):
    preds = (output[:, class_] > threshold).float()
    return (preds == targets[:, class_]).float().mean()
