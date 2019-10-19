import torch


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


def accuracy_0(output, targets, threshold=0.5):
    return accuracy(output, targets, 0)


def accuracy_1(output, targets, threshold=0.5):
    return accuracy(output, targets, 1)


def accuracy_2(output, targets, threshold=0.5):
    return accuracy(output, targets, 2)


def accuracy_3(output, targets, threshold=0.5):
    return accuracy(output, targets, 3)


def accuracy(output, targets, class_, threshold=0.5):
    preds = (output[:, class_] > threshold).float()
    return (preds == targets[:, class_]).float().mean()


def precision_0(output, targets, threshold=0.5):
    return precision(output, targets, 0)


def precision_1(output, targets, threshold=0.5):
    return precision(output, targets, 1)


def precision_2(output, targets, threshold=0.5):
    return precision(output, targets, 2)


def precision_3(output, targets, threshold=0.5):
    return precision(output, targets, 3)


def precision(output, targets, class_, threshold):
    preds = (output[:, class_] > threshold).float()
    tp, fp, tn, fn = _confusion(preds, targets)
    return tp / (tp + fp)


def recall_0(output, targets, threshold=0.5):
    return recall(output, targets, 0)


def recall_1(output, targets, threshold=0.5):
    return recall(output, targets, 1)


def recall_2(output, targets, threshold=0.5):
    return recall(output, targets, 2)


def recall_3(output, targets, threshold=0.5):
    return recall(output, targets, 3)


def recall(output, targets, class_, threshold):
    preds = (output[:, class_] > threshold).float()
    tp, fp, tn, fn = _confusion(preds, targets)
    return tp / (tp + fn)


def _confusion(prediction, truth):
    """
    https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
    Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives
