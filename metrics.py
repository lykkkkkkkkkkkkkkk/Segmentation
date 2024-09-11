import torch


def pixel_accuracy(preds, targets):

    preds = torch.argmax(preds, dim=1)  # Get the class with max probability for each pixel
    correct = (preds == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()

def mean_iou(preds, targets, num_classes):

    preds = torch.argmax(preds, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    return torch.nanmean(torch.tensor(ious)).item()

def dice_coefficient(preds, targets):

    preds = torch.sigmoid(preds)  # For binary segmentation
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    dice = (2. * intersection) / (preds.sum() + targets.sum())
    return dice.item()

def precision_recall_f1(preds, targets):

    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    true_positive = (preds * targets).sum()
    predicted_positive = preds.sum()
    actual_positive = targets.sum()

    precision = true_positive / (predicted_positive + 1e-6)
    recall = true_positive / (actual_positive + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return precision.item(), recall.item(), f1.item()


def iou_per_class(preds, targets, num_classes):

    preds = torch.argmax(preds, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    return ious
