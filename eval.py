import torch
import numpy as np


class Evaluator:
    def __init__(self, model, dataloader, device, num_classes):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes

    def evaluate(self):

        self.model.eval()
        total_correct = 0
        total_pixels = 0
        iou_per_class = np.zeros(self.num_classes)
        dice_per_class = np.zeros(self.num_classes)
        class_count = np.zeros(self.num_classes)

        with torch.no_grad():
            for images, masks in self.dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                correct = torch.eq(preds, masks).sum().item()
                total_correct += correct
                total_pixels += torch.numel(preds)
                for c in range(self.num_classes):
                    pred_class = preds == c
                    mask_class = masks == c

                    intersection = torch.logical_and(pred_class, mask_class).sum().item()
                    union = torch.logical_or(pred_class, mask_class).sum().item()
                    dice = (2 * intersection) / (pred_class.sum().item() + mask_class.sum().item() + 1e-6)

                    if union > 0:
                        iou_per_class[c] += intersection / union
                        dice_per_class[c] += dice
                        class_count[c] += 1
        iou_per_class = iou_per_class / (class_count + 1e-6)
        dice_per_class = dice_per_class / (class_count + 1e-6)

        mean_iou = np.mean(iou_per_class)
        mean_dice = np.mean(dice_per_class)
        accuracy = total_correct / total_pixels

        return {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'iou_per_class': iou_per_class,
            'dice_per_class': dice_per_class
        }

