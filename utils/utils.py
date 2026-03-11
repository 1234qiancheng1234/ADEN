import os
import numpy as np
from torch.optim import lr_scheduler


def get_metric(confusion_matrix):
    TP, FP, TN, FN = confusion_matrix
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total != 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0.0
    return accuracy, f1_score, iou, precision, recall

def get_confusion_matrix(predicted_labels, true_labels):
    true_binary = (true_labels >= 0.5).astype(np.uint8)
    pred_binary = (predicted_labels >= 0.5).astype(np.uint8)
    TP = np.sum((true_binary == 1) & (pred_binary == 1))
    FP = np.sum((true_binary == 0) & (pred_binary == 1))
    TN = np.sum((true_binary == 0) & (pred_binary == 0))
    FN = np.sum((true_binary == 1) & (pred_binary == 0))
    return [TP, FP, TN, FN]


def create_file(filename):
    if not os.path.exists(filename):
        return filename
    base, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_name = f"{base}_{counter}{ext}"
        if not os.path.exists(new_name):
            return new_name
        counter += 1

class PolyLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iter) ** self.power
                for base_lr in self.base_lrs]