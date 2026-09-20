def confusion_metrics(confusion):
    """Return dataset-level binary metrics from [TN, FP, FN, TP]."""
    tn, fp, fn, tp = (float(value) for value in confusion)
    eps = 1e-12
    bg_iou = tn / (tn + fp + fn + eps)
    fault_iou = tp / (tp + fp + fn + eps)
    bg_dice = 2 * tn / (2 * tn + fp + fn + eps)
    fault_dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        "miou": (bg_iou + fault_iou) / 2,
        "dice": (bg_dice + fault_dice) / 2,
        "fault_iou": fault_iou,
        "fault_dice": fault_dice,
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def mean_volume_metrics(confusions):
    """Average class metrics computed independently for each volume."""
    per_volume = [confusion_metrics(confusion) for confusion in confusions]
    keys = ("miou", "dice", "fault_iou", "fault_dice")
    return {
        key: sum(metrics[key] for metrics in per_volume) / len(per_volume)
        for key in keys
    }
