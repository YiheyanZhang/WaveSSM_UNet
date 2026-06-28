import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss (单通道Sigmoid版本)
    """
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, D, H, W] 或 [B, D, H, W] 预测概率 (已经过Sigmoid)
            target: [B, D, H, W] 标签
        """
        # 如果是5维，squeeze通道维度
        if pred.dim() == 5 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        target = target.float()

        # 计算Dice系数的分子和分母
        intersection = (pred * target).sum()
        dice_coefficient = (2. * intersection + self.epsilon) / (pred.sum() + target.sum() + self.epsilon)

        # 计算Dice Loss
        loss = 1 - dice_coefficient
        return loss
