import os
import torch
from dataloader.dataloader import FaultDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, average_precision_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.dice_loss import DiceLoss


class PhysicsConsistencyLoss(nn.Module):
    """
    物理一致性损失 - 仅在正样本（断层）区域计算

    核心思想：断层处应该有明显的地震信号梯度变化
    - 预测为断层的位置，原始数据应该有梯度异常
    - 只在真实标签为断层的区域计算，避免与主损失冲突

    优点：
    - 训练时约束，推理时不需要
    - 仅约束正样本区域，不会与主损失冲突
    - 域迁移友好：学到的是物理规律而非数据统计特性
    """
    def __init__(self, lambda_grad=0.1):
        super().__init__()
        self.lambda_grad = lambda_grad

    def compute_gradient(self, x):
        """计算3D梯度幅值"""
        # x: [B, 1, D, H, W] 或 [B, D, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(1)

        # 沿三个方向计算梯度
        gx = x[:, :, :, :, 2:] - x[:, :, :, :, :-2]
        gy = x[:, :, :, 2:, :] - x[:, :, :, :-2, :]
        gz = x[:, :, 2:, :, :] - x[:, :, :-2, :, :]

        # 对齐尺寸到最小公共尺寸
        min_d = min(gx.shape[2], gy.shape[2], gz.shape[2])
        min_h = min(gx.shape[3], gy.shape[3], gz.shape[3])
        min_w = min(gx.shape[4], gy.shape[4], gz.shape[4])

        gx = gx[:, :, :min_d, :min_h, :min_w]
        gy = gy[:, :, :min_d, :min_h, :min_w]
        gz = gz[:, :, :min_d, :min_h, :min_w]

        # 梯度幅值
        grad_mag = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)
        return grad_mag

    def forward(self, pred, seismic_data, label):
        """
        Args:
            pred: [B, 1, D, H, W] 或 [B, D, H, W] 模型预测的断层概率
            seismic_data: [B, 1, D, H, W] 原始地震数据
            label: [B, D, H, W] 真实标签
        Returns:
            physics_loss: 物理一致性损失
        """
        # 确保维度正确
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        if label.dim() == 4:
            label = label.unsqueeze(1)

        # 计算地震数据的梯度（物理先验）
        seismic_grad = self.compute_gradient(seismic_data)

        # 对齐pred和label到梯度的尺寸
        pred_aligned = pred[:, :, 1:-1, 1:-1, 1:-1]
        label_aligned = label[:, :, 1:-1, 1:-1, 1:-1]

        # 确保尺寸完全匹配
        target_shape = seismic_grad.shape[2:]
        if pred_aligned.shape[2:] != target_shape:
            pred_aligned = F.interpolate(pred_aligned, size=target_shape, mode='trilinear', align_corners=False)
            label_aligned = F.interpolate(label_aligned.float(), size=target_shape, mode='nearest')

        # 只在正样本（真实断层）区域计算物理一致性
        mask = (label_aligned > 0.5).float()

        # 如果没有正样本，返回0
        if mask.sum() < 10:
            return torch.tensor(0.0, device=pred.device)

        # 归一化梯度到[0,1]
        grad_norm = seismic_grad / (seismic_grad.max() + 1e-8)

        # 物理一致性：在断层区域，预测应该与梯度正相关
        # 使用加权MSE：断层区域的预测值应该在梯度高的地方也高
        pred_masked = pred_aligned * mask
        grad_masked = grad_norm * mask

        # 相关性损失：1 - cosine_similarity
        pred_flat = pred_masked.flatten(1)
        grad_flat = grad_masked.flatten(1)

        # 避免除零
        pred_norm = pred_flat / (pred_flat.norm(dim=1, keepdim=True) + 1e-8)
        grad_norm_flat = grad_flat / (grad_flat.norm(dim=1, keepdim=True) + 1e-8)

        cosine_sim = (pred_norm * grad_norm_flat).sum(dim=1).mean()
        physics_loss = 1 - cosine_sim

        return self.lambda_grad * physics_loss


def save_3d_orthogonal_view(seismic, pred, gt, save_path, name, slice_indices=None):
    """
    绘制三正交切片的3D可视化图（类似3D temp.png的排布）

    Args:
        seismic: 3D地震数据 [D, H, W]
        pred: 3D预测结果 [D, H, W]
        gt: 3D GT标签 [D, H, W]
        save_path: 保存路径
        name: 文件名前缀
        slice_indices: 切片位置 (depth_idx, inline_idx, xline_idx)，None则使用默认位置
    """
    D, H, W = seismic.shape

    # 切片位置：底部、前墙、左墙
    if slice_indices is None:
        depth_idx = D - 1       # 底部水平切片（深层，显示在下方）
        inline_idx = 0          # 前墙（Y=0）
        xline_idx = 0           # 左墙（X=0）
    else:
        depth_idx, inline_idx, xline_idx = slice_indices

    def overlay_fault_on_slice(seis_slice, fault_slice, vmin=0.3, vmax=1.0):
        """将断层概率以红色叠加在灰度地震切片上"""
        seis_norm = (seis_slice - seis_slice.min()) / (seis_slice.max() - seis_slice.min() + 1e-8)
        rgb = np.stack([seis_norm, seis_norm, seis_norm], axis=-1)
        fault_norm = np.clip((fault_slice - vmin) / (vmax - vmin), 0, 1)
        alpha = fault_norm * 0.8
        rgb[:, :, 0] = rgb[:, :, 0] * (1 - alpha) + alpha
        rgb[:, :, 1] = rgb[:, :, 1] * (1 - alpha)
        rgb[:, :, 2] = rgb[:, :, 2] * (1 - alpha)
        return np.clip(rgb, 0, 1)

    def draw_3d_slices(ax, seismic, fault, depth_idx, inline_idx, xline_idx, title):
        """绘制三正交切片（底部+后墙+右墙）"""
        D, H, W = seismic.shape

        # ====== 底部水平切片 (Z=depth_idx) ======
        slice_bottom = overlay_fault_on_slice(seismic[depth_idx, :, :], fault[depth_idx, :, :])
        x = np.arange(W)
        y = np.arange(H)
        X_bottom, Y_bottom = np.meshgrid(x, y)
        Z_bottom = np.ones_like(X_bottom) * depth_idx
        ax.plot_surface(X_bottom, Y_bottom, Z_bottom, rstride=1, cstride=1,
                        facecolors=slice_bottom, shade=False, alpha=0.95)

        # ====== 后墙垂直切片 (Y=inline_idx) ======
        slice_back = overlay_fault_on_slice(seismic[:, inline_idx, :], fault[:, inline_idx, :])
        x = np.arange(W)
        z = np.arange(D)
        X_back, Z_back = np.meshgrid(x, z)
        Y_back = np.ones_like(X_back) * inline_idx
        ax.plot_surface(X_back, Y_back, Z_back, rstride=1, cstride=1,
                        facecolors=slice_back, shade=False, alpha=0.95)

        # ====== 左墙垂直切片 (X=xline_idx) ======
        slice_left = overlay_fault_on_slice(seismic[:, :, xline_idx], fault[:, :, xline_idx])
        y = np.arange(H)
        z = np.arange(D)
        Y_left, Z_left = np.meshgrid(y, z)
        X_left = np.ones_like(Y_left) * xline_idx
        ax.plot_surface(X_left, Y_left, Z_left, rstride=1, cstride=1,
                        facecolors=slice_left, shade=False, alpha=0.95)

        # 设置坐标轴标签
        ax.set_xlabel('Crossline', fontsize=9, labelpad=8)
        ax.set_ylabel('Inline', fontsize=9, labelpad=8)
        ax.set_zlabel('Depth', fontsize=9, labelpad=8)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        # 设置坐标范围
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_zlim(0, D)

        # 反转Z轴（深度向下，0在顶部，D在底部）
        ax.invert_zaxis()

        # 设置视角：从右后方上方俯视（底部切片在下方）
        ax.view_init(elev=25, azim=125)

        # 设置背景透明
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')

        # 设置网格线
        ax.grid(True, alpha=0.3)

    # 创建图形：有GT时1行2列，无GT时1列
    if gt is not None:
        fig = plt.figure(figsize=(14, 6))

        # 左图：预测结果
        ax1 = fig.add_subplot(121, projection='3d')
        draw_3d_slices(ax1, seismic, pred, depth_idx, inline_idx, xline_idx, 'Prediction')

        # 右图：GT
        ax2 = fig.add_subplot(122, projection='3d')
        draw_3d_slices(ax2, seismic, gt, depth_idx, inline_idx, xline_idx, 'Ground Truth')
    else:
        fig = plt.figure(figsize=(8, 6))

        # 只显示预测结果
        ax1 = fig.add_subplot(111, projection='3d')
        draw_3d_slices(ax1, seismic, pred, depth_idx, inline_idx, xline_idx, 'Prediction')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{name}_3d_view.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[3D View] Saved to {os.path.join(save_path, f'{name}_3d_view.png')}")


class WeightedFocalLoss(nn.Module):
    """
    加权Focal Loss - 同时解决类别不平衡和难易样本不平衡

    公式: L = -α_t * (1 - p_t)^γ * log(p_t)

    参数:
        alpha: 正样本权重，None表示动态计算
        gamma: 聚焦参数，越大越关注难样本，通常取2.0
        dynamic_alpha: 是否根据正负样本比例动态计算alpha
    """
    def __init__(self, alpha=None, gamma=1.5, dynamic_alpha=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dynamic_alpha = dynamic_alpha

    def forward(self, pred, target):
        """
        Args:
            pred: [B, D, H, W] 预测概率 (已经过Sigmoid，值域0-1)
            target: [B, D, H, W] 标签 (0或1)
        Returns:
            focal_loss: 标量损失值
        """
        # 计算BCE（不reduce）
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')

        # 计算p_t
        p_t = torch.where(target == 1, pred, 1 - pred)

        # 计算alpha_t
        if self.dynamic_alpha:
            # 动态计算：正样本越少，权重越大
            pos = target.sum()
            neg = (1 - target).sum()
            alpha = (neg / (pos + neg + 1e-8)).clamp(0.25, 0.95)
        else:
            alpha = self.alpha if self.alpha is not None else 0.75

        alpha_t = torch.where(target == 1, alpha, 1 - alpha)

        # Focal Loss: α_t * (1 - p_t)^γ * BCE
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss

        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """
    边界检测损失 - 改进版

    改进：使用Focal Loss处理边界的稀疏性问题
    """
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.alpha = alpha  # 正样本权重
        self.gamma = gamma  # focal参数
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, D, H, W] 边界预测logits
            target: [B, 1, D, H, W] 边界GT (0/1)
        """
        pred_prob = torch.sigmoid(pred)

        # Focal Loss（处理稀疏边界）
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        focal_loss = focal_loss.mean()

        # Dice Loss
        intersection = (pred_prob * target).sum()
        dice = (2. * intersection + 1e-5) / (pred_prob.sum() + target.sum() + 1e-5)
        dice_loss = 1 - dice

        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss


class FaultContinuityLoss(nn.Module):
    """
    断层连续性损失 - 减少孤立错检

    原理：断层是连续面状结构，孤立点通常是错检
    """
    def __init__(self):
        super().__init__()
        # 26邻域核
        kernel = torch.ones(1, 1, 3, 3, 3) / 26.0
        kernel[0, 0, 1, 1, 1] = 0
        self.register_buffer('neighbor_kernel', kernel)

    def forward(self, pred, target):
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        # 邻域密度
        neighbor = F.conv3d(pred, self.neighbor_kernel, padding=1)

        # 孤立错检惩罚：预测为断层但邻域空，且GT不是断层
        isolation = pred * (1 - neighbor) * (1 - target)

        # 断裂惩罚：预测为背景但邻域满，且GT是断层
        discontinuity = (1 - pred) * neighbor * target

        return isolation.mean() + 1.0 * discontinuity.mean()


class FaultSegLoss(nn.Module):
    """
    专为地震断层分割设计的组合损失函数 (单通道Sigmoid版本)

    组合：
    1. Focal Loss: 处理严重的类别不平衡问题（断层区域通常<5%）
    2. Dice Loss: 处理区域重叠，对小目标友好
    3. Boundary Loss: 增强断层边界检测能力
    4. Continuity Loss: 增强断层连续性

    参数：
        alpha: Focal Loss的类别权重
        gamma: Focal Loss的聚焦参数，增大可更关注难分样本
        dice_weight: Dice Loss权重
        focal_weight: Focal Loss权重
        boundary_weight: Boundary Loss权重
        continuity_weight: Continuity Loss权重
    """
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=0.4,
                 focal_weight=0.3, boundary_weight=0.15, continuity_weight=0.15):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.continuity_weight = continuity_weight
        self.dice_loss = DiceLoss()

    def focal_loss(self, pred_prob, labels):
        """
        Binary Focal Loss: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
        对易分样本降权，聚焦于难分样本

        Args:
            pred_prob: [B, D, H, W] 预测概率 (已经过Sigmoid)
            labels: [B, D, H, W] 标签
        """
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy(pred_prob, labels, reduction='none')

        # 计算pt
        pt = torch.where(labels == 1, pred_prob, 1 - pred_prob)

        # 动态计算alpha：根据实际正负样本比例
        pos = labels.sum()
        neg = (1 - labels).sum()
        # 正样本权重 = neg/(pos+neg)，这样正样本越少权重越大
        dynamic_alpha = neg / (pos + neg + 1e-8)
        dynamic_alpha = dynamic_alpha.clamp(0.5, 0.95)  # 限制范围，避免极端值
        alpha_t = torch.where(labels == 1, dynamic_alpha, 1 - dynamic_alpha)

        # Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()

    def boundary_loss(self, pred_prob, labels):
        """
        Boundary Loss: 使用Sobel算子提取边界，增强断层边界检测

        Args:
            pred_prob: [B, D, H, W] 预测概率
            labels: [B, D, H, W] 标签
        """
        # 添加通道维度 [B, 1, D, H, W]
        pred = pred_prob.unsqueeze(1)
        target = labels.unsqueeze(1).float()

        # 3D Sobel算子（简化版，沿各轴计算梯度）
        pred_boundary = self._compute_boundary(pred)
        target_boundary = self._compute_boundary(target)

        # 边界损失：预测边界与真实边界的L1距离
        boundary_loss = F.l1_loss(pred_boundary, target_boundary)

        return boundary_loss

    def _compute_boundary(self, x):
        """
        使用差分近似计算3D边界
        """
        # 沿D、H、W三个方向计算梯度
        grad_d = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        grad_h = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        grad_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])

        # 填充使尺寸一致
        grad_d = F.pad(grad_d, (0, 0, 0, 0, 0, 1))
        grad_h = F.pad(grad_h, (0, 0, 0, 1, 0, 0))
        grad_w = F.pad(grad_w, (0, 1, 0, 0, 0, 0))

        # 合并梯度
        boundary = (grad_d + grad_h + grad_w) / 3.0

        return boundary

    def continuity_loss(self, pred_prob, labels):
        """
        连续性损失：惩罚断层预测的不连续性
        使用梯度平滑约束，鼓励断层沿主方向连续

        Args:
            pred_prob: [B, D, H, W] 预测概率
            labels: [B, D, H, W] 标签
        """
        pred = pred_prob.unsqueeze(1)  # [B, 1, D, H, W]

        # 计算预测的二阶导数（曲率），惩罚过大的曲率变化
        grad_d = pred[:, :, 2:, :, :] - 2*pred[:, :, 1:-1, :, :] + pred[:, :, :-2, :, :]
        grad_h = pred[:, :, :, 2:, :] - 2*pred[:, :, :, 1:-1, :] + pred[:, :, :, :-2, :]
        grad_w = pred[:, :, :, :, 2:] - 2*pred[:, :, :, :, 1:-1] + pred[:, :, :, :, :-2]

        # 只在断层区域附近计算连续性损失
        target = labels.unsqueeze(1).float()
        mask_d = F.max_pool3d(target, kernel_size=(3,1,1), stride=1, padding=(1,0,0))[:,:,1:-1,:,:]
        mask_h = F.max_pool3d(target, kernel_size=(1,3,1), stride=1, padding=(0,1,0))[:,:,:,1:-1,:]
        mask_w = F.max_pool3d(target, kernel_size=(1,1,3), stride=1, padding=(0,0,1))[:,:,:,:,1:-1]

        loss = (torch.abs(grad_d) * mask_d).mean() + \
               (torch.abs(grad_h) * mask_h).mean() + \
               (torch.abs(grad_w) * mask_w).mean()
        return loss / 3.0

    def forward(self, pred_prob, labels):
        """
        Args:
            pred_prob: [B, 1, D, H, W] 或 [B, D, H, W] 预测概率
            labels: [B, D, H, W] 标签
        """
        # 确保维度正确
        if pred_prob.dim() == 5 and pred_prob.size(1) == 1:
            pred_prob = pred_prob.squeeze(1)

        # 计算各项损失
        loss_dice = self.dice_loss(pred_prob.unsqueeze(1), labels)
        loss_focal = self.focal_loss(pred_prob, labels)
        loss_boundary = self.boundary_loss(pred_prob, labels)
        loss_continuity = self.continuity_loss(pred_prob, labels)

        # 加权组合
        total_loss = (self.dice_weight * loss_dice +
                      self.focal_weight * loss_focal +
                      self.boundary_weight * loss_boundary +
                      self.continuity_weight * loss_continuity)

        return total_loss


def save_args_info(args):
    # save args to config.txt
    argsDict = args.__dict__
    result_path = './EXP/' + '/' + args.exp + '/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if args.mode == 'train':
        with open(result_path + 'config.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    elif args.mode == 'valid_only':
        with open(result_path + 'config_valid_only.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    elif args.mode == 'pred':
        with open(result_path + 'config_pred.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')


def _scan_pairs(root):
    """扫描 root/seis 与 root/fault，返回按文件名排序的 (img, label) 对。"""
    img_dir = os.path.join(root, 'seis')
    lab_dir = os.path.join(root, 'fault')
    pairs = []
    for name in sorted(os.listdir(img_dir)):
        pairs.append((os.path.join(img_dir, name), os.path.join(lab_dir, name)))
    return pairs


def load_data(args):
    # args.mode=['train', 'valid_only', 'pred']

    if args.mode == 'train':
        # ratio 模式：仅对 train_path 子采样, valid_path 保持完整不变
        # 这样不同 train_ratio 之间的验证指标在同一把尺子下可比
        if getattr(args, 'train_ratio', None) is not None:
            assert 0.0 < args.train_ratio <= 1.0, \
                f"--train_ratio must be in (0,1], got {args.train_ratio}"

            full_train = _scan_pairs(args.train_path)
            rng = np.random.RandomState(args.split_seed)
            order = rng.permutation(len(full_train))
            n_train = max(1, int(round(len(full_train) * args.train_ratio)))
            train_pairs = [full_train[i] for i in order[:n_train]]

            print(f"--- train subsample: ratio={args.train_ratio}, seed={args.split_seed} ---")
            print(f"    train_path total: {len(full_train)}  ->  using {len(train_pairs)} samples")
            print(f"    valid_path keeps full set (untouched for fair comparison)")

            train_dataset = FaultDataset(mode=args.mode, transform=None, augment=True,
                                         file_list=train_pairs)
            valid_dataset = FaultDataset(args.valid_path, args.mode, transform=None, augment=False)
        else:
            # legacy: 直接用两个独立目录
            train_dataset = FaultDataset(args.train_path, args.mode, transform=None, augment=True)
            valid_dataset = FaultDataset(args.valid_path, args.mode, transform=None, augment=False)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size_not_train, shuffle=False, num_workers=args.workers, drop_last=True)

        print("--- create train dataloader (augment=True) ---")
        print(len(train_dataset), ", train dataset created")
        print(len(train_dataloader), ", train dataloader created")

        print("--- create valid dataloader (augment=False) ---")
        print(len(valid_dataset), ", valid dataset created")
        print(len(valid_dataloader), ", valid dataloaders created")

        return train_dataloader, valid_dataloader

    elif args.mode == 'valid_only':
        dataset = FaultDataset(args.valid_path, args.mode, transform=None, augment=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_not_train, shuffle=False, num_workers=args.workers, drop_last=True)

        print("--- create valid dataloader ---")
        print(len(dataset), ", valid dataset created")
        print(len(dataloader), ", valid dataloaders created")

        return dataloader

    else:  # args.mode=='pred'
        dataset = FaultDataset(args.pred_path, args.mode, transform=None, augment=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_not_train, shuffle=False, num_workers=args.workers, drop_last=True)
        print("--- create prediction dataloader ---")
        print(len(dataset), ", prediction dataset created")
        print(len(dataloader), ", prediction dataloaders created")
        return dataloader


def compute_loss(outputs, labels, args, inputs=None):
    """
    计算损失函数 (单通道Sigmoid二分类)

    Args:
        outputs: 模型输出
            - tuple(main_out, boundary_logits): 主输出和边界logits
            - 或单独的 [B, 1, D, H, W] 概率图
        labels: [B, D, H, W] 分割标签
        args: 参数配置
        inputs: [B, 1, D, H, W] 原始地震数据（可选，用于物理一致性损失）
    """
    # 解析输出格式
    boundary_outputs = None

    if isinstance(outputs, tuple):
        # 格式: (main_out, boundary_logits) 或 (main_out, boundary_logits, physics_clue)
        main_outputs = outputs[0]
        if len(outputs) >= 2:
            boundary_outputs = outputs[1]
    else:
        main_outputs = outputs

    # 确保维度匹配: [B, 1, D, H, W] -> [B, D, H, W]
    if main_outputs.dim() == 5 and main_outputs.size(1) == 1:
        main_outputs = main_outputs.squeeze(1)

    # 内部函数：计算分割损失 (单通道BCE)
    def calc_seg_loss(pred, target):
        if args.loss_func == 'dice':
            criterion = DiceLoss().to(args.device)
            return criterion(pred.unsqueeze(1), target)

        elif args.loss_func == 'bce':
            # 简单BCE，与faultSeg原版一致
            return nn.BCELoss()(pred, target)

        elif args.loss_func == 'bce_with_weight':
            # 加权BCE，处理样本不平衡
            neg = (1 - target).sum()
            pos = target.sum()
            pos_weight = neg / (pos + 1e-8)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                torch.logit(pred.clamp(1e-7, 1-1e-7)), target)

        elif args.loss_func == 'cross_with_weight':
            # 兼容旧配置，转换为加权BCE
            neg = (1 - target).sum()
            pos = target.sum()
            pos_weight = neg / (pos + 1e-8)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                torch.logit(pred.clamp(1e-7, 1-1e-7)), target)

        elif args.loss_func == 'bce+dice':
            # BCE + Dice + Focal Loss 组合
            # BCE Loss（保留计算，权重为0）
            neg = (1 - target).sum()
            pos = target.sum()
            pos_weight = neg / (pos + 1e-8)
            loss_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                torch.logit(pred.clamp(1e-7, 1-1e-7)), target)

            # Dice Loss
            criterion = DiceLoss().to(args.device)
            loss_dice = criterion(pred.unsqueeze(1), target)

            # 加权Focal Loss（动态alpha + gamma=2.0）
            focal_criterion = WeightedFocalLoss(gamma=3, dynamic_alpha=True)
            loss_focal = focal_criterion(pred, target)

            # 权重：BCE=0.0, Dice=0.7, Focal=0.3
            return 0.0 * loss_bce + 0.7 * loss_dice + 0.3 * loss_focal

        elif args.loss_func == 'cross+dice':
            # 兼容旧配置
            neg = (1 - target).sum()
            pos = target.sum()
            pos_weight = neg / (pos + 1e-8)
            loss_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                torch.logit(pred.clamp(1e-7, 1-1e-7)), target)
            criterion = DiceLoss().to(args.device)
            loss_dice = criterion(pred.unsqueeze(1), target)
            return 0.3 * loss_bce + 0.7 * loss_dice

        elif args.loss_func == 'fault_seg':
            # 加权BCE：动态计算pos_weight，对稀少的断层像素加权
            neg = (1 - target).sum()
            pos = target.sum()
            pos_weight = neg / (pos + 1e-8)  # 关键！动态权重防止漏检
            loss_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                torch.logit(pred.clamp(1e-7, 1-1e-7)), target)

            # FaultSegLoss：处理边界和连续性
            criterion = FaultSegLoss(
                alpha=0.85,
                gamma=3.0,
                dice_weight=0.30,
                focal_weight=0.30,
                boundary_weight=0.15,
                continuity_weight=0.25
            ).to(args.device)
            loss_fault = criterion(pred.unsqueeze(1), target)

            # 组合：权重各一半
            return 0.15 * loss_bce + 0.85 * loss_fault
        else:
            raise ValueError("Only ['dice', 'bce', 'bce_with_weight', 'bce+dice', 'cross_with_weight', 'cross+dice', 'fault_seg'] loss is supported.")

    # 计算主损失
    total_loss = calc_seg_loss(main_outputs, labels)

    # 边界损失（如果有）
    if boundary_outputs is not None:
        from models.MSM_UNet import BADH
        boundary_target = BADH.compute_boundary_target(labels)
        boundary_criterion = BoundaryLoss().to(args.device)
        boundary_loss = boundary_criterion(boundary_outputs, boundary_target)
        boundary_weight = getattr(args, 'boundary_loss_weight', 0.3)
        total_loss = total_loss + 0.0 * boundary_loss

    return total_loss


def con_matrix(outputs, labels, args):
    """
    计算混淆矩阵并返回IoU和Dice (单通道Sigmoid二分类)

    Args:
        outputs: 模型输出
            - tuple(main_out, boundary_logits): 主输出和边界logits
            - 或单独的 [B, 1, D, H, W] 概率图
        labels: [B, D, H, W] 分割标签
        args: 参数配置
    """
    # 提取主输出
    if isinstance(outputs, tuple):
        # 格式: (main_out, boundary_logits)
        outputs = outputs[0]

    y_pred = outputs.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    # 单通道: [B, 1, D, H, W] -> [B, D, H, W] -> 二值化
    if y_pred.ndim == 5 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)
    y_pred = (y_pred > 0.5).astype(np.int32).flatten()
    y_true = y_true.flatten().astype(np.int32)

    num_class = 2  # 二分类
    current = confusion_matrix(y_true, y_pred, labels=range(num_class))

    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection + 1e-7
    IoU = intersection / union.astype(np.float32)
    union_dice = ground_truth_set + predicted_set + 1e-7
    DICE = 2 * intersection / union_dice.astype(np.float32)

    return np.mean(IoU), np.mean(DICE)


def save_train_info(args, train_RESULT, val_RESULT):
    if not os.path.exists('./EXP/' + args.exp + '/results/train/'):
        os.makedirs('./EXP/' + args.exp + '/results/train/')

    data_df = pd.DataFrame(train_RESULT)
    data_df.columns = ['train_loss', 'train_iou', 'train_dice']
    start = getattr(args, 'start_epoch', 0)
    data_df.index = np.arange(start, start + len(train_RESULT), 1)
    writer = pd.ExcelWriter('./EXP/' + args.exp + '/results/train/train_result.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer._save()
    writer.close()

    data_df_val = pd.DataFrame(val_RESULT)
    data_df_val.columns = ['val_loss', 'val_iou', 'val_dice']
    data_df_val.index = np.arange(start, start + len(val_RESULT), 1)
    writer_val = pd.ExcelWriter('./EXP/' + args.exp + '/results/train/val_result.xlsx')
    data_df_val.to_excel(writer_val, 'page_1', float_format='%.5f')
    writer_val._save()


def save_result(args, segs, inputs, gts, val_loss, val_iou, val_dice):
    result_path = './EXP/' + args.exp + '/results/valid/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + "valid_final_result.txt", 'a+') as f:
        f.write('valid loss:\t' + str(val_loss) + '\n')
        f.write('valid iou:\t' + str(val_iou) + '\n')
        f.write('valid dice:\t' + str(val_dice) + '\n')

    if not os.path.exists(result_path + '/numpy/'):
        os.makedirs(result_path + '/numpy/')
    if not os.path.exists(result_path + '/picture/'):
        os.makedirs(result_path + '/picture/')

    for i in range(len(inputs)):
        # 单通道Sigmoid输出：squeeze后保留概率值用于3D可视化
        seg_prob = np.squeeze(segs[i])  # 保留概率值
        seg = (seg_prob > 0.5).astype(np.float32)  # 阈值二值化
        img = np.squeeze(inputs[i])
        gt = np.squeeze(gts[i])
        # save output
        np.save(result_path + '/numpy/' + str(i) + '_seg.npy', seg)
        np.save(result_path + '/numpy/' + str(i) + '_img.npy', img)
        np.save(result_path + '/numpy/' + str(i) + '_gt.npy', gt)

        # ====== 保存3D正交切片可视化 ======
        save_3d_orthogonal_view(
            seismic=img,
            pred=seg_prob,  # 使用概率值显示
            gt=gt,
            save_path=result_path + '/picture/',
            name=f'No_{i}'
        )

        # save picture (原有的2D切片图)
        index = np.arange(0, 128, 50)
        for idx in index:
            # dim 0
            plt.subplot(1, 3, 1)
            plt.imshow(img[idx, :, :])
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(gt[idx, :, :])
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(seg[idx, :, :])
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_0.png')
            plt.close()
            # dim 1
            plt.subplot(1, 3, 1)
            plt.imshow(img[:, idx, :])
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(gt[:, idx, :])
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(seg[:, idx, :])
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_1.png')
            plt.close()
            # dim 2
            plt.subplot(1, 3, 1)
            plt.imshow(img[:, :, idx])
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(gt[:, :, idx])
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(seg[:, :, idx])
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_2.png')
            plt.close()


def load_pred_data(args, dim=(512, 384, 128)):
    if args.pred_data_name == 'f3':
        print("Data use f3.")
        seis = np.fromfile('data/prediction/f3d/gxl.dat', dtype=np.float32).reshape(dim)
        fault = np.fromfile('data/prediction/f3d/fpx.dat', dtype=np.float32).reshape(dim)
        return seis
    elif args.pred_data_name == 'kerry':
        print("Data use kerry.")
        import segyio
        kerry_path = 'data/prediction/Kerry3D.segy'
        with segyio.open(kerry_path, 'r', ignore_geometry=True) as f:
            data = np.stack([f.trace[i] for i in range(len(f.trace))], axis=0)
        # Kerry3D维度: (287, 735, 1252) -> (inline, xline, samples)
        data = data.reshape(287, 735, 1252)
        print(f"Kerry3D data shape: {data.shape}")
        return data
    elif args.pred_data_name == 'thebe':
        print("Data use Thebe.")
        thebe_dir = 'data/prediction/thebe_data'
        seis_path = os.path.join(thebe_dir, 'seis', 'seistest1.npz')
        seis_npz = np.load(seis_path)
        # npz 可能有多个 key，取第一个
        seis_key = list(seis_npz.keys())[0]
        seis = seis_npz[seis_key]
        print(f"Thebe seismic shape: {seis.shape}, key='{seis_key}'")
        return seis
    else:
        raise ValueError("Only ['f3', 'kerry', 'thebe'] mode is supported.")


def save_pred_picture(gx, gy, save_path, pred_data_name):
    # 根据数据集选择切片索引
    if pred_data_name == 'kerry':
        # Kerry3D维度: (287, 735, 1252)
        k1, k2, k3 = 143, 367, 626  # 中间切片
        has_gt = False
        gt = None
    else:
        # F3维度: (512, 384, 128)
        k1, k2, k3 = 99, 29, 29
        # 加载F3原始断层标签数据
        gt_path = 'data/prediction/f3d/fpx.dat'
        try:
            gt = np.fromfile(gt_path, dtype=np.float32).reshape(512, 384, 128)
            has_gt = True
        except:
            has_gt = False
            gt = None
            print(f"Warning: Ground truth file {gt_path} not found, skipping GT display")

    gx1 = gx[k1, :, :]
    gy1 = gy[k1, :, :]
    gx2 = gx[:, k2, :]
    gy2 = gy[:, k2, :]
    gx3 = gx[:, :, k3]
    gy3 = gy[:, :, k3]

    if has_gt:
        gt1 = gt[k1, :, :]
        gt2 = gt[:, k2, :]
        gt3 = gt[:, :, k3]

    # ====== 方向变换：与原始faultSeg输出对齐 ======
    # dim_0 (xline slice): 转置使其横向显示
    gx1 = gx1.T
    gy1 = gy1.T
    if has_gt:
        gt1 = gt1.T

    # dim_1 (inline slice): 转置使其横向显示
    gx2 = gx2.T
    gy2 = gy2.T
    if has_gt:
        gt2 = gt2.T

    # dim_2 (time slice): 转置+上下翻转
    gx3 = np.flipud(gx3.T)
    gy3 = np.flipud(gy3.T)
    if has_gt:
        gt3 = np.flipud(gt3.T)

    # 辅助函数：在地震图上叠加断层（红色表示断层概率）
    def overlay_fault_on_seismic(seismic, fault, vmin=0.4, vmax=1.0):
        """将断层概率以红色叠加在灰度地震图上"""
        # 归一化地震数据到[0,1]
        seis_norm = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
        # 创建RGB图像（灰度地震图）
        rgb = np.stack([seis_norm, seis_norm, seis_norm], axis=-1)
        # 归一化断层概率
        fault_norm = np.clip((fault - vmin) / (vmax - vmin), 0, 1)
        # 用红色叠加断层，概率越高红色越深
        alpha = fault_norm * 0.7  # 透明度
        rgb[:, :, 0] = rgb[:, :, 0] * (1 - alpha) + alpha  # 红色通道
        rgb[:, :, 1] = rgb[:, :, 1] * (1 - alpha)  # 绿色通道减弱
        rgb[:, :, 2] = rgb[:, :, 2] * (1 - alpha)  # 蓝色通道减弱
        return np.clip(rgb, 0, 1)

    # 概率图可视化参数
    vmin, vmax = 0.4, 1.0

    # 创建3行2列的合并图（三个维度合并为一张图）
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # 准备数据列表
    seismic_list = [gx1, gx2, gx3]
    pred_list = [gy1, gy2, gy3]
    gt_list = [gt1, gt2, gt3] if has_gt else [None, None, None]
    dim_names = ['Xline (dim_0)', 'Inline (dim_1)', 'Time (dim_2)']

    for row, (seis, pred, gt_slice, dim_name) in enumerate(zip(seismic_list, pred_list, gt_list, dim_names)):
        # 左列：地震图 + 预测断层
        overlay_pred = overlay_fault_on_seismic(seis, pred, vmin, vmax)
        axes[row, 0].imshow(overlay_pred)
        axes[row, 0].set_title(f'{dim_name}: Prediction', fontsize=9)
        axes[row, 0].axis('on')

        # 右列：地震图 + 真实断层
        if has_gt and gt_slice is not None:
            overlay_gt = overlay_fault_on_seismic(seis, gt_slice, vmin, vmax)
            axes[row, 1].imshow(overlay_gt)
            axes[row, 1].set_title(f'{dim_name}: Ground Truth', fontsize=9)
        else:
            axes[row, 1].imshow(seis, cmap='gray')
            axes[row, 1].set_title(f'{dim_name}: Seismic', fontsize=9)
        axes[row, 1].axis('on')

    # 紧凑布局：减少子图间距（hspace控制上下间距）
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.02)
    plt.savefig(save_path + pred_data_name + '_combined.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # ====== 保存3D正交切片可视化 ======
    # 无论是否有GT都保存3D可视化
    save_3d_orthogonal_view(
        seismic=gx,
        pred=gy,
        gt=gt if has_gt else None,
        save_path=save_path,
        name=pred_data_name,
        slice_indices=(k1, k2, k3)
    )

    # ====== F3定量指标计算 ======
    if has_gt and pred_data_name == 'f3':
        print("\n" + "=" * 60)
        print("F3 定量评估 (参考标准: FaultSeg预测)")
        print("=" * 60)

        # 展平数据
        pred_flat = gy.flatten()
        gt_flat = gt.flatten()
        gt_binary = (gt_flat > 0.5).astype(np.float32)

        # 采样计算曲线指标（加速）
        n_samples = len(pred_flat)
        if n_samples > 2000000:
            sample_idx = np.random.choice(n_samples, 2000000, replace=False)
            pred_sample = pred_flat[sample_idx]
            gt_sample = gt_binary[sample_idx]
        else:
            pred_sample = pred_flat
            gt_sample = gt_binary

        # 计算多个阈值下的指标
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        print(f"\n{'阈值':<8} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-" * 55)

        best_f1 = 0
        best_th = 0.5
        for th in thresholds:
            pred_binary = (pred_flat > th).astype(np.float32)
            intersection = np.sum(pred_binary * gt_binary)
            union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
            iou = intersection / (union + 1e-8)
            dice = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)

            tp = np.sum(pred_binary * gt_binary)
            fp = np.sum(pred_binary * (1 - gt_binary))
            fn = np.sum((1 - pred_binary) * gt_binary)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            print(f"{th:<8.1f} {iou:>8.4f} {dice:>8.4f} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        # PR-AUC和ROC-AUC
        pr_auc = average_precision_score(gt_sample, pred_sample)
        fpr, tpr, _ = roc_curve(gt_sample, pred_sample)
        roc_auc = auc(fpr, tpr)

        # 计算最佳F1（从PR曲线）
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(gt_sample, pred_sample)
        f1_curve = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
        optimal_f1 = np.max(f1_curve)
        optimal_idx = np.argmax(f1_curve)
        optimal_th = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 0.5

        print("-" * 55)
        print(f"\nPR-AUC (AP):     {pr_auc:.4f}")
        print(f"ROC-AUC:         {roc_auc:.4f}")
        print(f"最佳F1:          {optimal_f1:.4f} (阈值={optimal_th:.4f})")

        # 保存指标到文件
        metrics_path = save_path + 'f3_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("F3 定量评估结果\n")
            f.write("参考标准: FaultSeg预测 (fpx.dat)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"PR-AUC (AP):     {pr_auc:.6f}\n")
            f.write(f"ROC-AUC:         {roc_auc:.6f}\n")
            f.write(f"最佳F1:          {optimal_f1:.6f}\n")
            f.write(f"最佳F1阈值:      {optimal_th:.4f}\n\n")
            f.write("各阈值下指标:\n")
            f.write(f"{'阈值':<8} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}\n")
            for th in thresholds:
                pred_binary = (pred_flat > th).astype(np.float32)
                intersection = np.sum(pred_binary * gt_binary)
                union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
                iou = intersection / (union + 1e-8)
                dice = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)
                tp = np.sum(pred_binary * gt_binary)
                fp = np.sum(pred_binary * (1 - gt_binary))
                fn = np.sum((1 - pred_binary) * gt_binary)
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f.write(f"{th:<8.1f} {iou:>8.4f} {dice:>8.4f} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f}\n")

        print(f"\n指标已保存: {metrics_path}")
        print("=" * 60)
