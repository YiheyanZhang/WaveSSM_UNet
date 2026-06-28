import os
import math
import copy
import torch
from torch.nn import DataParallel
from tqdm import tqdm
from utils.tools import load_data, compute_loss, con_matrix, save_train_info, save_result
import torch.optim as optim
from models.faultseg3d import FaultSeg3D
from models.MSM_UNet import MSM_UNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


# ============================== 训练辅助类 ==============================
class ModelEMA:
    """
    模型权重的指数滑动平均（EMA）。
    在每个 optimizer.step() 之后调用 update()，
    验证 / 保存 best checkpoint 时使用 ema.module 的权重。
    """
    def __init__(self, model, decay=0.999):
        # 反包装 DataParallel
        base = model.module if hasattr(model, 'module') else model
        self.module = copy.deepcopy(base).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        base = model.module if hasattr(model, 'module') else model
        msd = base.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


class WarmupCosineLR:
    """
    线性 warmup + 余弦退火，按 epoch 调度。
    第 0..warmup_epochs-1 epoch 线性升到 base_lr；
    之后从 base_lr 余弦退火到 min_lr，直到 total_epochs。
    """
    def __init__(self, optimizer, base_lr, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = max(0, warmup_epochs)
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def get_lr(self, epoch):
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            # 线性 warmup: lr from base_lr/warmup_epochs -> base_lr
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        # 余弦退火
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self, epoch):
        lr = self.get_lr(epoch)
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr


def plot_pr_roc_curves(y_true, y_pred, save_dir):
    """
    绘制PR曲线和ROC曲线，并标注AUC值

    Args:
        y_true: 真实标签（展平后的一维数组）
        y_pred: 预测概率（展平后的一维数组）
        save_dir: 保存目录
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 采样以减少计算量（如果数据量太大）
    n_samples = len(y_true)
    if n_samples > 1000000:  # 超过100万点则采样
        sample_idx = np.random.choice(n_samples, 1000000, replace=False)
        y_true_sample = y_true[sample_idx]
        y_pred_sample = y_pred[sample_idx]
        print(f"[PR/ROC] Sampled {1000000} points from {n_samples} for curve computation")
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred

    # ==================== PR曲线 ====================
    precision, recall, pr_thresholds = precision_recall_curve(y_true_sample, y_pred_sample)
    pr_auc = average_precision_score(y_true_sample, y_pred_sample)

    # 计算最佳F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1 = np.max(f1_scores)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5

    _font = {'family': 'serif', 'serif': ['Times New Roman', 'DejaVu Serif']}
    plt.figure(figsize=(8, 6))
    plt.rc('font', **_font)
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP = {pr_auc:.4f}, F1 = {best_f1:.4f})')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlabel('Recall', fontsize=9)
    plt.ylabel('Precision', fontsize=9)
    plt.legend(loc='lower left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'PR_curve.png'), dpi=300)
    plt.close()
    print(f"[PR Curve] Saved to {os.path.join(save_dir, 'PR_curve.png')}, AP = {pr_auc:.4f}, F1 = {best_f1:.4f}")

    # ==================== ROC曲线 ====================
    fpr, tpr, _ = roc_curve(y_true_sample, y_pred_sample)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.rc('font', **_font)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
    plt.xlabel('False Positive Rate', fontsize=9)
    plt.ylabel('True Positive Rate', fontsize=9)
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ROC_curve.png'), dpi=300)
    plt.close()
    print(f"[ROC Curve] Saved to {os.path.join(save_dir, 'ROC_curve.png')}, AUC = {roc_auc:.4f}")

    # ==================== 合并图 ====================
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.rc('font', **_font)

    # PR曲线
    axes[0].plot(recall, precision, 'b-', linewidth=2, label=f'AP = {pr_auc:.4f}, F1 = {best_f1:.4f}')
    axes[0].fill_between(recall, precision, alpha=0.2, color='blue')
    axes[0].set_xlabel('Recall', fontsize=9)
    axes[0].set_ylabel('Precision', fontsize=9)
    axes[0].legend(loc='lower left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].tick_params(labelsize=8)

    # ROC曲线
    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1].fill_between(fpr, tpr, alpha=0.2, color='blue')
    axes[1].set_xlabel('False Positive Rate', fontsize=9)
    axes[1].set_ylabel('True Positive Rate', fontsize=9)
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'PR_ROC_combined.png'), dpi=300)
    plt.close()
    print(f"[Combined] Saved to {os.path.join(save_dir, 'PR_ROC_combined.png')}")

    # 保存数值结果
    with open(os.path.join(save_dir, 'curve_metrics.txt'), 'w') as f:
        f.write(f"PR Curve - Average Precision (AP): {pr_auc:.6f}\n")
        f.write(f"PR Curve - Best F1 Score: {best_f1:.6f}\n")
        f.write(f"PR Curve - Best F1 Threshold: {best_f1_threshold:.4f}\n")
        f.write(f"ROC Curve - Area Under Curve (AUC): {roc_auc:.6f}\n")

    return pr_auc, roc_auc, best_f1


def train(args):
    # set device
    device = torch.device(args.device)
    print("---")
    print('Device is :', device)
    # Load data
    print("---")
    print("Loading data ... ")
    train_loader, val_loader = load_data(args)
    print('Create model...')
    
    # 根据model_type选择模型
    if args.model_type == 'msm_unet':
        model = MSM_UNet(args.in_channels, args.out_channels).to(device)
    elif args.model_type == 'FAULTSEG3D':
        # FaultSeg3D 内部已经带 Sigmoid，直接实例化，不走 compare_models/build.py
        model = FaultSeg3D(args.in_channels, args.out_channels).to(device)
        print(f"Using model: FaultSeg3D")
    else:
        # 使用compare_models中的对比模型
        from compare_models.build import build_model

        # 构造配置对象
        class ModelConfig:
            def __init__(self):
                self.model = type('obj', (object,), {
                    'name': args.model_type,
                    'in_chans': args.in_channels,
                    'num_classes': args.out_channels,
                    'drop': 0.0,
                    'attn_drop': 0.0,
                    'drop_path': 0.0
                })
                self.data = type('obj', (object,), {
                    'img_size': 128
                })

        config = ModelConfig()
        model = build_model(config).to(device)
        print(f"Using model: {args.model_type}")
    
    # model = FaultSeg3D(args.in_channels, args.out_channels).to(device)

    # DP 包装 (多卡并行)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    # Initialize optimizer
    print("---")
    print("Define optimizer ... ")

    # 选择优化器：默认 AdamW（带 weight_decay）
    opt_name = getattr(args, 'optimizer', 'adamw').lower()
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.optim_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=getattr(args, 'weight_decay', 1e-4),
        )
        print(f"  Optimizer: AdamW(lr={args.optim_lr}, wd={getattr(args, 'weight_decay', 1e-4)})")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.optim_lr)
        print(f"  Optimizer: Adam(lr={args.optim_lr})")

    # 学习率调度器
    sch_name = getattr(args, 'lr_scheduler', 'warmup_cosine').lower()
    if sch_name == 'warmup_cosine':
        scheduler = WarmupCosineLR(
            optimizer,
            base_lr=args.optim_lr,
            warmup_epochs=getattr(args, 'warmup_epochs', 5),
            total_epochs=args.epochs,
            min_lr=getattr(args, 'min_lr', 1e-6),
        )
        print(f"  Scheduler: WarmupCosine(warmup={getattr(args, 'warmup_epochs', 5)}, "
              f"total={args.epochs}, min_lr={getattr(args, 'min_lr', 1e-6)})")
    elif sch_name == 'cosine':
        scheduler = WarmupCosineLR(
            optimizer,
            base_lr=args.optim_lr,
            warmup_epochs=0,
            total_epochs=args.epochs,
            min_lr=getattr(args, 'min_lr', 1e-6),
        )
        print(f"  Scheduler: Cosine(total={args.epochs}, min_lr={getattr(args, 'min_lr', 1e-6)})")
    else:
        scheduler = None
        print("  Scheduler: None (constant lr)")

    # 梯度累积
    grad_accum_steps = max(1, int(getattr(args, 'grad_accum_steps', 1)))
    eff_bs = args.batch_size * grad_accum_steps
    print(f"  Grad accumulation: {grad_accum_steps} (effective batch size = {eff_bs})")

    # 早停机制参数
    early_stop_patience = int(getattr(args, 'early_stop_patience', 50))
    early_stop_counter = 0
    best_metric_name = getattr(args, 'best_metric', 'iou_dice')
    print(f"  Early stop patience: {early_stop_patience}")
    print(f"  Best metric        : {best_metric_name}")

    # Set model save path   ./EXP/<exp>/models/
    model_path = './EXP/' + args.exp + '/models/'

    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        state_dict = torch.load(args.resume, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        # 兼容 DataParallel 包装：base 是真实模型
        base_model = model.module if hasattr(model, 'module') else model
        # 修复可变形卷积权重 shape 不匹配（checkpoint [C,C,27] vs model [C,C,3,3,3]）
        model_sd = base_model.state_dict()
        for k in new_state_dict:
            if k in model_sd and new_state_dict[k].shape != model_sd[k].shape:
                if new_state_dict[k].numel() == model_sd[k].numel():
                    new_state_dict[k] = new_state_dict[k].reshape(model_sd[k].shape)
        base_model.load_state_dict(new_state_dict)
        start_epoch = args.start_epoch
        print(f"Loaded checkpoint, resuming from epoch {start_epoch + 1}/{args.epochs}")
    print("---")
    print("The model is saved in : ", model_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # EMA（必须在 resume 加载权重之后初始化，否则 EMA 拷贝的是随机权重）
    use_ema = getattr(args, 'use_ema', True)
    ema = ModelEMA(model, decay=getattr(args, 'ema_decay', 0.999)) if use_ema else None
    print(f"  EMA: {'on (decay=' + str(getattr(args, 'ema_decay', 0.999)) + ')' if use_ema else 'off'}")

    # 初始化训练状态
    train_RESULT = []
    val_RESULT = []
    best_metric_value = 0.0
    current_lr = args.optim_lr

    # start training
    print("---")
    print("Start training ... ")

    # 打印模型结构信息
    print("=== 模型结构信息 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # 按模块统计参数量
    print("  各模块参数量:")
    module_params = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in module_params:
            module_params[module_name] = 0
        module_params[module_name] += param.numel()
    for module_name, params in sorted(module_params.items(), key=lambda x: -x[1]):
        print(f"    {module_name}: {params:,} ({params/1e6:.2f}M)")

    def compute_best_metric(iou_val, dice_val):
        if best_metric_name == 'iou':
            return iou_val
        elif best_metric_name == 'dice':
            return dice_val
        else:  # 'iou_dice'
            return 0.5 * iou_val + 0.5 * dice_val

    for epoch in range(start_epoch, args.epochs):
        # 每个 epoch 开始时按调度器更新 lr
        if scheduler is not None:
            current_lr = scheduler.step(epoch)
        else:
            current_lr = args.optim_lr

        model.train()
        # 训练模式
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0

        optimizer.zero_grad(set_to_none=True)

        num_batches = len(train_loader)
        for step, data in enumerate(tqdm(train_loader, desc='[Train] Epoch' + str(epoch + 1) + '/' + str(args.epochs))):
            inputs, labels = data['x'].to(device), data['y'].to(device)

            outputs = model(inputs)
            loss = compute_loss(outputs, labels, args)
            # 梯度累积：缩放 loss
            loss_scaled = loss / grad_accum_steps

            loss_scaled.backward()

            # 累积满了再 step
            is_accum_step = ((step + 1) % grad_accum_steps == 0) or (step + 1 == num_batches)
            if is_accum_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # EMA 在 optimizer.step() 后更新
                if ema is not None:
                    ema.update(model)

            # 记录指标（用真实 loss，不是 scaled）
            with torch.no_grad():
                iou, dice = con_matrix(outputs, labels, args)
            train_loss += loss.item()
            train_iou += iou
            train_dice += dice

        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        # 验证：如启用 EMA，用 ema.module 来做验证 / 选 best
        eval_model = ema.module if ema is not None else model
        eval_model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for step, data in enumerate(tqdm(val_loader, desc='[VALID] Valid ')):
                inputs = data['x'].to(device)
                labels = data['y'].to(device)
                outputs = eval_model(inputs)
                loss = compute_loss(outputs, labels, args)
                iou, dice = con_matrix(outputs, labels, args)

                val_loss += loss.item()
                val_iou += iou
                val_dice += dice

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(
            " train loss: {:.4f}".format(avg_train_loss),
            " train iou: {:.4f}".format(avg_train_iou),
            " train dice:{:.4f}".format(avg_train_dice),
            " val loss: {:.4f}".format(avg_val_loss),
            " val iou: {:.4f}".format(avg_val_iou),
            " val dice:{:.4f}".format(avg_val_dice),
            " lr: {:.2e}".format(current_lr)
        )

        train_result = np.append(avg_train_loss, [avg_train_iou, avg_train_dice])
        train_RESULT.append(train_result)

        val_result = np.append(avg_val_loss, [avg_val_iou, avg_val_dice])
        val_RESULT.append(val_result)

        current_metric = compute_best_metric(avg_val_iou, avg_val_dice)
        if current_metric > best_metric_value:
            print("new best [{}] ({:.6f} --> {:.6f}) | iou={:.4f} dice={:.4f}".format(
                best_metric_name, best_metric_value, current_metric, avg_val_iou, avg_val_dice))
            best_metric_value = current_metric
            best_model_name = 'MSM_UNet_BEST.pth'
            # 保存时去掉 module. 前缀，兼容单卡加载；保存 EMA 权重（若启用）
            model_to_save = (ema.module if ema is not None
                             else (model.module if hasattr(model, 'module') else model))
            torch.save(model_to_save.state_dict(), model_path + best_model_name)
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered! No improvement for {early_stop_patience} epochs.")
                break

        if (epoch + 1) % args.val_every == 0:
            model_name = 'MSM_UNet_epoch_{}_iou_{:.4f}_CP.pth'.format(epoch + 1, avg_val_iou)
            model_to_save = (ema.module if ema is not None
                             else (model.module if hasattr(model, 'module') else model))
            torch.save(model_to_save.state_dict(), model_path + model_name)

    # Save training information

    print("---")
    print("Save training information ... ")
    save_train_info(args, train_RESULT, val_RESULT)
    print("---")
    print("Train Finish ! ")
    print("---")
    print("---")
    print("Last validation ... ")
    valid(args, val_loader)

    return 0


def valid(args, val_loader=None):

    device = torch.device(args.device)
    print("---")
    print('Device is :', device)
    # Load data
    print("---")
    print("Loading data ... ")
    if args.mode == 'valid_only':
        val_loader = load_data(args)
    # Load Model
    print("---")
    print("Loading Model ... ")
    if args.model_type == 'msm_unet':
        model = MSM_UNet(args.in_channels, args.out_channels).to(device)
    elif args.model_type == 'FAULTSEG3D':
        # FaultSeg3D 内部已带 Sigmoid，直接实例化
        model = FaultSeg3D(args.in_channels, args.out_channels).to(device)
    else:
        from compare_models.build import build_model
        class ModelConfig:
            def __init__(self):
                self.model = type('obj', (object,), {
                    'name': args.model_type,
                    'in_chans': args.in_channels,
                    'num_classes': args.out_channels,
                    'drop': 0.0, 'attn_drop': 0.0, 'drop_path': 0.0
                })
                self.data = type('obj', (object,), {'img_size': 128})
        config = ModelConfig()
        model = build_model(config).to(device)

    model_path = './EXP/' + args.exp + '/models/' + args.pretrained_model_name

    # 加载权重，处理module.前缀（兼容之前用DataParallel训练的模型）
    state_dict = torch.load(model_path, map_location=device)
    # 移除module.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    # ==================== 加载ResUNet和SwinUNETR模型 ====================
    from compare_models.build import build_model
    
    # ResUNet模型
    resunet_config = type('obj', (object,), {
        'model': type('obj', (object,), {
            'name': 'RESUNET',
            'in_chans': args.in_channels,
            'num_classes': args.out_channels,
            'drop': 0.0, 'attn_drop': 0.0, 'drop_path': 0.0
        }),
        'data': type('obj', (object,), {'img_size': 128})
    })()
    resunet_model = build_model(resunet_config).to(device)
    resunet_model_path = './EXP/resunet/models/MSM_UNet_BEST.pth'
    if os.path.exists(resunet_model_path):
        resunet_model.load_state_dict(torch.load(resunet_model_path, map_location=device))
        resunet_model.eval()
        print(f"  Loaded ResUNet model from {resunet_model_path}")
    else:
        print(f"  ResUNet model not found: {resunet_model_path}")
        resunet_model = None
    
    # SwinUNETR模型
    swinunetr_config = type('obj', (object,), {
        'model': type('obj', (object,), {
            'name': 'SWINUNETR',
            'in_chans': args.in_channels,
            'num_classes': args.out_channels,
            'drop': 0.0, 'attn_drop': 0.0, 'drop_path': 0.0
        }),
        'data': type('obj', (object,), {'img_size': 128})
    })()
    swinunetr_model = build_model(swinunetr_config).to(device)
    swinunetr_model_path = './EXP/swinunetr/models/MSM_UNet_BEST.pth'
    if os.path.exists(swinunetr_model_path):
        swinunetr_model.load_state_dict(torch.load(swinunetr_model_path, map_location=device))
        swinunetr_model.eval()
        print(f"  Loaded SwinUNETR model from {swinunetr_model_path}")
    else:
        print(f"  SwinUNETR model not found: {swinunetr_model_path}")
        swinunetr_model = None

    segs = []
    inputs = []
    gts = []
    resunet_segs = []
    swinunetr_segs = []

    print("---")
    print("Start validation ... ")

    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(tqdm(val_loader, desc='[Valid] Valid')):
            x = data['x'].to(args.device)
            y = data['y'].to(args.device)

            outputs = model(x)
            loss = compute_loss(outputs, y, args)
            iou, dice = con_matrix(outputs, y, args)

            val_loss += loss.item()
            val_iou += iou
            val_dice += dice

            # 处理增强模式的tuple输出
            seg_output = outputs[0] if isinstance(outputs, tuple) else outputs
            segs.append(seg_output.detach().cpu().numpy())
            inputs.append(x.detach().cpu().numpy())
            gts.append(y.detach().cpu().numpy())
            
            # ResUNet推理
            if resunet_model is not None:
                resunet_output = resunet_model(x)
                resunet_seg = resunet_output[0] if isinstance(resunet_output, tuple) else resunet_output
                resunet_segs.append(resunet_seg.detach().cpu().numpy())
            
            # SwinUNETR推理
            if swinunetr_model is not None:
                swinunetr_output = swinunetr_model(x)
                swinunetr_seg = swinunetr_output[0] if isinstance(swinunetr_output, tuple) else swinunetr_output
                swinunetr_segs.append(swinunetr_seg.detach().cpu().numpy())

        print(
            " val loss: {:.4f}".format(val_loss / len(val_loader)),
            " val iou: {:.4f}".format(val_iou / len(val_loader)),
            " val dice:{:.4f}".format(val_dice / len(val_loader)),
        )

        print("---")
        print("Save result of validation ... ")

        save_result(args, segs, inputs, gts, val_loss / len(val_loader), val_iou / len(val_loader), val_dice / len(val_loader))

        # ==================== 绘制PR曲线和ROC曲线 ====================
        print("---")
        print("Plotting PR and ROC curves ... ")

        # 将所有预测和标签展平
        all_preds = np.concatenate([s.flatten() for s in segs])
        all_gts = np.concatenate([g.flatten() for g in gts])
        
        # 展平ResUNet和SwinUNETR预测
        resunet_preds_all = np.concatenate([s.flatten() for s in resunet_segs]) if resunet_segs else None
        swinunetr_preds_all = np.concatenate([s.flatten() for s in swinunetr_segs]) if swinunetr_segs else None

        # 保存目录
        curve_save_dir = './EXP/' + args.exp + '/results/valid/'

        # 绘制单模型曲线
        pr_auc, roc_auc, best_f1 = plot_pr_roc_curves(all_gts, all_preds, curve_save_dir)
        print(f"[Metrics] PR-AUC (AP): {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}, Best F1: {best_f1:.4f}")

        # ==================== 生成对比曲线（论文用）====================
        print("---")
        print("Generating comparison PR/ROC curves (for paper)...")

        # 加载FaultSeg3D模型进行推理（获取连续概率值，使曲线平滑）
        faultseg_model_path = "./EXP/faultseg100/models/MSM_UNet_BEST.pth"
        faultseg_available = os.path.exists(faultseg_model_path)

        if faultseg_available:
            print(f"  Loading FaultSeg3D model from {faultseg_model_path}...")
            faultseg_model = FaultSeg3D(args.in_channels, args.out_channels).to(device)

            # 加载权重，处理module.前缀
            fs_state_dict = torch.load(faultseg_model_path, map_location=device)
            fs_new_state_dict = {}
            for k, v in fs_state_dict.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else k
                fs_new_state_dict[new_key] = v
            faultseg_model.load_state_dict(fs_new_state_dict)
            faultseg_model.eval()

            # 对验证集进行FaultSeg3D推理
            print("  Running FaultSeg3D inference on validation set...")
            faultseg_preds = []
            with torch.no_grad():
                for step, data in enumerate(tqdm(val_loader, desc='[FaultSeg3D] Inference')):
                    x = data['x'].to(device)
                    fs_output = faultseg_model(x)
                    # FaultSeg3D输出可能是tuple或tensor
                    fs_seg = fs_output[0] if isinstance(fs_output, tuple) else fs_output
                    faultseg_preds.append(fs_seg.detach().cpu().numpy())

            faultseg_preds_all = np.concatenate([p.flatten() for p in faultseg_preds])
            print(f"  FaultSeg3D predictions: min={faultseg_preds_all.min():.4f}, max={faultseg_preds_all.max():.4f}, unique values={len(np.unique(faultseg_preds_all[:10000]))}")
        else:
            print(f"  FaultSeg3D model not found: {faultseg_model_path}")
            faultseg_preds_all = None

        # 只有当所有模型都可用时才生成对比曲线
        comparison_available = faultseg_available and resunet_model is not None and swinunetr_model is not None and faultseg_preds_all is not None
        
        if comparison_available:

            # 二值化GT
            gt_binary = (all_gts > 0.5).astype(np.float32)

            # 采样（与单模型曲线一致）
            n_samples = len(gt_binary)
            if n_samples > 1000000:
                np.random.seed(42)
                idx = np.random.choice(n_samples, 1000000, replace=False)
                gt_sample = gt_binary[idx]
                ours_sample = all_preds[idx]
                fs_sample = faultseg_preds_all[idx]
                res_sample = resunet_preds_all[idx]
                swin_sample = swinunetr_preds_all[idx]
            else:
                gt_sample = gt_binary
                ours_sample = all_preds
                fs_sample = faultseg_preds_all
                res_sample = resunet_preds_all
                swin_sample = swinunetr_preds_all

            # 计算FaultSeg3D指标
            fs_precision, fs_recall, _ = precision_recall_curve(gt_sample, fs_sample)
            fs_ap = average_precision_score(gt_sample, fs_sample)
            fs_f1_scores = 2 * (fs_precision * fs_recall) / (fs_precision + fs_recall + 1e-8)
            fs_best_f1 = np.max(fs_f1_scores)
            fs_fpr, fs_tpr, _ = roc_curve(gt_sample, fs_sample)
            fs_roc_auc = auc(fs_fpr, fs_tpr)

            # 计算ResUNet指标
            res_precision, res_recall, _ = precision_recall_curve(gt_sample, res_sample)
            res_ap = average_precision_score(gt_sample, res_sample)
            res_f1_scores = 2 * (res_precision * res_recall) / (res_precision + res_recall + 1e-8)
            res_best_f1 = np.max(res_f1_scores)
            res_fpr, res_tpr, _ = roc_curve(gt_sample, res_sample)
            res_roc_auc = auc(res_fpr, res_tpr)

            # 计算SwinUNETR指标
            swin_precision, swin_recall, _ = precision_recall_curve(gt_sample, swin_sample)
            swin_ap = average_precision_score(gt_sample, swin_sample)
            swin_f1_scores = 2 * (swin_precision * swin_recall) / (swin_precision + swin_recall + 1e-8)
            swin_best_f1 = np.max(swin_f1_scores)
            swin_fpr, swin_tpr, _ = roc_curve(gt_sample, swin_sample)
            swin_roc_auc = auc(swin_fpr, swin_tpr)

            # 计算Our Model指标
            ours_precision, ours_recall, _ = precision_recall_curve(gt_sample, ours_sample)
            ours_ap = average_precision_score(gt_sample, ours_sample)
            ours_f1_scores = 2 * (ours_precision * ours_recall) / (ours_precision + ours_recall + 1e-8)
            ours_best_f1 = np.max(ours_f1_scores)
            ours_fpr, ours_tpr, _ = roc_curve(gt_sample, ours_sample)
            ours_roc_auc = auc(ours_fpr, ours_tpr)

            # 颜色配置
            COLOR_FAULTSEG = '#2ca02c'   # 绿色
            COLOR_RESUNET = '#1f77b4'    # 蓝色 (新增)
            COLOR_SWINUNETR = '#ff7f0e'  # 橙色 (新增)
            COLOR_OURS = '#d62728'       # 红色

            # 保存到figures/curves/
            figures_curve_dir = './EXP/' + args.exp + '/figures/curves/'
            os.makedirs(figures_curve_dir, exist_ok=True)

            # PR对比曲线
            _font = {'family': 'serif', 'serif': ['Times New Roman', 'DejaVu Serif']}
            fig, ax = plt.subplots(figsize=(4.0, 3.5))
            plt.rc('font', **_font)
            ax.plot(fs_recall, fs_precision, color=COLOR_FAULTSEG, linewidth=2,
                    label=f"FaultSeg3D (AP={fs_ap:.4f}, F1={fs_best_f1:.4f})")
            ax.plot(res_recall, res_precision, color=COLOR_RESUNET, linewidth=2,
                    label=f"ResUNet (AP={res_ap:.4f}, F1={res_best_f1:.4f})")
            ax.plot(swin_recall, swin_precision, color=COLOR_SWINUNETR, linewidth=2,
                    label=f"SwinUNETR (AP={swin_ap:.4f}, F1={swin_best_f1:.4f})")
            ax.plot(ours_recall, ours_precision, color=COLOR_OURS, linewidth=2,
                    label=f"Ours (AP={ours_ap:.4f}, F1={ours_best_f1:.4f})")
            ax.set_xlabel('Recall', fontsize=8)
            ax.set_ylabel('Precision', fontsize=8)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.legend(loc='lower left', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            pr_path = os.path.join(figures_curve_dir, 'PR_comparison.png')
            plt.savefig(pr_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {pr_path}")

            # ROC对比曲线
            fig, ax = plt.subplots(figsize=(4.0, 3.5))
            plt.rc('font', **_font)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax.plot(fs_fpr, fs_tpr, color=COLOR_FAULTSEG, linewidth=2,
                    label=f"FaultSeg3D (AUC={fs_roc_auc:.4f})")
            ax.plot(res_fpr, res_tpr, color=COLOR_RESUNET, linewidth=2,
                    label=f"ResUNet (AUC={res_roc_auc:.4f})")
            ax.plot(swin_fpr, swin_tpr, color=COLOR_SWINUNETR, linewidth=2,
                    label=f"SwinUNETR (AUC={swin_roc_auc:.4f})")
            ax.plot(ours_fpr, ours_tpr, color=COLOR_OURS, linewidth=2,
                    label=f"Ours (AUC={ours_roc_auc:.4f})")
            ax.set_xlabel('False Positive Rate', fontsize=8)
            ax.set_ylabel('True Positive Rate', fontsize=8)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.legend(loc='lower right', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            roc_path = os.path.join(figures_curve_dir, 'ROC_comparison.png')
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {roc_path}")

            print(f"  FaultSeg3D: AP={fs_ap:.4f}, AUC={fs_roc_auc:.4f}, F1={fs_best_f1:.4f}")
            print(f"  ResUNet:    AP={res_ap:.4f}, AUC={res_roc_auc:.4f}, F1={res_best_f1:.4f}")
            print(f"  SwinUNETR:  AP={swin_ap:.4f}, AUC={swin_roc_auc:.4f}, F1={swin_best_f1:.4f}")
            print(f"  Ours:       AP={ours_ap:.4f}, AUC={ours_roc_auc:.4f}, F1={ours_best_f1:.4f}")
        else:
            print("  Skipping comparison curves (not all models available)")
            if not faultseg_available:
                print(f"    - FaultSeg3D model not found: {faultseg_model_path}")
            if resunet_model is None:
                print(f"    - ResUNet model not found")
            if swinunetr_model is None:
                print(f"    - SwinUNETR model not found")
        # ==============================================================

        print("---")
        print("Save Finished ! ")
