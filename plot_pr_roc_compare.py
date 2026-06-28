"""
任意数量模型的 PR / ROC 对比曲线生成脚本（不依赖训练流程）。

用法（两模型对比）：
    python plot_pr_roc_compare.py \
        --models "Ours:msm_unet:EXP/wavessm_fast_r0.25_v2/models/MSM_UNet_BEST.pth" \
                 "SwinUNETR:SWINUNETR:EXP/swinunetr_fast_r0.25/models/MSM_UNet_BEST.pth" \
        --valid_path data/validation/ \
        --out_dir EXP/wavessm_fast_r0.25_v2/figures/curves_vs_swin

每个 --models 条目格式：  "显示名:model_type:权重路径"
    model_type 取值: msm_unet / FAULTSEG3D / RESUNET / SWINUNETR / RESACEUNET
    显示名会出现在图例里（如 "Ours" / "SwinUNETR"）

输出: out_dir 下生成
    PR_compare.png
    ROC_compare.png
    metrics_compare.txt
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

from dataloader.dataloader import FaultDataset
from models.MSM_UNet import MSM_UNet


# 给前 8 个模型预备一组好看的颜色（与 train.py 风格一致）
DEFAULT_COLORS = [
    '#d62728',  # 红 - 通常给 Ours
    '#ff7f0e',  # 橙
    '#1f77b4',  # 蓝
    '#2ca02c',  # 绿
    '#9467bd',  # 紫
    '#8c564b',  # 棕
    '#e377c2',  # 粉
    '#7f7f7f',  # 灰
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True,
                   help='每个条目: "显示名:model_type:权重路径"')
    p.add_argument('--valid_path', default='data/validation/', type=str)
    p.add_argument('--in_channels', default=1, type=int)
    p.add_argument('--out_channels', default=1, type=int)
    p.add_argument('--batch_size', default=1, type=int)
    p.add_argument('--workers', default=0, type=int)
    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--out_dir', required=True, type=str)
    p.add_argument('--max_samples', default=1_000_000, type=int,
                   help='画曲线时最多采样这么多体素，默认 100w')
    p.add_argument('--seed', default=42, type=int)
    return p.parse_args()


def build_one_model(model_type, in_channels, out_channels, device):
    """按 model_type 实例化模型（与 train.py 同口径，避免 sigmoid 重叠）。"""
    if model_type == 'msm_unet':
        return MSM_UNet(in_channels, out_channels).to(device)
    if model_type == 'FAULTSEG3D':
        from models.faultseg3d import FaultSeg3D
        return FaultSeg3D(in_channels, out_channels).to(device)
    # 其余走 compare_models/build.py
    from compare_models.build import build_model
    cfg = type('obj', (object,), {
        'model': type('obj', (object,), {
            'name': model_type,
            'in_chans': in_channels,
            'num_classes': out_channels,
            'drop': 0.0, 'attn_drop': 0.0, 'drop_path': 0.0,
        }),
        'data': type('obj', (object,), {'img_size': 128}),
    })()
    return build_model(cfg).to(device)


def load_weights(model, weight_path, device):
    sd = torch.load(weight_path, map_location=device)
    new_sd = {(k.replace('module.', '') if k.startswith('module.') else k): v
              for k, v in sd.items()}
    model.load_state_dict(new_sd)
    model.eval()
    return model


def infer_one_model(model, model_type, val_loader, device, name):
    """跑一遍验证集，返回展平的概率预测和（首次调用时）GT。"""
    preds = []
    gts = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f'[{name}] infer'):
            x = data['x'].to(device)
            y = data['y'].numpy()
            out = model(x)
            seg = out[0] if isinstance(out, tuple) else out
            preds.append(seg.detach().cpu().numpy())
            gts.append(y)
    preds = np.concatenate([p.flatten() for p in preds]).astype(np.float32)
    gts = np.concatenate([g.flatten() for g in gts]).astype(np.float32)
    return preds, gts


def parse_model_spec(spec):
    """'Ours:msm_unet:path/to/x.pth' -> (name, model_type, path)"""
    parts = spec.split(':', 2)
    if len(parts) != 3:
        raise ValueError(f"模型条目格式错误: {spec}\n应为 '显示名:model_type:权重路径'")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    # 数据集（推理模式不增强）
    val_ds = FaultDataset(args.valid_path, mode='train', augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, drop_last=False)
    print(f'[Data] {len(val_ds)} validation samples')

    specs = [parse_model_spec(s) for s in args.models]

    # 逐模型推理
    all_results = []   # list of (name, preds_flat)
    gt_flat = None
    for name, mtype, path in specs:
        if not os.path.exists(path):
            raise FileNotFoundError(f'权重不存在: {path}')
        print(f'\n[Model] {name} | type={mtype} | path={path}')
        model = build_one_model(mtype, args.in_channels, args.out_channels, device)
        model = load_weights(model, path, device)
        preds, gts = infer_one_model(model, mtype, val_loader, device, name)
        if gt_flat is None:
            gt_flat = (gts > 0.5).astype(np.float32)
        else:
            assert preds.shape[0] == gt_flat.shape[0], \
                f'{name} 预测体素数 {preds.shape[0]} 与 GT 不一致'
        all_results.append((name, preds))
        # 释放显存，下一个模型再加载
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # 采样（避免 100w+ 体素跑 sklearn 太慢）
    n = gt_flat.shape[0]
    if n > args.max_samples:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(n, args.max_samples, replace=False)
        gt_sample = gt_flat[idx]
        sampled = [(name, preds[idx]) for name, preds in all_results]
        print(f'[Sample] {n} -> {args.max_samples} voxels')
    else:
        gt_sample = gt_flat
        sampled = all_results

    # 计算各模型指标
    metrics = []
    for i, (name, preds) in enumerate(sampled):
        precision, recall, _ = precision_recall_curve(gt_sample, preds)
        ap = average_precision_score(gt_sample, preds)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_f1 = float(np.max(f1))
        fpr, tpr, _ = roc_curve(gt_sample, preds)
        roc_auc = float(auc(fpr, tpr))
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        metrics.append({
            'name': name, 'color': color,
            'precision': precision, 'recall': recall, 'ap': ap, 'best_f1': best_f1,
            'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
        })

    # ============== PR 曲线 ==============
    plt.rc('font', family='serif', serif=['Times New Roman', 'DejaVu Serif'])
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    for m in metrics:
        ax.plot(m['recall'], m['precision'], color=m['color'], linewidth=2,
                label=f"{m['name']} (AP={m['ap']:.4f}, F1={m['best_f1']:.4f})")
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)
    ax.set_title('PR Curve Comparison', fontsize=11)
    pr_path = os.path.join(args.out_dir, 'PR_compare.png')
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[Save] {pr_path}')

    # ============== ROC 曲线 ==============
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    for m in metrics:
        ax.plot(m['fpr'], m['tpr'], color=m['color'], linewidth=2,
                label=f"{m['name']} (AUC={m['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)  # 对角线
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title('ROC Curve Comparison', fontsize=11)
    roc_path = os.path.join(args.out_dir, 'ROC_compare.png')
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[Save] {roc_path}')

    # ============== 指标 txt ==============
    txt_path = os.path.join(args.out_dir, 'metrics_compare.txt')
    with open(txt_path, 'w') as f:
        f.write(f'Validation samples : {len(val_ds)}\n')
        f.write(f'Voxels used        : {gt_sample.shape[0]}\n\n')
        f.write(f"{'Model':<24}{'AP':>10}{'AUC':>10}{'Best F1':>10}\n")
        f.write('-' * 54 + '\n')
        for m in metrics:
            f.write(f"{m['name']:<24}{m['ap']:>10.4f}{m['roc_auc']:>10.4f}{m['best_f1']:>10.4f}\n")
    print(f'[Save] {txt_path}')

    print('\n=== Summary ===')
    for m in metrics:
        print(f"  {m['name']:<20} AP={m['ap']:.4f}  AUC={m['roc_auc']:.4f}  F1={m['best_f1']:.4f}")


if __name__ == '__main__':
    main()
