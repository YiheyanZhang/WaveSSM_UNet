"""
论文图片生成脚本 - IEEE TGRS

使用方法:
    # 单独生成某类图片
    python generate_paper_figures.py --exp msmunet_V12 --pretrained_model_name MSM_UNet_BEST.pth --mode valid
    python generate_paper_figures.py --exp msmunet_V12 --pretrained_model_name MSM_UNet_BEST.pth --mode pred --pred_data_name f3
    python generate_paper_figures.py --exp msmunet_V12 --pretrained_model_name MSM_UNet_BEST.pth --mode comparison
    python generate_paper_figures.py --exp msmunet_V12 --pretrained_model_name MSM_UNet_BEST.pth --mode f3_comparison
    python generate_paper_figures.py --exp msmunet_V12 --pretrained_model_name MSM_UNet_BEST.pth --mode kerry_comparison

    # 一键生成所有图片
    python generate_paper_figures.py --exp msmunet_V12 --pretrained_model_name MSM_UNet_BEST.pth --mode generate_all

注意: PR/ROC曲线现在在train.py的valid函数中生成（使用连续概率值而非二值化预测）

输出目录: EXP/{exp}/figures/
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from models.MSM_UNet import MSM_UNet
from dataloader.dataloader import FaultDataset
from torch.utils.data import DataLoader
from utils.tools import load_pred_data

# ============== 全局配置 ==============
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# IEEE TGRS 尺寸 (inch)
SINGLE_COL = 3.5
DOUBLE_COL = 7.0

# 3D视图统一参数
FIG_3D_SIZE = (7.0, 3.5)  # 双栏宽度，固定高度
VIEW_ELEV = 25
VIEW_AZIM = 125


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--exp", required=True, type=str, help="Experiment name")
    parser.add_argument("--pretrained_model_name", default="MSM_UNet_BEST.pth", type=str)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--mode", default='valid', choices=['valid', 'pred', 'all', 'valid_3d', 'comparison', 'f3_comparison', 'kerry_comparison', 'generate_all', 'generate_3d'], type=str)
    parser.add_argument("--valid_path", default="data/validation/", type=str)
    parser.add_argument("--pred_data_name", default="f3", choices=['f3', 'kerry'], type=str)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=1, type=int)
    parser.add_argument("--sample_ids", default="0,10,11,12", type=str, help="Comma-separated sample IDs for valid_3d mode")
    parser.add_argument("--model_type", default='msm_unet',
                       choices=['msm_unet', 'UNET3', 'RESUNET', 'SWINUNETR', 'FAULTSEG3D'],
                       type=str, help="Model type")
    return parser.parse_args()


def load_model(args):
    """加载模型"""
    if args.model_type == 'msm_unet':
        model = MSM_UNet(args.in_channels, args.out_channels)
    elif args.model_type == 'FAULTSEG3D':
        from models.faultseg3d import FaultSeg3D
        model = FaultSeg3D(n_channels=args.in_channels, n_classes=args.out_channels)
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
        model = build_model(config)
    
    model_path = f"./EXP/{args.exp}/models/{args.pretrained_model_name}"

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=args.device)
        model.load_state_dict(checkpoint)
        print(f"[Model] Loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = model.to(args.device)
    model.eval()
    return model


def save_3d_figure(filename, dpi=300):
    """
    保存3D图。
    使用 matplotlib 自带的 tight bbox 裁白边，pad_inches 留足空间避免轴标签被截。
    """
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def overlay_fault_on_slice(seis_slice, fault_slice, vmin=0.4, vmax=1.0):
    """将断层概率以红色叠加在灰度地震切片上"""
    seis_norm = (seis_slice - seis_slice.min()) / (seis_slice.max() - seis_slice.min() + 1e-8)
    rgb = np.stack([seis_norm, seis_norm, seis_norm], axis=-1)
    fault_norm = np.clip((fault_slice - vmin) / (vmax - vmin), 0, 1)
    alpha = fault_norm * 0.8
    rgb[:, :, 0] = rgb[:, :, 0] * (1 - alpha) + alpha
    rgb[:, :, 1] = rgb[:, :, 1] * (1 - alpha)
    rgb[:, :, 2] = rgb[:, :, 2] * (1 - alpha)
    return np.clip(rgb, 0, 1)


def draw_3d_slices(ax, seismic, fault, title, slice_indices=None, time_dt=None, elev=25, azim=125):
    """
    绘制三正交切片的3D视图。

    约定轴顺序: seismic.shape = [D, H, W] = [Time, Inline, Crossline]
    调用方需提前将数据转置至此顺序。

    Args:
        time_dt: 每个时间样本的时间间隔(ms)，如 4 表示 4ms/sample。
                 None 表示使用样本索引作为刻度。
    """
    D, H, W = seismic.shape

    # 切片位置
    if slice_indices is None:
        depth_idx  = D - 1    # 底部：最深时间切片
        inline_idx = 0        # 后墙：最小 Inline（Y=0，视角最远处）
        xline_idx  = W - 1   # 左墙：最大 Crossline（X=W-1，视角左侧）
    else:
        depth_idx, inline_idx, xline_idx = slice_indices

    # ====== 底部水平 Time 切片 (Z=depth_idx) ======
    # 数据 shape [H, W]=[Inline, Crossline]，行→Y(Inline)，列→X(Crossline)
    slice_bottom = overlay_fault_on_slice(seismic[depth_idx, :, :], fault[depth_idx, :, :])
    X_bottom, Y_bottom = np.meshgrid(np.arange(W), np.arange(H))  # 均为 [H, W]
    Z_bottom = np.full_like(X_bottom, depth_idx, dtype=float)
    ax.plot_surface(X_bottom, Y_bottom, Z_bottom,
                    facecolors=slice_bottom, shade=False, alpha=0.95,
                    rcount=H, ccount=W)

    # ====== Inline 剖面 (Y=inline_idx)：展示 Time×Crossline ======
    # 数据 shape [D, W]=[Time, Crossline]，行→Z(Time)，列→X(Crossline)
    slice_il = overlay_fault_on_slice(seismic[:, inline_idx, :], fault[:, inline_idx, :])
    X_il, Z_il = np.meshgrid(np.arange(W), np.arange(D))           # 均为 [D, W]
    Y_il = np.full_like(X_il, inline_idx, dtype=float)
    ax.plot_surface(X_il, Y_il, Z_il,
                    facecolors=slice_il, shade=False, alpha=0.95,
                    rcount=D, ccount=W)

    # ====== Crossline 剖面 (X=xline_idx)：展示 Time×Inline ======
    # 数据 shape [D, H]=[Time, Inline]，行→Z(Time)，列→Y(Inline)
    slice_xl = overlay_fault_on_slice(seismic[:, :, xline_idx], fault[:, :, xline_idx])
    Y_xl, Z_xl = np.meshgrid(np.arange(H), np.arange(D))           # 均为 [D, H]
    X_xl = np.full_like(Y_xl, xline_idx, dtype=float)
    ax.plot_surface(X_xl, Y_xl, Z_xl,
                    facecolors=slice_xl, shade=False, alpha=0.95,
                    rcount=D, ccount=H)

    if title:  # 只有标题非空时才显示
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # 设置坐标范围
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, D)

    # 反转Z轴（Time向下，0在顶部，D在底部）
    ax.invert_zaxis()

    # 设置视角（移入函数统一管理）
    ax.view_init(elev=elev, azim=azim)

    # ====== 坐标轴标签和刻度 ======
    _lfs = 16   # 轴标签字号（原8，加倍到16）
    _tfs = 12   # 刻度字号（原6，加倍到12）
    _n_ticks = 5  # 各轴刻度数

    # X 轴 → Crossline（0 ~ W-1）
    ax.set_xlabel('Crossline', fontsize=_lfs, labelpad=2)
    x_ticks = np.linspace(0, W - 1, _n_ticks, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in x_ticks], fontsize=_tfs)

    # Y 轴 → Inline（0 ~ H-1）
    ax.set_ylabel('Inline', fontsize=_lfs, labelpad=2)
    y_ticks = np.linspace(0, H - 1, _n_ticks, dtype=int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(v) for v in y_ticks], fontsize=_tfs)

    # Z 轴 → Time（样本索引 或 ms）
    z_ticks = np.linspace(0, D - 1, _n_ticks, dtype=int)
    ax.set_zticks(z_ticks)
    if time_dt is not None:
        ax.set_zticklabels([str(int(v * time_dt)) for v in z_ticks], fontsize=_tfs)
        z_label = 'Time (ms)'
    else:
        ax.set_zticklabels([str(v) for v in z_ticks], fontsize=_tfs)
        z_label = 'Time (sample)'
    # set_zlabel 在 invert_zaxis() + 特定视角下会被 3D box 遮挡，
    # 改用 text2D 在 2D 层绘制，始终可见。
    # 两种视角（azim=125 和 azim=-60）的 z 轴刻度均在右侧，固定放右侧。
    ax.set_zlabel('')
    ax.text2D(1.06, 0.5, z_label, transform=ax.transAxes,
              fontsize=_lfs, rotation=90, va='center', ha='left',
              clip_on=False)

    ax.tick_params(axis='x', labelsize=_tfs, pad=1)
    ax.tick_params(axis='y', labelsize=_tfs, pad=1)
    ax.tick_params(axis='z', labelsize=_tfs, pad=1)


def generate_valid_3d_figure(seis, pred, gt, save_path, sample_id):
    """
    生成单个验证集样本的3D视图（仅GT）
    与tools.py中save_3d_orthogonal_view的GT部分完全一致

    布局: 单图 (Ground Truth)
    尺寸: 7 x 6 inch (与原始valid图片一致)

    Args:
        seis: 地震数据 [D, H, W]
        pred: 预测结果 [D, H, W] (未使用)
        gt: Ground Truth [D, H, W]
        save_path: 保存路径
        sample_id: 样本ID
    """
    fig = plt.figure(figsize=(7, 6))

    # 仅显示 Ground Truth
    ax = fig.add_subplot(111, projection='3d')
    draw_3d_slices(ax, seis, gt, '', time_dt=4)

    filename = f"{save_path}/valid_3d_gt_sample_{sample_id}.png"
    save_3d_figure(filename, dpi=300)

    print(f"  Saved: {filename}")
    return filename


def run_validation_3d(model, args, sample_ids):
    """
    生成指定ID的验证集样本的3D视图
    优先从现有npy文件加载，如果不存在则运行模型推理

    Args:
        model: 加载的模型
        args: 参数
        sample_ids: 要生成的样本ID列表
    """
    print(f"\n[Valid 3D] Generating 3D views for samples: {sample_ids}")

    # 创建保存目录
    save_dir = f"./EXP/{args.exp}/figures/valid_3d"
    os.makedirs(save_dir, exist_ok=True)

    # npy文件目录
    numpy_dir = f"./EXP/{args.exp}/results/valid/numpy"

    generated_files = []

    for i in sample_ids:
        print(f"\n  Processing sample {i}...")

        # 尝试从npy文件加载
        gt_path = os.path.join(numpy_dir, f'{i}_gt.npy')
        img_path = os.path.join(numpy_dir, f'{i}_img.npy')
        seg_path = os.path.join(numpy_dir, f'{i}_seg.npy')

        if os.path.exists(gt_path) and os.path.exists(img_path):
            print(f"    Loading from existing npy files...")
            gt = np.load(gt_path)
            seis = np.load(img_path)
            if os.path.exists(seg_path):
                pred = np.load(seg_path)
            else:
                pred = None
            print(f"    Loaded: seis {seis.shape}, gt {gt.shape}")
        else:
            print(f"    npy files not found, running inference...")
            # 加载验证数据集
            valid_dataset = FaultDataset(args.valid_path)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

            for j, batch in enumerate(valid_loader):
                if j == i:
                    seis_tensor = batch['x'].to(args.device)
                    gt = batch['y'].numpy()[0]
                    if gt.ndim == 4:
                        gt = gt[0]

                    with torch.no_grad():
                        if args.model_type == 'msm_unet':
                            pred, _ = model(seis_tensor)
                        else:
                            pred = model(seis_tensor)
                        pred = pred.cpu().numpy()[0, 0]

                    seis = seis_tensor.cpu().numpy()[0, 0]
                    break

        # 生成3D图片
        filename = generate_valid_3d_figure(seis, pred, gt, save_dir, i)
        generated_files.append(filename)

    return generated_files


def sliding_window_predict(model, data, device, patch_size=128, overlap=0.5, model_type='msm_unet'):
    """滑动窗口预测"""
    D, H, W = data.shape
    stride = int(patch_size * (1 - overlap))

    output = np.zeros_like(data)
    count = np.zeros_like(data)

    # 归一化
    data_norm = (data - data.mean()) / (data.std() + 1e-8)

    for d in range(0, max(1, D - patch_size + 1), stride):
        for h in range(0, max(1, H - patch_size + 1), stride):
            for w in range(0, max(1, W - patch_size + 1), stride):
                # 边界处理
                d_end = min(d + patch_size, D)
                h_end = min(h + patch_size, H)
                w_end = min(w + patch_size, W)
                d_start = d_end - patch_size
                h_start = h_end - patch_size
                w_start = w_end - patch_size

                patch = data_norm[d_start:d_end, h_start:h_end, w_start:w_end]
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    if model_type == 'msm_unet':
                        pred, _ = model(patch_tensor)
                    else:
                        pred = model(patch_tensor)
                    pred = pred.cpu().numpy()[0, 0]

                output[d_start:d_end, h_start:h_end, w_start:w_end] += pred
                count[d_start:d_end, h_start:h_end, w_start:w_end] += 1

    count[count == 0] = 1
    output = output / count

    return output


def run_validation(model, args):
    """运行验证集推理"""
    print("\n[Validation] Loading validation dataset...")

    valid_dataset = FaultDataset(args.valid_path)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    all_seis = []
    all_pred = []
    all_gt = []

    for i, batch in enumerate(valid_loader):
        seis = batch['x'].to(args.device)
        gt = batch['y'].numpy()[0]

        with torch.no_grad():
            if args.model_type == 'msm_unet':
                pred, _ = model(seis)
            else:
                pred = model(seis)
            pred = pred.cpu().numpy()[0, 0]

        seis = seis.cpu().numpy()[0, 0]

        all_seis.append(seis)
        all_pred.append(pred)
        all_gt.append(gt)

        print(f"  Processed sample {i+1}/{len(valid_loader)}")

    return all_seis, all_pred, all_gt


def run_prediction(model, args):
    """运行预测"""
    print(f"\n[Prediction] Loading {args.pred_data_name} dataset...")

    seis = load_pred_data(args)
    print(f"  Data shape: {seis.shape}")

    print(f"  Running sliding window prediction...")
    pred = sliding_window_predict(model, seis, args.device, model_type=args.model_type)

    # 加载GT（如果有）
    gt = None
    if args.pred_data_name == 'f3':
        gt_path = 'data/prediction/f3d/fpx.dat'
        if os.path.exists(gt_path):
            gt = np.fromfile(gt_path, dtype=np.float32).reshape(512, 384, 128)
            print(f"  GT loaded: {gt.shape}")

    return seis, pred, gt


def generate_comparison_3d(seis, gt, pred, save_dir, sample_id):
    """
    生成单个样本的3D视图（3张独立图片）
    - 原始地震数据
    - 原始地震数据 + GT
    - 原始地震数据 + 当前模型预测
    """
    fig_size = (7, 6)  # 统一尺寸

    zeros = np.zeros_like(seis)

    data_pairs = [
        (zeros, 'seis'),
        (gt, 'gt'),
        (pred, 'ours')
    ]

    saved_files = []
    for fault_data, suffix in data_pairs:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        draw_3d_slices(ax, seis, fault_data, '', time_dt=4)

        filename = f"{save_dir}/valid_3d_sample{sample_id}_{suffix}.png"
        save_3d_figure(filename, dpi=300)
        saved_files.append(filename)
        print(f"      Saved: {filename}")

    return saved_files


def generate_comparison_2d_slices(seis, gt, pred, save_dir, sample_id, slice_config):
    """
    生成单个样本的2D切片对比图

    Args:
        slice_config: dict, 指定要生成的切片，格式如 {'inline': [50], 'crossline': [], 'depth': []}
    """
    D, H, W = seis.shape
    fig_size = (4, 4)  # 统一尺寸

    # 用全零的fault生成纯地震数据图
    zeros = np.zeros_like(seis)

    data_pairs = [
        (zeros, 'seis'),  # 原始地震数据
        (gt, 'gt'),
        (pred, 'ours')
    ]

    saved_files = []

    # ====== Inline切片 (沿Y轴切) ======
    for pos in slice_config.get('inline', []):
        if pos >= H:
            print(f"      Warning: Inline position {pos} >= H={H}, skipping")
            continue
        seis_slice = seis[:, pos, :]  # [D, W]

        for fault_data, suffix in data_pairs:
            fault_slice = fault_data[:, pos, :]

            fig, ax = plt.subplots(figsize=fig_size)
            overlay = overlay_fault_on_slice(seis_slice, fault_slice)
            # Time 轴用 4ms 分辨率：extent=[x_min, x_max, y_max, y_min]
            ax.imshow(overlay, aspect='auto', extent=[0, W, D*4, 0])
            ax.set_xlabel('Crossline', fontsize=12)
            ax.set_ylabel('Time (ms)', fontsize=12)
            ax.tick_params(labelsize=11)
            plt.tight_layout()

            filename = f"{save_dir}/sample{sample_id}_inline{pos}_{suffix}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(filename)
            print(f"      Saved: {filename}")

    # ====== Crossline切片 (沿X轴切) ======
    for pos in slice_config.get('crossline', []):
        if pos >= W:
            print(f"      Warning: Crossline position {pos} >= W={W}, skipping")
            continue
        seis_slice = seis[:, :, pos]  # [D, H]

        for fault_data, suffix in data_pairs:
            fault_slice = fault_data[:, :, pos]

            fig, ax = plt.subplots(figsize=fig_size)
            overlay = overlay_fault_on_slice(seis_slice, fault_slice)
            # Time 轴用 4ms 分辨率：extent=[x_min, x_max, y_max, y_min]
            ax.imshow(overlay, aspect='auto', extent=[0, H, D*4, 0])
            ax.set_xlabel('Inline', fontsize=12)
            ax.set_ylabel('Time (ms)', fontsize=12)
            ax.tick_params(labelsize=11)
            plt.tight_layout()

            filename = f"{save_dir}/sample{sample_id}_crossline{pos}_{suffix}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(filename)
            print(f"      Saved: {filename}")

    # ====== Depth切片 (沿Z轴切) ======
    for pos in slice_config.get('depth', []):
        if pos >= D:
            print(f"      Warning: Depth position {pos} >= D={D}, skipping")
            continue
        seis_slice = seis[pos, :, :]  # [H, W]

        for fault_data, suffix in data_pairs:
            fault_slice = fault_data[pos, :, :]

            fig, ax = plt.subplots(figsize=fig_size)
            overlay = overlay_fault_on_slice(seis_slice, fault_slice)
            ax.imshow(overlay, aspect='auto')
            ax.set_xlabel('Crossline', fontsize=12)
            ax.set_ylabel('Inline', fontsize=12)
            ax.tick_params(labelsize=11)
            plt.tight_layout()

            filename = f"{save_dir}/sample{sample_id}_depth{pos}_{suffix}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(filename)
            print(f"      Saved: {filename}")

    return saved_files


def run_comparison_mode(args):
    """
    运行当前模型的可视化模式（合成数据集）：
    - ID 0: 生成3D图
    - ID 10: depth=50 切片
    - ID 11: inline=50 切片
    - ID 12: crossline=50 切片

    数据来源：
    - Current Model: EXP/{exp}/results/valid/numpy/
    """
    print("\n[Comparison Mode] Generating comparison figures...")

    # 创建输出目录
    save_dir_3d = f"./EXP/{args.exp}/figures/comparison_3d"
    save_dir_2d = f"./EXP/{args.exp}/figures/comparison_2d"
    os.makedirs(save_dir_3d, exist_ok=True)
    os.makedirs(save_dir_2d, exist_ok=True)

    # npy文件目录
    ours_numpy_dir = f"./EXP/{args.exp}/results/valid/numpy"

    # 配置：哪些ID生成什么图片
    id_3d = [0]  # 3D图片
    id_2d_config = {
        10: {'inline': [], 'crossline': [], 'depth': [50]},
        11: {'inline': [50], 'crossline': [], 'depth': []},
        12: {'inline': [], 'crossline': [50], 'depth': []}
    }
    all_ids = set(id_3d + list(id_2d_config.keys()))

    # 从npy文件加载
    print("\n  [1] Loading from npy files...")
    all_saved_files = []

    for i in all_ids:
        print(f"\n      Processing sample {i}...")

        # 检查文件存在
        ours_seg_path = os.path.join(ours_numpy_dir, f'{i}_seg.npy')
        ours_img_path = os.path.join(ours_numpy_dir, f'{i}_img.npy')
        ours_gt_path = os.path.join(ours_numpy_dir, f'{i}_gt.npy')

        if not os.path.exists(ours_seg_path):
            raise FileNotFoundError(f"Our Model prediction not found: {ours_seg_path}")
        if not os.path.exists(ours_img_path):
            raise FileNotFoundError(f"Seismic data not found: {ours_img_path}")
        if not os.path.exists(ours_gt_path):
            raise FileNotFoundError(f"Ground truth not found: {ours_gt_path}")

        # 加载数据
        seis_np = np.load(ours_img_path)
        gt = np.load(ours_gt_path)
        ours_pred = np.load(ours_seg_path)

        print(f"      Loaded: seis {seis_np.shape}, gt {gt.shape}, ours {ours_pred.shape}")

        # 生成3D图
        if i in id_3d:
            print(f"      Generating 3D figures...")
            files = generate_comparison_3d(seis_np, gt, ours_pred, save_dir_3d, i)
            all_saved_files.extend(files)

        # 生成2D切片图
        if i in id_2d_config:
            print(f"      Generating 2D slice figures...")
            slice_config = id_2d_config[i]
            files = generate_comparison_2d_slices(seis_np, gt, ours_pred, save_dir_2d, i, slice_config)
            all_saved_files.extend(files)

    return all_saved_files


def run_f3_comparison_mode(args):
    """
    在F3数据集上生成当前模型的3D和2D图
    - Current Model: 从EXP/{exp}/results/pred/f3/numpy/f3.npy加载

    F3数据轴顺序说明：
      原始存储: [Crossline, Inline, Time] = [512, 384, 128]
      转置后:   [Time, Inline, Crossline] = [128, 384, 512]
      对应 draw_3d_slices 约定: D=Time, H=Inline, W=Crossline
    """
    print("\n[F3 Comparison Mode] Generating F3 comparison figures...")

    # 创建输出目录
    save_dir = f"./EXP/{args.exp}/figures/f3_comparison"
    os.makedirs(save_dir, exist_ok=True)

    # ====== 加载F3地震数据 ======
    print("\n  [1] Loading F3 seismic data...")
    args.pred_data_name = 'f3'
    seis = load_pred_data(args)
    print(f"      Raw seismic shape [Crossline, Inline, Time]: {seis.shape}")

    # F3原始存储顺序为 [Crossline, Inline, Time]，转置为 [Time, Inline, Crossline]
    seis = np.transpose(seis, (2, 1, 0))
    print(f"      Transposed shape  [Time, Inline, Crossline]: {seis.shape}")
    T, NI, NX = seis.shape  # T=128, NI=384, NX=512

    # 加载GT并做相同转置
    gt_path = 'data/prediction/f3d/fpx.dat'
    if os.path.exists(gt_path):
        gt = np.fromfile(gt_path, dtype=np.float32).reshape(512, 384, 128)
        gt = np.transpose(gt, (2, 1, 0))  # [Crossline, Inline, Time] -> [Time, Inline, Crossline]
        print(f"      GT shape [Time, Inline, Crossline]: {gt.shape}")
    else:
        raise FileNotFoundError(f"F3 GT not found: {gt_path}")

    # ====== 从npy文件加载预测结果 ======
    print("\n  [2] Loading prediction from npy file...")

    ours_pred_path = f"./EXP/{args.exp}/results/pred/f3/numpy/f3.npy"
    if not os.path.exists(ours_pred_path):
        raise FileNotFoundError(f"Our Model F3 prediction not found: {ours_pred_path}")
    ours_pred = np.load(ours_pred_path)
    # 预测结果与原始数据同轴顺序，同样转置
    if ours_pred.shape == (512, 384, 128):
        ours_pred = np.transpose(ours_pred, (2, 1, 0))
    print(f"      Prediction shape [Time, Inline, Crossline]: {ours_pred.shape}")

    # ====== 生成3D图 ======
    print("\n  [3] Generating 3D figures...")

    fig_size = (7, 6)  # 统一尺寸
    zeros = np.zeros_like(seis)

    data_pairs = [
        (zeros, 'seis'),
        (gt, 'gt'),
        (ours_pred, 'ours')
    ]

    saved_files = []
    for fault_data, suffix in data_pairs:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        draw_3d_slices(ax, seis, fault_data, '', time_dt=4)

        filename = f"{save_dir}/f3_3d_{suffix}.png"
        save_3d_figure(filename, dpi=300)
        saved_files.append(filename)
        print(f"      Saved: {filename}")

    # ====== 生成2D Time切片对比图 (time=100) ======
    print(f"\n  [4] Generating 2D Time-slice figures (time_pos=100 / {T} samples)...")

    time_pos = 100
    seis_slice = seis[time_pos, :, :]  # [NI, NX] = [Inline, Crossline]

    for fault_data, suffix in data_pairs:
        fault_slice = fault_data[time_pos, :, :]

        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * NI / NX))
        overlay = overlay_fault_on_slice(seis_slice, fault_slice)
        ax.imshow(overlay, origin='upper', aspect='equal',
                  extent=[0, NX, NI, 0])
        ax.set_xlabel('Crossline', fontsize=9)
        ax.set_ylabel('Inline', fontsize=9)
        ax.tick_params(labelsize=7)

        filename = f"{save_dir}/f3_time{time_pos}_{suffix}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        saved_files.append(filename)
        print(f"      Saved: {filename}")

    return saved_files


def run_kerry_comparison_mode(args):
    """
    在Kerry数据集上生成当前模型的3D和2D图
    - Current Model: 从EXP/{exp}/results/pred/kerry/numpy/kerry.npy加载
    """
    print("\n[Kerry Comparison Mode] Generating Kerry comparison figures...")

    # 创建输出目录
    save_dir = f"./EXP/{args.exp}/figures/kerry_comparison"
    os.makedirs(save_dir, exist_ok=True)

    # ====== 加载Kerry地震数据 ======
    print("\n  [1] Loading Kerry seismic data...")
    args.pred_data_name = 'kerry'
    seis = load_pred_data(args)
    print(f"      Original shape: {seis.shape}")

    # Kerry数据维度转置为 (D, H, W)
    seis = np.transpose(seis, (2, 0, 1))
    print(f"      Transposed shape (D, H, W): {seis.shape}")

    # ====== 加载预测结果 ======
    print("\n  [2] Loading prediction...")

    ours_pred_path = f"./EXP/{args.exp}/results/pred/kerry/numpy/kerry.npy"
    if not os.path.exists(ours_pred_path):
        raise FileNotFoundError(f"Our Model Kerry prediction not found: {ours_pred_path}")
    ours_pred = np.load(ours_pred_path)
    if ours_pred.shape != seis.shape:
        print(f"      Current model pred shape {ours_pred.shape} != seis shape {seis.shape}, transposing...")
        ours_pred = np.transpose(ours_pred, (2, 0, 1))
    print(f"      Current model prediction loaded: {ours_pred.shape}")

    # ====== 准备数据对 (Kerry没有GT) ======
    zeros = np.zeros_like(seis)
    data_pairs = [
        (zeros, 'seis'),
        (ours_pred, 'ours')
    ]

    saved_files = []

    # ====== 生成3D对比图 ======
    print("\n  [3] Generating 3D comparison figures...")

    fig_size = (7, 6)  # 统一尺寸
    D, H, W = seis.shape

    # Kerry数据边界有大量0值，使用指定的有效切片位置
    # 切片放在数据体的"后面"和"右边"，配合azim=-60视角更清晰
    # (depth_idx, inline_idx, xline_idx) = (time, inline, crossline)
    kerry_slice_indices = (400, 237, 535)
    print(f"      Using slice indices: time={kerry_slice_indices[0]}, inline={kerry_slice_indices[1]}, crossline={kerry_slice_indices[2]}")

    for fault_data, suffix in data_pairs:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        draw_3d_slices(ax, seis, fault_data, '', slice_indices=kerry_slice_indices,
                       time_dt=4, elev=20, azim=-60)

        filename = f"{save_dir}/kerry_3d_{suffix}.png"
        save_3d_figure(filename, dpi=300)
        saved_files.append(filename)
        print(f"      Saved: {filename}")

    # ====== 生成2D切片对比图 (time=215, 220, 225) ======
    print("\n  [4] Generating 2D slice comparison figures...")

    D, H, W = seis.shape
    time_positions = [215, 220, 225]

    for time_pos in time_positions:
        if time_pos >= D:
            print(f"      Warning: time position {time_pos} >= D={D}, skipping")
            continue

        print(f"      Generating time={time_pos} slices...")
        seis_slice = seis[time_pos, :, :]  # [H, W] = [Inline, Crossline]

        for fault_data, suffix in data_pairs:
            fault_slice = fault_data[time_pos, :, :]

            fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * H / W))
            overlay = overlay_fault_on_slice(seis_slice, fault_slice)
            ax.imshow(overlay, origin='upper', aspect='equal',
                      extent=[0, W, H, 0])
            ax.set_xlabel('Crossline', fontsize=9)
            ax.set_ylabel('Inline', fontsize=9)
            ax.tick_params(labelsize=7)

            filename = f"{save_dir}/kerry_time{time_pos}_{suffix}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            saved_files.append(filename)
            print(f"        Saved: {filename}")

    return saved_files


def main():
    args = parse_args()

    print("=" * 60)
    print("Paper Figure Generator")
    print("=" * 60)
    print(f"Experiment: {args.exp}")
    print(f"Model Type: {args.model_type}")
    print(f"Model File: {args.pretrained_model_name}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # 创建输出目录
    fig_dir = f"./EXP/{args.exp}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Output directory: {fig_dir}")

    # generate_all模式：一键生成所有图片
    if args.mode == 'generate_all':
        all_files = []

        print("\n" + "=" * 60)
        print("[1/4] Generating validation set comparison figures...")
        print("=" * 60)
        try:
            files = run_comparison_mode(args)
            all_files.extend(files)
            print(f"  Comparison figures generated: {len(files)} files")
        except Exception as e:
            print(f"  Comparison figures failed: {e}")

        print("\n" + "=" * 60)
        print("[2/4] Generating F3 comparison figures...")
        print("=" * 60)
        try:
            args.pred_data_name = 'f3'
            files = run_f3_comparison_mode(args)
            all_files.extend(files)
            print(f"  F3 comparison figures generated: {len(files)} files")
        except Exception as e:
            print(f"  F3 comparison figures failed: {e}")

        print("\n" + "=" * 60)
        print("[3/4] Generating Kerry comparison figures...")
        print("=" * 60)
        try:
            args.pred_data_name = 'kerry'
            files = run_kerry_comparison_mode(args)
            all_files.extend(files)
            print(f"  Kerry comparison figures generated: {len(files)} files")
        except Exception as e:
            print(f"  Kerry comparison figures failed: {e}")

        print("\n" + "=" * 60)
        print("[4/4] Generating validation 3D figures...")
        print("=" * 60)
        try:
            model = load_model(args)
            sample_ids = [int(x.strip()) for x in args.sample_ids.split(',')]
            files = run_validation_3d(model, args, sample_ids)
            all_files.extend(files)
            print(f"  Validation 3D figures generated: {len(files)} files")
        except Exception as e:
            print(f"  Validation 3D figures failed: {e}")

        print("\n" + "=" * 60)
        print(f"[Done] Generated {len(all_files)} figures in total!")
        print("=" * 60)
        print(f"\nOutput directories:")
        print(f"  - {fig_dir}/comparison_3d/")
        print(f"  - {fig_dir}/comparison_2d/")
        print(f"  - {fig_dir}/f3_comparison/")
        print(f"  - {fig_dir}/kerry_comparison/")
        print(f"  - {fig_dir}/valid_3d/")
        print(f"\nNote: PR/ROC curves are generated during validation (train.py valid function)")
        print("\n" + "=" * 60)
        print("Finished!")
        print("=" * 60)
        return

    # comparison模式：自己加载两个模型
    if args.mode == 'comparison':
        saved_files = run_comparison_mode(args)
        print(f"\n[Done] Generated {len(saved_files)} comparison figures:")
        print(f"  3D figures: ./EXP/{args.exp}/figures/comparison_3d/")
        print(f"  2D figures: ./EXP/{args.exp}/figures/comparison_2d/")
        print("\n" + "=" * 60)
        print("Finished!")
        print("=" * 60)
        return

    # f3_comparison模式：在F3数据集上生成3D对比图
    if args.mode == 'f3_comparison':
        saved_files = run_f3_comparison_mode(args)
        print(f"\n[Done] Generated {len(saved_files)} F3 comparison figures:")
        print(f"  Figures: ./EXP/{args.exp}/figures/f3_comparison/")
        for f in saved_files:
            print(f"    - {f}")
        print("\n" + "=" * 60)
        print("Finished!")
        print("=" * 60)
        return

    # kerry_comparison模式：在Kerry数据集上生成3D和2D对比图
    if args.mode == 'kerry_comparison':
        saved_files = run_kerry_comparison_mode(args)
        print(f"\n[Done] Generated {len(saved_files)} Kerry comparison figures:")
        print(f"  Figures: ./EXP/{args.exp}/figures/kerry_comparison/")
        for f in saved_files:
            print(f"    - {f}")
        print("\n" + "=" * 60)
        print("Finished!")
        print("=" * 60)
        return

    # generate_3d模式：只生成 F3/Kerry 3D + 合成数据 3D/2D
    if args.mode == 'generate_3d':
        all_files = []

        print("\n" + "=" * 60)
        print("[1/3] Generating synthetic data 3D and 2D figures...")
        print("=" * 60)
        try:
            files = run_comparison_mode(args)
            all_files.extend(files)
            print(f"  Synthetic figures generated: {len(files)} files")
        except Exception as e:
            print(f"  Synthetic figures failed: {e}")

        print("\n" + "=" * 60)
        print("[2/3] Generating F3 3D figures...")
        print("=" * 60)
        try:
            args.pred_data_name = 'f3'
            files = run_f3_comparison_mode(args)
            all_files.extend(files)
            print(f"  F3 figures generated: {len(files)} files")
        except Exception as e:
            print(f"  F3 figures failed: {e}")

        print("\n" + "=" * 60)
        print("[3/3] Generating Kerry 3D figures...")
        print("=" * 60)
        try:
            args.pred_data_name = 'kerry'
            files = run_kerry_comparison_mode(args)
            all_files.extend(files)
            print(f"  Kerry figures generated: {len(files)} files")
        except Exception as e:
            print(f"  Kerry figures failed: {e}")

        print("\n" + "=" * 60)
        print(f"[Done] Generated {len(all_files)} figures in total!")
        print("=" * 60)
        print(f"\nOutput directories:")
        print(f"  - {fig_dir}/comparison_3d/")
        print(f"  - {fig_dir}/comparison_2d/")
        print(f"  - {fig_dir}/f3_comparison/")
        print(f"  - {fig_dir}/kerry_comparison/")
        print("\n" + "=" * 60)
        print("Finished!")
        print("=" * 60)
        return

    # 加载模型
    model = load_model(args)

    # 根据模式运行
    if args.mode == 'valid':
        seis_list, pred_list, gt_list = run_validation(model, args)
        print(f"\n[Done] Validation completed. {len(pred_list)} samples processed.")
        print(f"  Data available: seis_list, pred_list, gt_list")

    elif args.mode == 'valid_3d':
        # 解析样本ID
        sample_ids = [int(x.strip()) for x in args.sample_ids.split(',')]
        print(f"Generating 3D views for samples: {sample_ids}")

        generated_files = run_validation_3d(model, args, sample_ids)

        print(f"\n[Done] Generated {len(generated_files)} 3D figures:")
        for f in generated_files:
            print(f"  - {f}")

    elif args.mode == 'pred':
        seis, pred, gt = run_prediction(model, args)
        has_gt = gt is not None
        print(f"\n[Done] Prediction completed.")
        print(f"  Seismic: {seis.shape}")
        print(f"  Prediction: {pred.shape}")
        print(f"  GT: {'Available' if has_gt else 'Not available'}")

        # 保存预测结果
        save_dir = f"{fig_dir}/{args.pred_data_name}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/seis.npy", seis)
        np.save(f"{save_dir}/pred.npy", pred)
        if has_gt:
            np.save(f"{save_dir}/gt.npy", gt)
        print(f"  Results saved to: {save_dir}/")

    elif args.mode == 'all':
        # 运行所有
        seis_list, pred_list, gt_list = run_validation(model, args)
        print(f"\n[Done] Validation: {len(pred_list)} samples")

        for data_name in ['f3', 'kerry']:
            args.pred_data_name = data_name
            seis, pred, gt = run_prediction(model, args)
            save_dir = f"{fig_dir}/{data_name}"
            os.makedirs(save_dir, exist_ok=True)
            np.save(f"{save_dir}/seis.npy", seis)
            np.save(f"{save_dir}/pred.npy", pred)
            if gt is not None:
                np.save(f"{save_dir}/gt.npy", gt)
            print(f"  {data_name} saved to: {save_dir}/")

    print("\n" + "=" * 60)
    print("Finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()
