import os

from models.MSM_UNet import MSM_UNet
from utils.tools import save_pred_picture, load_pred_data
from models.faultseg3d import FaultSeg3D
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


def get_gaussian_mask(n1, n2, n3, os):
    """
    生成高斯权重掩码（修复版：三维联合计算）

    在重叠边界处使用高斯衰减权重，实现平滑过渡

    Args:
        n1, n2, n3: 块的尺寸
        os: overlap宽度（像素）
    Returns:
        sc: [n1, n2, n3] 高斯权重掩码
    """
    # 为每个维度生成1D权重
    def get_1d_weight(n, overlap):
        w = np.ones(n, dtype=np.float32)
        sig = overlap / 4
        sig = 0.5 / (sig * sig + 1e-8)
        for k in range(overlap):
            ds = k - overlap + 1
            weight = np.exp(-ds * ds * sig)
            w[k] = weight
            w[n - k - 1] = weight
        return w

    # 生成三个维度的1D权重
    w1 = get_1d_weight(n1, os)
    w2 = get_1d_weight(n2, os)
    w3 = get_1d_weight(n3, os)

    # 三维联合计算：三个维度权重相乘（而非覆盖）
    sc = np.einsum('i,j,k->ijk', w1, w2, w3)

    return sc


def sliding_window_prediction_faultseg(input_data, model, args):
    """
    滑动窗口预测（与训练一致使用z-score标准化）

    特点：
    1. 使用args.overlap参数控制重叠比例
    2. 高斯加权融合（不是简单平均）
    3. 每块单独z-score标准化（与训练一致！）
    4. Mirror padding消除边界伪影
    """
    m1, m2, m3 = input_data.shape
    n1, n2, n3 = 128, 128, 128  # 块尺寸
    os = int(n1 * args.overlap)  # 例如 0.25 * 128 = 32像素

    # Mirror padding - 在边界处扩展数据，消除边界伪影
    pad = os  # padding大小等于overlap
    input_padded = np.pad(input_data,
                          ((pad, pad), (pad, pad), (pad, pad)),
                          mode='reflect')
    pm1, pm2, pm3 = input_padded.shape
    print(f"[Mirror Padding] Original: ({m1}, {m2}, {m3}), Padded: ({pm1}, {pm2}, {pm3})")

    # 计算需要切割的块数（基于padded数据）
    c1 = int(np.round((pm1 + os) / (n1 - os) + 0.5))
    c2 = int(np.round((pm2 + os) / (n2 - os) + 0.5))
    c3 = int(np.round((pm3 + os) / (n3 - os) + 0.5))

    # 计算填充后的尺寸
    p1 = (n1 - os) * c1 + os
    p2 = (n2 - os) * c2 + os
    p3 = (n3 - os) * c3 + os

    print(f"[Sliding Window] Input: ({pm1}, {pm2}, {pm3}), Sliding: ({p1}, {p2}, {p3})")
    print(f"[Sliding Window] Blocks: ({c1}, {c2}, {c3}), Overlap: {os} pixels ({args.overlap*100:.0f}%)")

    # 初始化
    gp = np.zeros((p1, p2, p3), dtype=np.float32)
    gy = np.zeros((p1, p2, p3), dtype=np.float32)
    mk = np.zeros((p1, p2, p3), dtype=np.float32)

    gp[0:pm1, 0:pm2, 0:pm3] = input_padded

    # 获取高斯权重掩码
    sc = get_gaussian_mask(n1, n2, n3, os)

    total_blocks = c1 * c2 * c3
    progress_bar = tqdm(total=total_blocks, desc='[Pred-Sliding]', unit='it')

    # Batch 推理加速：30G 显存，128³ patch，batch=8 约占 4GB
    batch_size = 8
    batch_inputs = []
    batch_coords = []

    for k1 in range(c1):
        for k2 in range(c2):
            for k3 in range(c3):
                # 计算块的起始和结束位置
                b1 = k1 * n1 - k1 * os
                e1 = b1 + n1
                b2 = k2 * n2 - k2 * os
                e2 = b2 + n2
                b3 = k3 * n3 - k3 * os
                e3 = b3 + n3

                # 提取块
                gs = gp[b1:e1, b2:e2, b3:e3].copy()

                # 与训练一致：z-score标准化
                xm = np.mean(gs)
                xs = np.std(gs)
                if xs > 0:
                    gs = (gs - xm) / xs

                # 转置并reshape为模型输入格式
                gs_input = np.transpose(gs)
                gs_input = gs_input.reshape((1, 1, n1, n2, n3))

                batch_inputs.append(gs_input)
                batch_coords.append((b1, e1, b2, e2, b3, e3))

                # 攒够一个 batch 或到最后一块时推理
                if len(batch_inputs) == batch_size or (k1 == c1-1 and k2 == c2-1 and k3 == c3-1):
                    input_tensor = torch.from_numpy(np.concatenate(batch_inputs, axis=0)).to(args.device).float()
                    with torch.no_grad():
                        predictions = model(input_tensor)

                    if isinstance(predictions, tuple):
                        predictions = predictions[0]

                    predictions = predictions.detach().cpu().numpy()

                    for idx, (bb1, be1, bb2, be2, bb3, be3) in enumerate(batch_coords):
                        pred_block = np.squeeze(predictions[idx])
                        pred_block = np.transpose(pred_block)
                        gy[bb1:be1, bb2:be2, bb3:be3] += pred_block * sc
                        mk[bb1:be1, bb2:be2, bb3:be3] += sc

                    progress_bar.update(len(batch_inputs))
                    batch_inputs = []
                    batch_coords = []

    progress_bar.close()

    # 归一化
    gy = gy / mk

    # 【修改】裁剪掉padding区域，返回原始尺寸
    return gy[pad:pad+m1, pad:pad+m2, pad:pad+m3]


def direct_prediction_faultseg(input_data, model, args):
    """
    直接整体预测（与原始faultSeg apply.py完全一致）

    特点：
    1. 全局z-score标准化
    2. 一次性输入整个数据
    3. 无块边界问题
    4. 支持FP16推理节省显存
    """
    m1, m2, m3 = input_data.shape
    use_fp16 = getattr(args, 'fp16', False)
    print(f"[Direct Prediction] Input shape: ({m1}, {m2}, {m3}), FP16: {use_fp16}")

    # 与faultSeg apply.py完全一致：全局z-score标准化
    gm = np.mean(input_data)
    gs = np.std(input_data)
    gx = (input_data - gm) / gs
    print(f"[Direct Prediction] Normalization: mean={gm:.4f}, std={gs:.4f}")

    # 转置
    gx = np.transpose(gx)

    # reshape为模型输入格式 [1, 1, n1, n2, n3]
    gx_input = gx.reshape((1, 1, gx.shape[0], gx.shape[1], gx.shape[2]))

    # 模型预测
    if use_fp16:
        # FP16推理：节省约50%显存
        model.half()
        input_tensor = torch.from_numpy(gx_input).to(args.device).half()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                prediction = model(input_tensor)
        # 转回FP32
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = prediction.float().detach().cpu().numpy()
    else:
        # FP32推理
        input_tensor = torch.from_numpy(gx_input).to(args.device).float()
        with torch.no_grad():
            prediction = model(input_tensor)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = prediction.detach().cpu().numpy()

    prediction = np.squeeze(prediction)

    # 反向转置
    output = np.transpose(prediction)

    return output

def _create_gaussian_weight(block_shape, sigma_ratio=0.25):
    """
    创建3D高斯权重块 - 中心权重高、边缘权重低

    Args:
        block_shape: 块的形状 (D, H, W)
        sigma_ratio: sigma相对于块大小的比例
    """
    weight = np.ones(block_shape, dtype=np.float32)

    for dim in range(3):
        size = block_shape[dim]
        sigma = size * sigma_ratio
        # 创建1D高斯权重
        center = size / 2
        coords = np.arange(size)
        gaussian_1d = np.exp(-((coords - center) ** 2) / (2 * sigma ** 2))

        # 扩展到对应维度
        shape = [1, 1, 1]
        shape[dim] = size
        gaussian_1d = gaussian_1d.reshape(shape)

        weight = weight * gaussian_1d

    # 归一化到[0.1, 1]，避免边缘权重过低
    weight = weight / weight.max()
    weight = np.clip(weight, 0.1, 1.0)

    return weight



def pred_Gaussian(args):
    print("============================== pred_Gaussian ==============================")

    # Thebe 走专门流程（多子体 + 有GT定量评估）
    if args.pred_data_name == 'thebe':
        pred_thebe(args)
        return

    input_data = load_pred_data(args)  # 输入数据

    # 使用训练好的模型进行预测
    if args.model_type == 'msm_unet':
        model = MSM_UNet(args.in_channels, args.out_channels).to(args.device)
    else:
        # 使用compare_models中的对比模型
        from compare_models.build import build_model
        
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
                    'img_size': (128, 128, 128)
                })
        
        config = ModelConfig()
        model = build_model(config).to(args.device)
        print(f"Using model: {args.model_type}")
    
    model_path = './EXP/' + args.exp + '/models/' + args.pretrained_model_name

    # 加载权重，兼容带/不带 module. 前缀的模型
    state_dict = torch.load(model_path, map_location=args.device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    print("Loaded model from disk")
    model.eval()

    # 根据参数选择预测方式
    if args.use_sliding_window:
        print("[Prediction Mode] Sliding Window")
        output_data = sliding_window_prediction_faultseg(input_data, model, args)
    else:
        print("[Prediction Mode] Direct (Full Volume)")
        output_data = direct_prediction_faultseg(input_data, model, args)

    # 不做二值化，直接输出概率图（与faultSeg原版一致）
    # threshold = args.threshold
    # output_data[output_data > threshold] = 1
    # output_data[output_data <= threshold] = 0

    print("---Start Save results  ······")
    save_path = './EXP/' + args.exp + '/results/pred/' + args.pred_data_name + '/'
    if not os.path.exists(save_path + '/numpy/'):
        os.makedirs(save_path + '/numpy/')
    if not os.path.exists(save_path + '/picture/'):
        os.makedirs(save_path + '/picture/')
    np.save(save_path + '/numpy/' + args.pred_data_name + '.npy', output_data)

    save_pred_picture(input_data, output_data, save_path + '/picture/', args.pred_data_name)
    print("Finish!!!")


def pred_thebe(args):
    """
    Thebe 真实数据集预测 + 定量评估（有GT）

    数据结构:
        data/prediction/thebe_data/seis/seistest1.npz  (可能含多个子体)
        data/prediction/thebe_data/label/faulttest1~7.npy

    流程:
        1. 加载地震数据（npz 解压后按子体切分或直接对应 label）
        2. 逐子体滑动窗口推理
        3. 与 GT 计算 IoU / Dice / Precision / Recall / F1
        4. 汇总输出
    """
    from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

    print("=" * 60)
    print("[Thebe] Real-world dataset evaluation (with Ground Truth)")
    print("=" * 60)

    device = args.device if torch.cuda.is_available() else 'cpu'
    thebe_dir = 'data/prediction/thebe_data'

    # ====== 加载模型 ======
    if args.model_type == 'msm_unet':
        model = MSM_UNet(args.in_channels, args.out_channels).to(device)
    elif args.model_type == 'FAULTSEG3D':
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
                self.data = type('obj', (object,), {'img_size': (128, 128, 128)})
        config = ModelConfig()
        model = build_model(config).to(device)

    model_path = './EXP/' + args.exp + '/models/' + args.pretrained_model_name
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    # 修复可变形卷积权重 shape 不匹配（checkpoint [C,C,27] vs model [C,C,3,3,3]）
    model_sd = model.state_dict()
    for k in new_state_dict:
        if k in model_sd and new_state_dict[k].shape != model_sd[k].shape:
            if new_state_dict[k].numel() == model_sd[k].numel():
                new_state_dict[k] = new_state_dict[k].reshape(model_sd[k].shape)
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"[Model] {args.model_type} loaded from {model_path}")

    # ====== 加载地震数据 ======
    seis_path = os.path.join(thebe_dir, 'seis', 'seistest1.npz')
    seis_npz = np.load(seis_path)
    seis_keys = sorted(seis_npz.keys())
    print(f"[Data] Seismic npz keys: {seis_keys}")

    # ====== 加载标签 ======
    label_dir = os.path.join(thebe_dir, 'label')
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
    # 只用第一个子体做评估（快速验证）
    label_files = label_files[:1]
    n_volumes = len(label_files)
    print(f"[Data] Using {n_volumes} label volumes: {label_files}")

    # 确定地震子体与标签的对应关系
    # 情况1: npz 里有多个 key，一一对应 label
    # 情况2: npz 里只有 1 个大体，需要按 label shape 切分
    if len(seis_keys) >= n_volumes:
        seis_volumes = [seis_npz[seis_keys[i]] for i in range(n_volumes)]
        print(f"[Data] Using {n_volumes} arrays from npz (one per label)")
    else:
        # 只有一个大体，先加载所有 label 看 shape，再决定怎么切
        seis_full = seis_npz[seis_keys[0]]
        print(f"[Data] Single seismic volume: {seis_full.shape}")
        # 按第一个维度均分
        label_0 = np.load(os.path.join(label_dir, label_files[0]))
        vol_shape = label_0.shape
        print(f"[Data] Label volume shape: {vol_shape}")
        # 尝试按 label shape 从大体中切出子体
        # 如果大体就是和 label 同 shape，说明所有 label 对应同一个地震体
        if seis_full.shape == vol_shape:
            seis_volumes = [seis_full] * n_volumes
            print(f"[Data] Seismic shape == label shape, using same volume for all labels")
        else:
            # 按第一维切分
            chunk = seis_full.shape[0] // n_volumes
            seis_volumes = [seis_full[i*chunk:(i+1)*chunk] for i in range(n_volumes)]
            print(f"[Data] Split seismic along dim0: chunk_size={chunk}")

    # ====== 保存路径 ======
    save_path = './EXP/' + args.exp + '/results/pred/thebe/'
    os.makedirs(save_path + '/numpy/', exist_ok=True)
    os.makedirs(save_path + '/picture/', exist_ok=True)

    # ====== 逐子体推理 + 评估 ======
    threshold = args.threshold
    all_metrics = []

    for i in range(n_volumes):
        print(f"\n{'='*40} Volume {i+1}/{n_volumes} {'='*40}")

        # 加载地震子体
        seis_vol = seis_volumes[i].astype(np.float32)
        # 加载 GT
        gt = np.load(os.path.join(label_dir, label_files[i])).astype(np.float32)
        print(f"  Seismic: {seis_vol.shape}, GT: {gt.shape}")
        print(f"  GT stats: min={gt.min():.4f}, max={gt.max():.4f}, "
              f"fault_ratio={gt[gt>0.5].size / gt.size * 100:.2f}%")

        # 滑动窗口推理
        print(f"  Running sliding window prediction (overlap={args.overlap})...")
        pred = sliding_window_prediction_faultseg(seis_vol, model, args)
        print(f"  Prediction: min={pred.min():.4f}, max={pred.max():.4f}")

        # 保存预测
        np.save(save_path + f'/numpy/thebe_vol{i+1}.npy', pred.astype(np.float32))
        # 保存对应的 GT 和地震数据（便于后续可视化）
        np.save(save_path + f'/numpy/thebe_vol{i+1}_gt.npy', gt.astype(np.float32))
        np.save(save_path + f'/numpy/thebe_vol{i+1}_seis.npy', seis_vol)

        # ====== 计算定量指标 ======
        gt_binary = (gt > 0.5).astype(np.float32).flatten()
        pred_flat = pred.flatten()
        pred_binary = (pred_flat > threshold).astype(np.float32)

        # IoU
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
        iou = intersection / (union + 1e-8)

        # Dice
        dice = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)

        # Precision / Recall / F1
        tp = np.sum(pred_binary * gt_binary)
        fp = np.sum(pred_binary * (1 - gt_binary))
        fn = np.sum((1 - pred_binary) * gt_binary)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics = {'vol': i+1, 'iou': iou, 'dice': dice,
                   'precision': precision, 'recall': recall, 'f1': f1}
        all_metrics.append(metrics)

        print(f"  [Vol {i+1}] IoU={iou:.4f}  Dice={dice:.4f}  "
              f"Prec={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

        # ====== 可视化切片对比 ======
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        vis_dir = save_path + '/picture/'
        # 沿三个轴各取中间切片
        d0, d1, d2 = seis_vol.shape
        slices = [
            ('dim0', d0 // 2, seis_vol[d0//2, :, :], pred[d0//2, :, :], gt[d0//2, :, :]),
            ('dim1', d1 // 2, seis_vol[:, d1//2, :], pred[:, d1//2, :], gt[:, d1//2, :]),
            ('dim2', d2 // 2, seis_vol[:, :, d2//2], pred[:, :, d2//2], gt[:, :, d2//2]),
        ]

        for dim_name, idx, seis_slice, pred_slice, gt_slice in slices:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(seis_slice.T, cmap='gray', aspect='auto')
            axes[0].set_title(f'Seismic ({dim_name}={idx})')
            axes[0].axis('off')

            axes[1].imshow(pred_slice.T, cmap='hot', vmin=0, vmax=1, aspect='auto')
            axes[1].set_title(f'Prediction (prob)')
            axes[1].axis('off')

            axes[2].imshow(gt_slice.T, cmap='hot', vmin=0, vmax=1, aspect='auto')
            axes[2].set_title(f'Ground Truth')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(vis_dir + f'thebe_vol{i+1}_{dim_name}.png', dpi=150)
            plt.close()

        print(f"  [Vis] Saved to {vis_dir}thebe_vol{i+1}_dim*.png")

    # ====== 汇总 ======
    print("\n" + "=" * 70)
    print(f"[Thebe] Summary  (threshold={threshold})")
    print("=" * 70)
    print(f"{'Vol':<6}{'IoU':>10}{'Dice':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    print("-" * 58)
    for m in all_metrics:
        print(f"{m['vol']:<6}{m['iou']:>10.4f}{m['dice']:>10.4f}"
              f"{m['precision']:>12.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}")
    print("-" * 58)

    # 平均
    avg_iou = np.mean([m['iou'] for m in all_metrics])
    avg_dice = np.mean([m['dice'] for m in all_metrics])
    avg_prec = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    print(f"{'Avg':<6}{avg_iou:>10.4f}{avg_dice:>10.4f}"
          f"{avg_prec:>12.4f}{avg_recall:>10.4f}{avg_f1:>10.4f}")
    print("=" * 70)

    # 保存指标到文件
    metrics_path = save_path + 'thebe_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Thebe Quantitative Evaluation\n")
        f.write(f"Model: {args.model_type} | Exp: {args.exp}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"{'='*58}\n\n")
        f.write(f"{'Vol':<6}{'IoU':>10}{'Dice':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}\n")
        f.write(f"{'-'*58}\n")
        for m in all_metrics:
            f.write(f"{m['vol']:<6}{m['iou']:>10.4f}{m['dice']:>10.4f}"
                    f"{m['precision']:>12.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}\n")
        f.write(f"{'-'*58}\n")
        f.write(f"{'Avg':<6}{avg_iou:>10.4f}{avg_dice:>10.4f}"
                f"{avg_prec:>12.4f}{avg_recall:>10.4f}{avg_f1:>10.4f}\n")
    print(f"\n[Save] Metrics saved to {metrics_path}")
    print("[Thebe] Finish!!!")
