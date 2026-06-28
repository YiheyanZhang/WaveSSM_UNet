# Fault Segmentation Based on Pytorch
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch

from utils.train import train, valid
from utils.test import pred_Gaussian
from utils.tools import save_args_info


def add_args():
    parser = argparse.ArgumentParser(description="FaultSeg3D_pytorch")

    parser.add_argument("--exp", default="test", type=str, help="Name of each run")
    parser.add_argument("--device", default='cuda', type=str, help="GPU id for training")
    parser.add_argument("--mode", default='train', choices=['train', 'valid_only', 'pred'], type=str, help='network run mode')
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--batch_size_not_train", default=1, type=int, help="number of batch size when not training")
    parser.add_argument("--epochs", default=250, type=int, help="max number of training epochs")
    parser.add_argument("--train_path", default="/root/autodl-tmp/WaveSSM/data/train/", type=str, help="dataset directory")
    parser.add_argument("--valid_path", default="/root/autodl-tmp/WaveSSM/data/validation/", type=str, help="dataset directory")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=1, type=int, help="number of output channels (固定为1，单通道Sigmoid)")
    parser.add_argument("--loss_func", default="bce+dice", choices=['dice', 'bce', 'bce_with_weight', 'bce+dice', 'cross_with_weight', 'cross+dice', 'fault_seg'], type=str, help="choose loss function")
    parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
    parser.add_argument("--optim_lr", default=2e-4, type=float, help="optimization learning rate")
    parser.add_argument("--workers", default=0, type=int, help="number of workers")
    # ============ 新增：优化器 / 调度器 / EMA / 早停 / 梯度累积 ============
    parser.add_argument("--optimizer", default="adamw", choices=['adam', 'adamw'], type=str, help="optimizer type")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay (only for adamw)")
    parser.add_argument("--lr_scheduler", default="warmup_cosine", choices=['none', 'cosine', 'warmup_cosine'], type=str, help="learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs (linear warmup)")
    parser.add_argument("--min_lr", default=1e-6, type=float, help="minimum lr for cosine annealing")
    parser.add_argument("--grad_accum_steps", default=1, type=int, help="gradient accumulation steps (effective_bs = batch_size * grad_accum_steps)")
    parser.add_argument("--use_ema", default=True, type=lambda x: str(x).lower() == 'true', help="use EMA (Exponential Moving Average) of weights")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="EMA decay")
    parser.add_argument("--early_stop_patience", default=50, type=int, help="early stopping patience (in validation steps)")
    parser.add_argument("--best_metric", default="iou_dice", choices=['iou', 'dice', 'iou_dice'], type=str, help="metric for selecting best checkpoint: iou / dice / 0.5*iou+0.5*dice")
    parser.add_argument("--model_type", default="msm_unet",
                       choices=['msm_unet', 'FAULTSEG3D', 'RESUNET', 'RESACEUNET', 'SWINUNETR'],
                       help="选择模型类型: msm_unet或对比模型")
    parser.add_argument("--pretrained_model_name", default="MSM_UNet_BEST.pth", type=str, help="pretrained model name")
    parser.add_argument("--pred_data_name", default="f3", choices=['f3', 'kerry', 'thebe'], type=str, help="pretrained data name")
    parser.add_argument('--overlap', default=0.5, type=float, help='pred overlap ratio')
    parser.add_argument('--threshold', default=0.5, type=float, help='Classification threshold')
    parser.add_argument('--sigma', default=0.0, type=float, help='Gaussian filter sigma')
    parser.add_argument('--boundary_loss_weight', default=0.0, type=float, help='Weight for boundary loss')
    parser.add_argument('--use_sliding_window', default=True, type=lambda x: x.lower() != 'false', help='Use sliding window prediction (True) or direct prediction (False)')
    parser.add_argument('--fp16', default=False, type=lambda x: x.lower() == 'true', help='Use FP16 inference for direct prediction (saves ~50% VRAM)')
    parser.add_argument('--resume', default='', type=str, help='resume checkpoint path, e.g. ./EXP/myexp/models/MSM_UNet_epoch_50_CP.pth')
    parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch for resume (0-indexed)')
    # ============ 数据效率消融：训练集子采样（valid 不动）============
    parser.add_argument('--train_ratio', default=None, type=float,
                        help='训练集子采样比例 (0,1]。仅对 train_path 下采样, valid_path 保持完整。None=用全部训练集 (默认行为)')
    parser.add_argument('--split_seed', default=42, type=int,
                        help='train_ratio 子采样的随机种子 (固定后多模型同一份子集，公平比较)')

    args = parser.parse_args()

    print()
    print(">>>============= args ====================<<<")
    print()
    print(args)  # print command line args
    print()
    print(">>>=======================================<<<")

    return args


def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'valid_only':
        valid(args)
    elif args.mode == 'pred':
        pred_Gaussian(args)
    else:
        raise ValueError("Only ['train', 'valid_only', 'pred'] mode is supported.")

    save_args_info(args)


if __name__ == "__main__":
    args = add_args()
    main(args)
