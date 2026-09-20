"""Train, evaluate and transfer CIG-Bench's public fault HRNet."""

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "CIG-bench"))

from cig_bench.networks.hrnet_skipconect_opt import HRNet
from cig_bench.predictor import ensure_weight
from cig_bench.predictor.utils import tensor_z_score_clip
from dataloader.dataloader import FaultDataset
from utils.segmentation_metrics import confusion_metrics, mean_volume_metrics
from utils.tools import load_pred_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval", "predict"))
    parser.add_argument("--exp", default="cig_bench")
    parser.add_argument("--train_path", default="data/train")
    parser.add_argument("--valid_path", default="data/validation")
    parser.add_argument("--weights", default="public",
                        help="CIG-Bench .pth path, or 'public' for the published checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_curve_voxels", type=int, default=1_000_000)
    parser.add_argument("--dataset", choices=("f3", "kerry"), default="f3")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="field inference scale; use 0.5 only if Kerry exceeds VRAM")
    parser.add_argument("--inference_mode", choices=("native", "sliding"), default="native")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.5)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess(x):
    """Official CIG-Bench tensor normalization, mapped to [-1, 1]."""
    return tensor_z_score_clip(x, clp_s=3.0) * 2 - 1


def window_starts(size, patch_size, overlap):
    if patch_size <= 0 or patch_size > size or not 0 <= overlap < 1:
        raise ValueError("patch_size must fit the volume and overlap must be in [0, 1)")
    stride = max(1, int(patch_size * (1 - overlap)))
    starts = list(range(0, size - patch_size + 1, stride))
    if starts[-1] != size - patch_size:
        starts.append(size - patch_size)
    return starts


@torch.no_grad()
def sliding_window_probability(model, volume, device, patch_size=128, overlap=0.5):
    starts = [window_starts(size, patch_size, overlap) for size in volume.shape]
    probability = np.zeros(volume.shape, dtype=np.float32)
    count = np.zeros(volume.shape, dtype=np.float32)
    windows = itertools.product(*starts)
    total = np.prod([len(axis) for axis in starts])
    for d, h, w in tqdm(windows, total=total, desc="[CIG-Bench] sliding"):
        region = np.s_[d:d + patch_size, h:h + patch_size, w:w + patch_size]
        x = torch.from_numpy(volume[region])[None, None].float().to(device)
        with torch.autocast("cuda", enabled=str(device).startswith("cuda")):
            patch = torch.sigmoid(model(preprocess(x), rank=0))
        probability[region] += patch[0, 0].float().cpu().numpy()
        count[region] += 1
    return probability / count


def paper_region_loss(probability, target):
    """Paper region loss: 0.7 Dice + 0.3 Focal(alpha=.75, gamma=2)."""
    probability = probability.squeeze(1).clamp(1e-7, 1 - 1e-7)
    intersection = (probability * target).sum()
    dice = 1 - (2 * intersection + 1e-5) / (probability.sum() + target.sum() + 1e-5)
    pt = torch.where(target == 1, probability, 1 - probability)
    alpha = torch.where(target == 1, 0.75, 0.25)
    focal = (alpha * (1 - pt).square() * F.binary_cross_entropy(
        probability, target, reduction="none")).mean()
    return 0.7 * dice + 0.3 * focal


def load_state(model, weights, device):
    path = ensure_weight("fault") if weights == "public" else weights
    state = torch.load(path, map_location=device)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    print(f"[Weights] {path}")


def update_confusion(counts, probability, target, threshold):
    pred = probability.squeeze(1) > threshold
    truth = target > 0.5
    counts[0] += torch.sum(~pred & ~truth).item()
    counts[1] += torch.sum(pred & ~truth).item()
    counts[2] += torch.sum(~pred & truth).item()
    counts[3] += torch.sum(pred & truth).item()


@torch.no_grad()
def evaluate(model, loader, args, save=False):
    model.eval()
    counts = [0, 0, 0, 0]
    volume_counts = []
    probabilities, labels = [], []
    numpy_dir = ROOT / "EXP" / args.exp / "results" / "valid" / "numpy"
    if save:
        numpy_dir.mkdir(parents=True, exist_ok=True)

    for index, data in enumerate(tqdm(loader, desc="[CIG-Bench] eval")):
        x, y = data["x"].to(args.device), data["y"].to(args.device)
        logits = model(preprocess(x), rank=args.rank, chunk_size=args.chunk_size)
        probability = torch.sigmoid(logits)
        update_confusion(counts, probability, y, args.threshold)
        for batch_index in range(probability.shape[0]):
            current = [0, 0, 0, 0]
            update_confusion(
                current,
                probability[batch_index:batch_index + 1],
                y[batch_index:batch_index + 1],
                args.threshold,
            )
            volume_counts.append(current)
        if save:
            p = probability[0, 0].float().cpu().numpy()
            image = x[0, 0].float().cpu().numpy()
            gt = y[0].float().cpu().numpy()
            np.save(numpy_dir / f"{index}_seg.npy", p)
            np.save(numpy_dir / f"{index}_img.npy", image)
            np.save(numpy_dir / f"{index}_gt.npy", gt)
            probabilities.append(p.ravel())
            labels.append(gt.ravel())

    metrics = confusion_metrics(counts)
    metrics.update(mean_volume_metrics(volume_counts))
    metrics["validation_volumes"] = len(volume_counts)
    metrics["miou_dice_aggregation"] = "mean_over_volumes"
    if save:
        probability = np.concatenate(probabilities)
        label = (np.concatenate(labels) > 0.5).astype(np.uint8)
        if label.size > args.max_curve_voxels:
            indices = np.random.default_rng(args.seed).choice(
                label.size, args.max_curve_voxels, replace=False)
            probability, label = probability[indices], label[indices]
        precision, recall, thresholds = precision_recall_curve(label, probability)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best = int(np.argmax(f1))
        metrics.update({
            "f1": float(f1[best]),
            "best_threshold": float(thresholds[best]) if best < thresholds.size else 0.5,
            "roc_auc": float(roc_auc_score(label, probability)),
            "pr_auc": float(average_precision_score(label, probability)),
            "curve_voxels": int(label.size),
        })
        result_dir = numpy_dir.parent
        np.savez_compressed(result_dir / "curves.npz", y_true=label, y_score=probability)
        (result_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train(args):
    seed_everything(args.seed)
    train_loader = DataLoader(
        FaultDataset(args.train_path, mode="train", augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    valid_loader = DataLoader(
        FaultDataset(args.valid_path, mode="train", augment=False),
        batch_size=1, shuffle=False, num_workers=args.workers)
    model = HRNet(c=32).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_dir = ROOT / "EXP" / args.exp / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best, stale = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for data in tqdm(train_loader, desc=f"[CIG-Bench] {epoch}/{args.epochs}"):
            x, y = data["x"].to(args.device), data["y"].to(args.device)
            probability = torch.sigmoid(model(preprocess(x), rank=0))
            loss = paper_region_loss(probability, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        metrics = evaluate(model, valid_loader, args)
        score = (metrics["miou"] + metrics["dice"]) / 2
        print(f"epoch={epoch} loss={np.mean(losses):.6f} "
              f"mIoU={metrics['miou']:.4f} Dice={metrics['dice']:.4f}")
        if score > best:
            best, stale = score, 0
            torch.save(model.state_dict(), model_dir / "CIG_Bench_BEST.pth")
        else:
            stale += 1
        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_dir / f"CIG_Bench_epoch_{epoch}.pth")
        if stale >= args.patience:
            print(f"[Early stop] no improvement for {args.patience} epochs")
            break
    torch.save(model.state_dict(), model_dir / "CIG_Bench_LAST.pth")


def eval_model(args):
    seed_everything(args.seed)
    loader = DataLoader(FaultDataset(args.valid_path, mode="train", augment=False),
                        batch_size=1, shuffle=False, num_workers=args.workers)
    model = HRNet(c=32).to(args.device)
    load_state(model, args.weights, args.device)
    print(json.dumps(evaluate(model, loader, args, save=True), indent=2))


@torch.no_grad()
def predict(args):
    model = HRNet(c=32).to(args.device).eval()
    load_state(model, args.weights, args.device)
    volume = load_pred_data(argparse.Namespace(pred_data_name=args.dataset))
    volume = np.transpose(volume, (2, 1, 0)) if args.dataset == "f3" else np.transpose(volume, (2, 0, 1))
    if args.inference_mode == "sliding":
        if args.scale != 1.0:
            raise ValueError("sliding inference requires --scale 1.0")
        probability = sliding_window_probability(
            model, volume, args.device, args.patch_size, args.overlap)
    else:
        original_shape = volume.shape
        work_shape = tuple(max(16, (round(size * args.scale) // 16) * 16)
                           for size in original_shape)
        x = torch.from_numpy(volume)[None, None].float()
        x = F.interpolate(x, work_shape, mode="trilinear").to(args.device)
        with torch.autocast("cuda", enabled=args.device.startswith("cuda")):
            probability = torch.sigmoid(model(preprocess(x), rank=args.rank,
                                              chunk_size=args.chunk_size))
        probability = F.interpolate(probability, original_shape, mode="nearest-exact")
        probability = probability[0, 0].float().cpu().numpy()
    output = ROOT / "EXP" / args.exp / "results" / "pred" / args.dataset / "numpy"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f"{args.dataset}.npy", probability)
    np.save(output / f"{args.dataset}_img.npy", volume)
    print(f"[Saved] {output}")


if __name__ == "__main__":
    arguments = parse_args()
    {"train": train, "eval": eval_model, "predict": predict}[arguments.mode](arguments)
