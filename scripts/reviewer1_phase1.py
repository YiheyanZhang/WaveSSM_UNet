"""Audits and benchmarks requested by Reviewer 1, points 1--3 and 6.

This script deliberately keeps the reported measurements separate from model
training.  It writes JSON records that can be cited when the manuscript is
updated.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


SHAPE = (128, 128, 128)
PATCH = (128, 128, 128)
OVERLAP = 0.5


def _stats(values):
    values = [float(v) for v in values]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def inventory(root: Path):
    result = {"shape": list(SHAPE), "dtype": "float32", "splits": {}}
    expected_bytes = int(np.prod(SHAPE)) * np.dtype(np.float32).itemsize
    result["expected_bytes_per_volume"] = expected_bytes

    for split in ("train", "validation"):
        split_dir = root / "data" / split
        seis = sorted((split_dir / "seis").glob("*.dat"))
        fault = sorted((split_dir / "fault").glob("*.dat"))
        seis_names = {p.name for p in seis}
        fault_names = {p.name for p in fault}
        sizes = [p.stat().st_size for p in seis + fault]
        result["splits"][split] = {
            "seismic_count": len(seis),
            "fault_count": len(fault),
            "paired_count": len(seis_names & fault_names),
            "missing_seismic": sorted(fault_names - seis_names),
            "missing_fault": sorted(seis_names - fault_names),
            "unexpected_file_sizes": sum(size != expected_bytes for size in sizes),
            "seismic_bytes": _stats([p.stat().st_size for p in seis]),
            "fault_bytes": _stats([p.stat().st_size for p in fault]),
        }
    return result


def snr_audit(root: Path, seed: int = 42):
    files = sorted((root / "data" / "train" / "seis").glob("*.dat"))
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    snr_values = []
    sigmas = []
    noisy_count = 0
    for path in files:
        image = np.fromfile(path, dtype=np.float32)
        if image.size != np.prod(SHAPE):
            continue
        image = image.reshape(SHAPE).astype(np.float32, copy=False)
        image = (image - image.mean()) / (image.std() + 1e-8)
        if py_rng.random() <= 0.7:
            continue
        noisy_count += 1
        sigma = float(np_rng.uniform(0.01, 0.03))
        noise = np_rng.normal(0.0, sigma, image.shape).astype(np.float32)
        signal_rms = float(np.sqrt(np.mean(np.square(image), dtype=np.float64)))
        noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
        snr_values.append(20.0 * math.log10(signal_rms / noise_rms))
        sigmas.append(sigma)
    return {
        "seed": seed,
        "training_volume_count": len(files),
        "noise_probability": 0.3,
        "noise_std_range": [0.01, 0.03],
        "nominal_snr_db_range": [20.0 * math.log10(1.0 / 0.03), 20.0 * math.log10(1.0 / 0.01)],
        "audited_noisy_volume_count": noisy_count,
        "sampled_noise_std": _stats(sigmas),
        "measured_snr_db": _stats(snr_values),
        "definition": "20*log10(signal_RMS/noise_RMS) after per-volume z-score normalization",
    }


def sliding_geometry(shape, patch=PATCH, overlap=OVERLAP):
    overlap_voxels = tuple(int(n * overlap) for n in patch)
    stride = tuple(n - o for n, o in zip(patch, overlap_voxels))
    padded = tuple(m + 2 * o for m, o in zip(shape, overlap_voxels))
    blocks = tuple(
        int(np.round((m + o) / (n - o) + 0.5))
        for m, n, o in zip(padded, patch, overlap_voxels)
    )
    covered = tuple((n - o) * c + o for n, o, c in zip(patch, overlap_voxels, blocks))
    return {
        "original_shape": list(shape),
        "patch_shape": list(patch),
        "overlap_ratio": overlap,
        "overlap_voxels": list(overlap_voxels),
        "stride_voxels": list(stride),
        "mirror_padding_voxels": list(overlap_voxels),
        "padded_shape": list(padded),
        "blocks_per_axis": list(blocks),
        "covered_shape_before_crop": list(covered),
        "total_windows": int(np.prod(blocks)),
        "fusion": "3D Gaussian mask normalized by accumulated weights",
        "normalization": "per-patch z-score",
    }


def field_audit(root: Path):
    f3 = root / "data" / "prediction" / "f3d" / "gxl.dat"
    f3_shape = (512, 384, 128)
    f3_size_ok = f3.exists() and f3.stat().st_size == int(np.prod(f3_shape)) * 4
    fields = {"f3": {**sliding_geometry(f3_shape), "raw_size_ok": f3_size_ok}}

    kerry_shape = (287, 735, 1252)
    kerry_path = root / "data" / "prediction" / "Kerry3D.segy"
    kerry_record = {**sliding_geometry(kerry_shape), "path_exists": kerry_path.exists()}
    if kerry_path.exists():
        try:
            import segyio

            with segyio.open(str(kerry_path), "r", ignore_geometry=True) as stream:
                kerry_record["trace_count"] = len(stream.trace)
                kerry_record["samples_per_trace"] = len(stream.samples)
                kerry_record["shape_matches_loader"] = (
                    len(stream.trace) == kerry_shape[0] * kerry_shape[1]
                    and len(stream.samples) == kerry_shape[2]
                )
        except Exception as exc:  # keep F3 audit available if SEG-Y metadata fails
            kerry_record["segy_error"] = f"{type(exc).__name__}: {exc}"
    fields["kerry"] = kerry_record
    return fields


def _build_models(device):
    import torch
    from models.MSM_UNet import MSM_UNet
    from compare_models.build import build_model

    wave = MSM_UNet(1, 1).to(device)

    class Config:
        class model:
            name = "SWINUNETR"
            in_chans = 1
            num_classes = 1
            drop = 0.0
            attn_drop = 0.0
            drop_path = 0.0

        class data:
            img_size = (128, 128, 128)

    swin = build_model(Config()).to(device)
    return {"WaveSSM-UNet": wave, "SwinUNETR": swin}


def _profile_flops(model, sample):
    import torch

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_flops=True,
        ) as prof:
            with torch.inference_mode():
                output = model(sample)
                if isinstance(output, tuple):
                    output = output[0]
        del output
        total = sum(float(getattr(event, "flops", 0) or 0) for event in prof.key_averages())
        return {"flops": total, "source": "torch.profiler", "error": None}
    except Exception as exc:
        return {"flops": None, "source": "torch.profiler", "error": f"{type(exc).__name__}: {exc}"}


def benchmark(device_name="cuda", warmup=20, iterations=100, repeats=3):
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(42)
    models = _build_models(device)
    sample = torch.randn((1, 1, *SHAPE), device=device)
    records = {}
    for name, model in models.items():
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(warmup):
                output = model(sample)
                if isinstance(output, tuple):
                    output = output[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
        else:
            baseline = None

        timings = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(iterations):
                    output = model(sample)
                    if isinstance(output, tuple):
                        output = output[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) / iterations * 1000.0)

        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        else:
            peak = peak_reserved = None
        flops = _profile_flops(model, sample)
        records[name] = {
            "input_shape": [1, 1, *SHAPE],
            "params": params,
            "latency_ms": _stats(timings),
            "throughput_volumes_per_second": 1000.0 / statistics.fmean(timings),
            "peak_memory_allocated_bytes": peak,
            "peak_memory_reserved_bytes": peak_reserved,
            "forward_memory_delta_bytes": (peak - baseline) if peak is not None else None,
            "flops": flops,
        }
        del model, output
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "device": str(device),
        "precision": "float32",
        "warmup_iterations": warmup,
        "timed_iterations_per_repeat": iterations,
        "repeats": repeats,
        "models": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reviewer1_phase1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "inventory": inventory(root),
        "snr_audit": snr_audit(root),
        "field_inference": field_audit(root),
    }
    if not args.skip_benchmark:
        report["efficiency_benchmark"] = benchmark(
            args.device, args.warmup, args.iterations, args.repeats
        )
    assert report["field_inference"]["f3"]["stride_voxels"] == [64, 64, 64]
    output_path = args.output / "reviewer1_phase1.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)
    print(json.dumps(report["inventory"], indent=2))
    print(json.dumps(report["snr_audit"], indent=2))
    print(json.dumps(report["field_inference"], indent=2))
    if "efficiency_benchmark" in report:
        print(json.dumps(report["efficiency_benchmark"], indent=2))


if __name__ == "__main__":
    main()
