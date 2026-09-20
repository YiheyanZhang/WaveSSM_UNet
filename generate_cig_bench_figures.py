"""Generate only the CIG-Bench PNG panels used by the paper."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_paper_figures import (SINGLE_COL, draw_3d_slices,
                                    overlay_fault_on_slice, save_3d_figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="cig_bench")
    parser.add_argument("--mode", choices=("synthetic", "f3", "kerry", "all"), default="all")
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def save_3d(seismic, fault, path, **kwargs):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    draw_3d_slices(ax, seismic, fault, "", **kwargs)
    save_3d_figure(path, dpi=300)


def save_slice(seismic, fault, axis, position, path, field=False):
    depth, height, width = seismic.shape
    if axis == 0:
        image, mask = seismic[position], fault[position]
        x_label, y_label = "Crossline", "Inline"
        extent = [0, width, height, 0] if field else None
    elif axis == 1:
        image, mask = seismic[:, position, :], fault[:, position, :]
        x_label, y_label = "Crossline", "Time (ms)"
        extent = [0, width, depth * 4, 0]
    else:
        image, mask = seismic[:, :, position], fault[:, :, position]
        x_label, y_label = "Inline", "Time (ms)"
        extent = [0, height, depth * 4, 0]

    if field:
        height, width = image.shape
        size = (SINGLE_COL, SINGLE_COL * height / width)
    else:
        size = (4, 4)
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(overlay_fault_on_slice(image, mask), origin="upper",
              aspect="equal" if field else "auto", extent=extent)
    ax.set_xlabel(x_label, fontsize=9 if field else 12)
    ax.set_ylabel(y_label, fontsize=9 if field else 12)
    ax.tick_params(labelsize=7 if field else 11)
    if not field:
        plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0 if field else 0.1)
    plt.close()


def synthetic(exp_dir, output):
    source = exp_dir / "results" / "valid" / "numpy"
    save_3d(np.load(source / "0_img.npy"), np.load(source / "0_seg.npy"),
            output / "valid_3d_sample0_cigbench.png", time_dt=4)
    selections = {
        10: (0, 50, "depth"),
        11: (1, 50, "inline"),
        12: (2, 50, "crossline"),
    }
    for sample, (axis, position, name) in selections.items():
        save_slice(np.load(source / f"{sample}_img.npy"),
                   np.load(source / f"{sample}_seg.npy"), axis, position,
                   output / f"sample{sample}_{name}{position}_cigbench.png")


def field(exp_dir, output, name):
    source = exp_dir / "results" / "pred" / name / "numpy"
    seismic = np.load(source / f"{name}_img.npy")
    fault = np.load(source / f"{name}.npy")
    if name == "f3":
        save_3d(seismic, fault, output / "f3_3d_cigbench.png", time_dt=4)
        save_slice(seismic, fault, 0, 100, output / "f3_time100_cigbench.png", field=True)
    else:
        save_3d(seismic, fault, output / "kerry_3d_cigbench.png",
                slice_indices=(400, 237, 535), time_dt=4, elev=20, azim=-60)
        for position in (215, 220, 225):
            save_slice(seismic, fault, 0, position,
                       output / f"kerry_time{position}_cigbench.png", field=True)


def main():
    args = parse_args()
    exp_dir = Path("EXP") / args.exp
    output = Path(args.out_dir) if args.out_dir else exp_dir / "figures" / "cig_bench"
    output.mkdir(parents=True, exist_ok=True)
    if args.mode in ("synthetic", "all"):
        synthetic(exp_dir, output)
    if args.mode in ("f3", "all"):
        field(exp_dir, output, "f3")
    if args.mode in ("kerry", "all"):
        field(exp_dir, output, "kerry")
    print(f"[Saved] {output}")


if __name__ == "__main__":
    main()
