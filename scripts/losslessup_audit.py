"""Numerically verify the Channel-to-Space stage of LosslessUp."""

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.MSM_UNet import LosslessUp


def main():
    torch.manual_seed(42)
    module = LosslessUp(4, 4).eval()
    expanded, rearranged = [], []
    module.channel_expand.register_forward_hook(lambda _, __, output: expanded.append(output.detach()))
    module.feature_mix.register_forward_pre_hook(lambda _, args: rearranged.append(args[0].detach()))
    with torch.inference_mode():
        module(torch.randn(1, 4, 7, 6, 5))

    b, _, d, h, w = expanded[0].shape
    restored = rearranged[0].view(b, 4, d, 2, h, 2, w, 2)
    restored = restored.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous().view_as(expanded[0])
    error = (restored - expanded[0]).abs()
    result = {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "relative_l2_error": float(error.norm() / expanded[0].norm()),
        "exact_equal": bool(torch.equal(restored, expanded[0])),
        "input_shape": [1, 4, 7, 6, 5],
        "expanded_shape": list(expanded[0].shape),
        "rearranged_shape": list(rearranged[0].shape),
    }
    output = Path("reviewer1_phase2") / "losslessup_rearrangement.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
