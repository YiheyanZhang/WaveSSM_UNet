"""
run_c3.py
Whole-volume C3 (Gersztenkorn & Marfurt 1999) eigenstructure coherence baseline.

Algorithm reimplemented from d2geo (Fitz-Gerald 2018), attributes/EdgeDetection.py::eig_complex,
with the following changes:
  - bug fix: the d2geo source shadows the inner function `cov` by assigning
    `cov = np.apply_along_axis(cov, ...)` inside `operation()`, which raises
    UnboundLocalError on every call. We compute the covariance directly per chunk.
  - eigvalsh instead of eigvals: the covariance is real symmetric, so eigvalsh
    is faster and numerically stable, and we don't need np.abs() on the result.
  - drop dask: F3 fits in RAM; Kerry runs in numpy chunks along axis 0.
  - drop the 128^3 sliding window / Gaussian stitching / pad=64 from the OSV-style
    pipeline: C3's analysis window is only 3x3x9, so it doesn't need patching.

Pipeline:
  1) load volume
  2) hilbert(.) along the time axis -> analytic trace (complex64)
  3) reflect pad (J//2, K//2, L//2) for the 3x3x9 analysis window only
  4) for each chunk along axis 0:
       w = sliding_window_view(padded, (J,K,L))[i0:i1]    # zero-copy view
       w = w.reshape(chunk, D1, D2, J*K, L)               # actual copy here
       X = concat([Re(w), Im(w)], axis=-1)                # d2geo: complex -> real (..., JK, 2L)
       C = X X^T                                          # (..., JK, JK), real symmetric
       coh = eigvalsh(C)[..., -1] / trace(C)
       fault[i0:i1] = 1 - coh
  5) percentile [1%, 99.5%] stretch to [0,1]   (paper-figure vmin=0.4 compatible)
  6) save EXP/c3/results/pred/{name}/numpy/{name}.npy

Per-patch z-score is omitted: eigenstructure coherence is scale-invariant
(scaling all traces by alpha scales C by alpha^2, so lam_max/trace is unchanged).

Usage (run from WaveSSM root):
  python C3/run_c3.py --dataset f3
  python C3/run_c3.py --dataset kerry
"""
import os
import argparse
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import hilbert
from tqdm import tqdm


# C3 analysis window: J,K small in lateral, L longer along time. 3x3x9 = Gersztenkorn 1999 default.
J_WIN, K_WIN, L_WIN = 3, 3, 9

F3_SEIS  = "data/prediction/f3d/gxl.dat"
F3_SHAPE = (512, 384, 128)            # (Crossline, Inline, Time)
KERRY_SEGY  = "data/prediction/Kerry3D.segy"
KERRY_SHAPE = (287, 735, 1252)        # (Inline, Xline, Sample)

OUT_ROOT = "EXP/c3/results/pred"


def load_f3():
    print("[F3] loading", F3_SEIS)
    arr = np.fromfile(F3_SEIS, dtype=np.float32).reshape(F3_SHAPE)
    print("    shape:", arr.shape, "(Cross, Inline, Time)")
    return arr


def load_kerry():
    print("[Kerry] loading", KERRY_SEGY)
    import segyio
    with segyio.open(KERRY_SEGY, "r", ignore_geometry=True) as f:
        data = np.stack([f.trace[i] for i in range(len(f.trace))], axis=0)
    arr = data.reshape(KERRY_SHAPE).astype(np.float32)
    print("    shape:", arr.shape, "(Inline, Xline, Sample)")
    return arr


def compute_c3_volume(vol, J=J_WIN, K=K_WIN, L=L_WIN, chunk=8):
    """Whole-volume C3 (eigenstructure coherence on analytic trace).

    vol: (D0, D1, D2) float32, with D2 == time axis.
    Returns fault = 1 - coherence in [0, 1], shape (D0, D1, D2), float32.
    """
    pj, pk, pl = J // 2, K // 2, L // 2
    D0, D1, D2 = vol.shape

    print("    hilbert along time axis ...")
    analytic = hilbert(vol, axis=-1).astype(np.complex64)

    print("    reflect pad +(%d,%d,%d) for %dx%dx%d analysis window ..."
          % (pj, pk, pl, J, K, L))
    padded = np.pad(analytic, ((pj, pj), (pk, pk), (pl, pl)), mode="reflect")
    del analytic

    JK = J * K
    fault = np.empty((D0, D1, D2), dtype=np.float32)

    # zero-copy stride view over padded: (D0, D1, D2, J, K, L)
    windows = sliding_window_view(padded, (J, K, L))

    pbar = tqdm(total=D0, desc="    C3", unit="slice")
    for i0 in range(0, D0, chunk):
        i1 = min(i0 + chunk, D0)
        # touching windows[i0:i1] forces the (chunk, D1, D2, J, K, L) materialization
        w = windows[i0:i1].reshape(i1 - i0, D1, D2, JK, L)
        # d2geo eig_complex: stack [Re, Im] so analytic-trace covariance is real symmetric
        X = np.concatenate([w.real, w.imag], axis=-1)               # (..., JK, 2L)
        del w
        C = np.einsum("...il,...jl->...ij", X, X, optimize=True)    # (..., JK, JK)
        del X
        evals = np.linalg.eigvalsh(C)                               # ascending
        lam_max = evals[..., -1]
        tr = np.einsum("...ii->...", C)
        coh = lam_max / (tr + 1e-12)
        fault[i0:i1] = (1.0 - coh).astype(np.float32)
        pbar.update(i1 - i0)
    pbar.close()
    return fault


def run(name):
    if name == "f3":
        vol = load_f3()
        chunk = 8
    else:
        vol = load_kerry()
        chunk = 2          # Kerry: D1*D2 = 735*1252, keep per-chunk tmp tensors modest

    fault = compute_c3_volume(vol, chunk=chunk)
    del vol

    p_lo = float(np.percentile(fault, 1.0))
    p_hi = float(np.percentile(fault, 99.5))
    fault = np.clip((fault - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0).astype(np.float32)
    print("    percentile stretch [%.4f, %.4f] -> [0,1]" % (p_lo, p_hi))
    print("    fault range: [%.4f, %.4f]  mean=%.4f" %
          (float(fault.min()), float(fault.max()), float(fault.mean())))

    out_dir = os.path.join(OUT_ROOT, name, "numpy")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "%s.npy" % name)
    np.save(out_path, fault)
    print("    saved -> %s  shape=%s" % (out_path, fault.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["f3", "kerry"])
    args = ap.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
