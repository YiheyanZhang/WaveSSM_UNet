<div align="center">

# WaveSSM: An Enhanced Mamba-UNet with Wavelet-Domain Multi-scale Physical Priors for Seismic Fault Detection

[![Paper](https://img.shields.io/badge/Paper-Geophysics-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-brightgreen.svg)]()

*Under review at **Geophysics** (2026)*

</div>

---

## Overview

Seismic fault detection is a fundamental task in subsurface interpretation and hydrocarbon exploration. While deep learning has achieved remarkable progress in this domain, existing CNN-based methods suffer from limited receptive fields that hinder the capture of long-range fault continuity, and most approaches lack explicit integration of physical prior knowledge, resulting in suboptimal interpretability and generalization.

**WaveSSM** (Wavelet State Space Model) addresses these challenges through three synergistic innovations:

1. **Wavelet-domain physical priors** — Multi-scale instantaneous attributes (amplitude, phase, frequency) extracted via continuous wavelet transform (CWT) are deeply fused into the network, providing physically meaningful guidance that regularizes training and enhances noise robustness.

2. **Mamba architecture for long-range modeling** — We introduce the Mamba state space model into seismic fault detection for the first time. Mamba achieves global receptive field modeling with linear time complexity, offering the representational power of Transformers at substantially lower computational cost.

3. **Boundary-aware decoupled supervision** — A dedicated boundary loss function, combined with a decoupled region–boundary prediction head, explicitly constrains fault edge sharpness and continuity.

**Quantitative results on synthetic datasets demonstrate state-of-the-art performance, and experiments on the public F3 and Kerry3D field datasets validate superior cross-domain generalization — all with only 3.33M trainable parameters.**

<p align="center">
  <em>Model architecture overview. See the paper for full details.</em>
</p>

---

## Model Architecture

WaveSSM-UNet adopts a U-shaped encoder–decoder backbone decomposed into three tightly integrated components:

| Component | Description |
|-----------|-------------|
| **Physics-Guided Prior Extraction** | CWT-based front-end that pre-computes multi-scale instantaneous attributes from seismic volumes, forming a 16-channel physically meaningful input. |
| **Mamba-Enhanced U-Net** | Encoder with `ResidualDeformConv` blocks and `WaveletFusion` modules; Bottleneck centered around `FaultOrientedMamba` for tri-directional bidirectional scanning; Decoder with `LosslessUp` (pixel-shuffle) upsampling, `FDEM` (Fault Detail Enhancement Module), `SFAM` (Seismic Fault-aware Attention Mechanism), and `MambaSkip` connections. |
| **Boundary-Aware Decoupled Head (BADH)** | Splits prediction into a region branch and a boundary branch, supervised by a composite loss incorporating BCE, Dice, and a dedicated boundary loss term. |

### Core Modules

- **ResidualDeformConv** — Deformable convolutions for modeling curved and tilted fault geometries
- **WaveletFusion** — Deep fusion of wavelet-domain priors into each encoder stage
- **FaultOrientedMamba** — Tri-directional (depth/inline/crossline) bidirectional Mamba scanning with trainable direction weighting
- **MambaSkip** — Mamba-enhanced skip connections that propagate multi-directional context from encoder to decoder
- **LosslessUp** — Channel-to-space pixel-shuffle upsampling minimizing information loss
- **FDEM** — Laplacian-guided local contrast enhancement for fine fault detail preservation
- **SFAM** — Variance-aware channel + spatial attention for fault-sensitive feature re-calibration

<p align="center">
  <img src="docs/overview.png" alt="WaveSSM-UNet Architecture Overview" width="100%">
  <br><em>Figure 1: Overview of the WaveSSM-UNet model architecture.</em>
</p>

---

## Key Contributions

- ✅ **First application of Mamba to seismic fault detection**, achieving global receptive field with linear complexity (3.33M params vs. ~66M for SwinUNETR)
- ✅ **Wavelet-domain multi-scale physical priors** that regularize learning and enable strong generalization from limited synthetic training data
- ✅ **Boundary-aware loss function** explicitly targeting fault discontinuities for sharper, more accurate fault delineation
- ✅ **State-of-the-art performance** on synthetic benchmarks with superior cross-domain transfer to F3 and Kerry3D field datasets
- ✅ **Exceptional data efficiency** — maintains high quantitative metrics even when trained on only 50 synthetic volumes

---

## Requirements

### Environment

```bash
# Create and activate conda environment (recommended)
conda create -n wavessm python=3.10
conda activate wavessm

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install Mamba SSM
pip install mamba-ssm>=2.2.0

# Install remaining dependencies
pip install -r requirements.txt
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.1 | Deep learning framework |
| mamba-ssm | ≥2.2.0 | State space model backbone |
| NumPy | ≥1.21, <2.0 | Numerical computation |
| SciPy | ≥1.7 | Scientific computing |
| MONAI | ≥1.3.0 | Medical/volumetric image utilities |
| scikit-learn | ≥1.0 | Evaluation metrics |
| Matplotlib | ≥3.4 | Visualization |
| OpenPyXL | ≥3.0 | Excel report generation |

### Hardware

- **GPU**: NVIDIA RTX 4080 SUPER or equivalent (≥16 GB VRAM recommended)
- **RAM**: ≥32 GB system memory
- **Storage**: ≥50 GB free disk space for datasets

---

## Quick Start

### Training

Train WaveSSM-UNet from scratch on synthetic data:

```bash
python main.py \
    --mode train \
    --exp wavessm_experiment \
    --model_type msm_unet \
    --train_path /path/to/train/data/ \
    --valid_path /path/to/validation/data/ \
    --epochs 200 \
    --batch_size 2 \
    --optim_lr 1e-4 \
    --optimizer adamw \
    --lr_scheduler warmup_cosine \
    --loss_func bce+dice \
    --use_ema True \
    --early_stop_patience 20
```

**Key training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `msm_unet` | Model selection: `msm_unet`, `FAULTSEG3D`, `RESUNET`, `RESACEUNET`, `SWINUNETR` |
| `--loss_func` | `bce+dice` | Loss function: `dice`, `bce`, `bce+dice`, `bce_with_weight`, `cross+dice`, `fault_seg` |
| `--optimizer` | `adamw` | Optimizer: `adam`, `adamw` |
| `--lr_scheduler` | `warmup_cosine` | LR schedule: `none`, `cosine`, `warmup_cosine` |
| `--use_ema` | `True` | Exponential moving average of weights |
| `--early_stop_patience` | `50` | Early stopping patience (validation steps) |
| `--grad_accum_steps` | `1` | Gradient accumulation steps |
| `--boundary_loss_weight` | `0.0` | Weight for boundary loss term |
| `--fp16` | `False` | FP16 inference (saves VRAM) |

### Validation Only

Evaluate a trained model on validation data:

```bash
python main.py \
    --mode valid_only \
    --exp wavessm_experiment \
    --model_type msm_unet \
    --valid_path /path/to/validation/data/ \
    --pretrained_model_name MSM_UNet_BEST.pth
```

### Prediction

Run inference with sliding window and Gaussian blending:

```bash
python main.py \
    --mode pred \
    --exp wavessm_experiment \
    --model_type msm_unet \
    --pretrained_model_name MSM_UNet_BEST.pth \
    --pred_data_name f3 \
    --overlap 0.5 \
    --threshold 0.5 \
    --sigma 0.0 \
    --use_sliding_window True
```

**Prediction targets** (`--pred_data_name`): `f3`, `kerry`, `thebe`

### Reproducing Baseline Comparisons

Train and evaluate baseline models for comparison:

```bash
# FaultSeg3D (CNN baseline)
python main.py --mode train --exp baseline_faultseg3d --model_type FAULTSEG3D ...

# ResUNet (CNN baseline)
python main.py --mode train --exp baseline_resunet --model_type RESUNET ...

# ResACEUnet
python main.py --mode train --exp baseline_resaceunet --model_type RESACEUNET ...

# SwinUNETR (Transformer baseline)
python main.py --mode train --exp baseline_swinunetr --model_type SWINUNETR ...
```

Use `--train_ratio` and `--split_seed` for data efficiency ablation studies:

```bash
# Train with only 25% of training data (for data efficiency experiments)
python main.py --mode train --train_ratio 0.25 --split_seed 42 ...
```

---

## Datasets

### Download

All datasets (training, validation, and real seismic prediction data) are provided via Baidu Wangpan:

| Contents | Samples | Directory |
|----------|---------|-----------|
| Training set | ~200 | `train/` |
| Validation set | ~20 | `validation/` |
| F3 real seismic data | — | `prediction/f3/` |
| Kerry3D real seismic data | — | `prediction/kerry3d/` |

> **Download link**: [Baidu Wangpan](https://pan.baidu.com/s/15m7RWQxu86C6Ej3-9RSUoQ) — Extraction code: `ju47`
>
> 220 samples total (train + validation). F3 and Kerry3D prediction data are in the `prediction/` subdirectory.

### Real Seismic Datasets (Original Sources)

| Dataset | Source | Description |
|---------|--------|-------------|
| **Netherlands F3** | [TerraNubis](https://terranubis.com/datainfo/F3-Demo-2023) (dGB Earth Sciences) | Public 3D seismic survey, North Sea |
| **Kerry3D** | [SEG Wiki](https://wiki.seg.org/wiki/Kerry-3D) / [AWS Open Data](http://s3.amazonaws.com/open.source.geoscience/open_data/newzealand/Taranaki_Basin/Keri_3D/Kerry3D.segy) | Public 3D seismic survey, Taranaki Basin, New Zealand |

### Data Preprocessing

All datasets should be preprocessed before training:
1. Convert `.dat` / `.segy` to `.npy` format
2. Apply Z-score normalization (zero mean, unit variance)

---

## Experimental Results

### Quantitative Evaluation (Synthetic Dataset — Full Training Set, 200 Samples)

Our model achieves state-of-the-art performance across all metrics with only **3.33 M** parameters:

| Model | Params | mIoU ↑ | Dice ↑ | F1 ↑ | ROC-AUC ↑ | PR-AUC ↑ |
|-------|--------|--------|--------|------|-----------|----------|
| FaultSeg3D | — | 0.7775 | 0.8624 | 0.7462 | 0.9773 | 0.7877 |
| ResUNet | — | 0.7988 | 0.8796 | 0.7805 | 0.9727 | 0.8395 |
| SwinUNETR | ~66 M | 0.8438 | 0.9094 | 0.8313 | 0.9878 | 0.8957 |
| **WaveSSM-UNet (Ours)** | **3.33 M** | **0.8607** | **0.9203** | **0.8517** | **0.9924** | **0.9107** |

<p align="center">
  <img src="docs/PR_comparison.png" alt="PR Curves" width="45%">
  <img src="docs/ROC_comparison.png" alt="ROC Curves" width="45%">
  <br><em>PR and ROC curves on the full synthetic validation set (200 training samples).</em>
</p>

### Data Efficiency (Synthetic Dataset — 50 Samples)

When trained on only 50 synthetic volumes (25% of the full training set), WaveSSM-UNet retains robust performance while the Transformer baseline degrades severely:

| Model | mIoU ↑ | Dice ↑ | F1 ↑ | ROC-AUC ↑ | PR-AUC ↑ |
|-------|--------|--------|------|-----------|----------|
| SwinUNETR | 0.7255 | 0.8219 | 0.6821 | 0.9561 | 0.7208 |
| **WaveSSM-UNet (Ours)** | **0.8148** | **0.8893** | **0.7945** | **0.9757** | **0.8458** |

The wavelet-domain physical priors enable the model to extract deep fault-related features from extremely limited data, avoiding the overfitting that plagues the highly complex Transformer architecture.

### Visual Comparison — Synthetic Data

<p align="center">
  <em>2D slice at Depth = 50. From left to right: seismic data, ground truth, FaultSeg3D, ResUNet, SwinUNETR, and WaveSSM-UNet (Ours).</em>
</p>

| Seismic | Ground Truth | FaultSeg3D | ResUNet | SwinUNETR | **Ours** |
|---------|-------------|------------|---------|-----------|----------|
| <img src="docs/sample10_depth50_seis.png" width="100%"> | <img src="docs/sample10_depth50_gt.png" width="100%"> | <img src="docs/sample10_depth50_faultseg.png" width="100%"> | <img src="docs/sample10_depth50_resunet.png" width="100%"> | <img src="docs/sample10_depth50_swinunetr.png" width="100%"> | <img src="docs/sample10_depth50_ours.png" width="100%"> |

### Cross-Domain Generalization — F3 Dataset (Netherlands North Sea)

Models trained exclusively on synthetic data, directly applied to the F3 real seismic volume without fine-tuning:

| Seismic Data | FaultSeg3D | SwinUNETR | **Ours** |
|-------------|------------|-----------|----------|
| <img src="docs/f3_3d_seis.png" width="100%"> | <img src="docs/f3_3d_faultseg.png" width="100%"> | <img src="docs/f3_3d_swinunetr.png" width="100%"> | <img src="docs/f3_3d_ours.png" width="100%"> |

### Cross-Domain Generalization — Kerry3D Dataset (Taranaki Basin, New Zealand)

| Seismic Data | FaultSeg3D | SwinUNETR | **Ours** |
|-------------|------------|-----------|----------|
| <img src="docs/kerry_3d_seis.png" width="100%"> | <img src="docs/kerry_3d_faultseg.png" width="100%"> | <img src="docs/kerry_3d_swinunetr.png" width="100%"> | <img src="docs/kerry_3d_ours.png" width="100%"> |

### Ablation Study

Ablation experiments confirm the contribution of each component:

| Configuration | Effect |
|---------------|--------|
| w/o Wavelet Priors | Degraded generalization, increased noise sensitivity |
| w/o FaultOrientedMamba | Reduced fault continuity, fragmented predictions |
| w/o MambaSkip | Loss of fine structural detail in decoder |
| w/o Boundary Loss | Blurred fault edges, reduced boundary sharpness |

<p align="center">
  <em>Ablation study on the F3 dataset. Removing any component degrades prediction quality. Full model (rightmost) achieves the best continuity and boundary sharpness.</em>
</p>

| w/o Wavelet | w/o FaultOrientedMamba | w/o MambaSkip | w/o Boundary Loss | **Full Model** |
|-------------|------------------------|---------------|-------------------|----------------|
| <img src="docs/abulation_f3_nowavelet.png" width="100%"> | <img src="docs/abulation_f3_nofaultorientedmamba.png" width="100%"> | <img src="docs/abulation_f3_nomambaskip.png" width="100%"> | <img src="docs/abulation_f3_noboundaryloss.png" width="100%"> | <img src="docs/abulation_f3_full.png" width="100%"> |

*Full ablation study: see Figures 23–24 in the paper.*

---

## Project Structure

```
WaveSSM/
├── main.py                  # Entry point: training, validation, prediction
├── models/                  # Model definitions
│   ├── wavelet_analyze.py   # CWT-based physical prior extraction
│   └── faultseg3d.py        # WaveSSM-UNet + Mamba modules
├── compare_models/          # Baseline model implementations
│   ├── build.py             # Model factory
│   ├── ResUnet.py           # ResUNet
│   ├── UNet3.py             # 3D UNet
│   ├── ResACEUnet.py        # ResACEUNet
│   └── SwinUNETR.py         # SwinUNETR (Transformer baseline)
├── dataloader/              # Data loading pipeline
├── utils/                   # Training, evaluation, and utility functions
│   ├── train.py             # Training loop + EMA + early stopping
│   ├── test.py              # Sliding window prediction + metrics
│   ├── tools.py             # Helper utilities
│   └── dice_loss.py         # Loss functions
├── configs/                 # Configuration files
├── scripts/                 # Ablation and comparison scripts
│   └── run_ratio_ablation.sh
├── C3/                      # C3 coherence algorithm
├── generate_paper_figures.py
├── plot_pr_roc_compare.py   # PR/ROC curve plotting
├── requirements.txt
└── LICENSE
```

---

## Citation

If you use WaveSSM in your research, please cite our work:

```bibtex
@article{waveSSM2026,
  title   = {An Enhanced Mamba-UNet with Wavelet-Domain Multi-scale Physical
             Priors for Seismic Fault Detection},
  author  = {},
  journal = {Geophysics},
  year    = {2026},
  note    = {Under review}
}
```

---

## Attribution

This work builds upon and gratefully acknowledges the following open-source projects and public datasets:

- **FaultSeg3D** — [Xinming Wu et al. (2019)](https://github.com/xinwucwp/faultSeg). Synthetic training data and original fault segmentation framework.
- **FaultSeg3D_pytorch** — [Chaofan Ke (Ifjmww)](https://github.com/Ifjmww/FaultSeg3D_pytorch). PyTorch re-implementation serving as the codebase foundation.
- **ResACEUnet** — [Chaofan Ke](https://github.com/39c5bb-miku/ResACEUnet). Baseline implementations of SwinUNETR and ResUNet.
- **Netherlands F3 dataset** — [dGB Earth Sciences](https://terranubis.com/datainfo/F3-Demo-2023). Public 3D seismic survey.
- **Kerry3D dataset** — New Zealand Crown Minerals, via [SEG Wiki](https://wiki.seg.org/wiki/Kerry-3D) and [AWS Open Data](http://s3.amazonaws.com/open.source.geoscience/open_data/newzealand/Taranaki_Basin/Keri_3D/Kerry3D.segy).

If you use or reference this code in your project, we appreciate an attribution statement:

```text
Parts of this code are based on modifications of Ifjmww's FaultSeg3D_pytorch.
Original project link: https://github.com/Ifjmww/FaultSeg3D_pytorch
```

---

## License

This project is released under the **MIT License**. See [LICENSE](./LICENSE) for details.

---

## Contact

For questions, please open an issue on this repository or contact the corresponding author.
