# AutoDL reproduction and CIG-Bench comparison

The experiments used CIG-Bench `0.2.0`.
Run every command below from the WaveSSM-UNet repository root.

## 1. Environment and data

Use an AutoDL PyTorch image compatible with the installed CUDA driver, then run:

```bash
pip install -r requirements.txt
pip install cig-bench==0.2.0 modelscope
```

The synthetic data must keep the existing layout and use matching filenames:

```text
data/
  train/seis/*.dat
  train/fault/*.dat
  validation/seis/*.dat
  validation/fault/*.dat
```

Each file is a float32 `128 x 128 x 128` volume. The paper uses 200 training
volumes and 20 validation volumes. F3 and Kerry3D keep the paths documented in
the project README (`data/prediction/f3d/` and
`data/prediction/Kerry3D.segy`).

## 2. Retrain WaveSSM-UNet with the paper settings

The defaults now match the manuscript: AdamW, 200 epochs, learning rate `1e-4`,
early-stopping patience 20, Dice/Focal/Boundary weights `0.7/0.3/0.3`, Focal
`alpha=0.75`, `gamma=2`, constant learning rate, and no EMA.

```bash
python main.py \
  --mode train \
  --exp wavessm_paper_retrain \
  --train_path data/train \
  --valid_path data/validation \
  --device cuda \
  --batch_size 2 \
  --batch_size_not_train 1 \
  --workers 0 \
  --seed 42
```

The selected weight is:

```text
EXP/wavessm_paper_retrain/models/MSM_UNet_BEST.pth
```

An epoch checkpoint is also written every 10 epochs. Copy the whole model
directory out of the disposable system disk immediately after training:

```bash
mkdir -p /root/autodl-tmp/checkpoints/wavessm_paper_retrain
cp -a EXP/wavessm_paper_retrain/models/. \
  /root/autodl-tmp/checkpoints/wavessm_paper_retrain/
```

Validate the saved weight before publishing it:

```bash
python main.py \
  --mode valid_only \
  --exp wavessm_paper_retrain \
  --valid_path data/validation \
  --pretrained_model_name MSM_UNet_BEST.pth \
  --device cuda
```

## 3. CIG-Bench experiments

The reported comparison uses the public checkpoint:

- **CIG-Bench (public)**: the checkpoint published by CIG-Bench, evaluated
  zero-shot.
- **CIG-Bench (retrained)** is an optional controlled experiment using the same
  200/20 split and region loss as the paper. CIG-Bench has no boundary head, so
  its loss is `0.7 Dice + 0.3 Focal`; do not claim it used boundary supervision.

Train the controlled baseline:

```bash
python cig_bench_experiment.py train \
  --exp cig_bench_retrained \
  --train_path data/train \
  --valid_path data/validation \
  --device cuda \
  --epochs 200 \
  --lr 1e-4 \
  --batch_size 2 \
  --patience 20 \
  --workers 0 \
  --seed 42
```

It saves `CIG_Bench_BEST.pth`, `CIG_Bench_LAST.pth`, and a checkpoint every 10
epochs in `EXP/cig_bench_retrained/models/`.

Evaluate the public checkpoint:

```bash
python cig_bench_experiment.py eval \
  --exp cig_bench_public \
  --valid_path data/validation \
  --weights public
```

`--weights public` downloads the official fault checkpoint through CIG-Bench's
ModelScope registry. Each evaluation writes `metrics.json`, `curves.npz`, and
per-sample image/label/probability arrays under `EXP/<exp>/results/valid/`.
The table metrics are dataset-level mIoU, mean Dice, best-threshold F1, ROC-AUC,
and PR-AUC. Curve sampling is deterministic (`seed=42`, at most one million
voxels), matching the existing comparison script.

Regenerate only the PR/ROC comparison charts:

```bash
python plot_pr_roc_compare.py \
  --models \
    "Ours:msm_unet:EXP/wavessm_paper_retrain/models/MSM_UNet_BEST.pth" \
    "CIG-Bench(public):CIG_BENCH:public" \
  --valid_path data/validation \
  --out_dir EXP/wavessm_paper_retrain/figures/curves_with_cig \
  --device cuda \
  --seed 42
```

## 4. Qualitative CIG-Bench panels

The following commands run field-data inference with the public checkpoint:

```bash
python cig_bench_experiment.py predict --exp cig_bench_public \
  --dataset f3 --weights public --device cuda

python cig_bench_experiment.py predict --exp cig_bench_public \
  --dataset kerry --weights public --device cuda
```

The default `--scale 1.0` preserves resolution. If Kerry3D still exceeds GPU
memory with rank 4, rerun only that command with `--scale 0.5` and disclose the
rescaling in the experiment settings.

Generate only the new CIG-Bench PNG panels:

```bash
python generate_cig_bench_figures.py --exp cig_bench_public --mode all
```

Outputs are placed in `EXP/cig_bench_public/figures/cig_bench/` with the same
300-dpi style, views and locations as the existing paper figures:

```text
valid_3d_sample0_cigbench.png
sample10_depth50_cigbench.png
sample11_inline50_cigbench.png
sample12_crossline50_cigbench.png
f3_3d_cigbench.png
f3_time100_cigbench.png
kerry_3d_cigbench.png
kerry_time215_cigbench.png
kerry_time220_cigbench.png
kerry_time225_cigbench.png
```

Keep public and retrained qualitative panels in separate experiment directories;
never label a retrained prediction as the public CIG-Bench model.
