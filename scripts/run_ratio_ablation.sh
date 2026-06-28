#!/bin/bash
# 训练集/验证集比例消融实验脚本
# 在多个 train_ratio 下训练多个模型，使用同一 split_seed 保证公平比较。
#
# 用法（在 WaveSSM 根目录下运行）:
#   bash scripts/run_ratio_ablation.sh
#
# 自定义（环境变量覆盖）:
#   RATIOS="0.6 0.8" MODELS="msm_unet RESUNET" bash scripts/run_ratio_ablation.sh
#   EPOCHS=50 SEED=2026 bash scripts/run_ratio_ablation.sh
#   DRY_RUN=1 bash scripts/run_ratio_ablation.sh        # 只打印命令，不执行

set -euo pipefail

# ------------------------- 可配置项 -------------------------
RATIOS="${RATIOS:-0.1 0.3 0.5 0.7 0.9}"
MODELS="${MODELS:-msm_unet RESUNET SWINUNETR}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
LOSS_FUNC="${LOSS_FUNC:-bce+dice}"
TRAIN_PATH="${TRAIN_PATH:-/root/autodl-tmp/WaveSSM/data/train/}"
VALID_PATH="${VALID_PATH:-/root/autodl-tmp/WaveSSM/data/validation/}"
LOG_DIR="${LOG_DIR:-EXP/ratio_ablation_logs}"
DRY_RUN="${DRY_RUN:-0}"
# -----------------------------------------------------------

mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/summary_${TS}.txt"
echo "ratio_ablation run started at $(date)" | tee "$SUMMARY"
echo "ratios : $RATIOS"   | tee -a "$SUMMARY"
echo "models : $MODELS"   | tee -a "$SUMMARY"
echo "epochs : $EPOCHS"   | tee -a "$SUMMARY"
echo "seed   : $SEED"     | tee -a "$SUMMARY"
echo "----------------------------------------" | tee -a "$SUMMARY"

total=0
fail=0

for r in $RATIOS; do
    for m in $MODELS; do
        # exp 名小写化，避免大小写敏感的目录混淆
        m_lc=$(echo "$m" | tr '[:upper:]' '[:lower:]')
        exp_name="${m_lc}_r${r}_s${SEED}"
        log_file="$LOG_DIR/${exp_name}_${TS}.log"

        cmd=(python main.py
            --exp           "$exp_name"
            --mode          train
            --model_type    "$m"
            --train_path    "$TRAIN_PATH"
            --valid_path    "$VALID_PATH"
            --train_ratio   "$r"
            --split_seed    "$SEED"
            --epochs        "$EPOCHS"
            --batch_size    "$BATCH_SIZE"
            --optim_lr      "$LR"
            --loss_func     "$LOSS_FUNC"
        )

        echo ""
        echo "[$((total+1))] >>> $exp_name"
        echo "    log: $log_file"
        echo "    cmd: ${cmd[*]}"

        if [ "$DRY_RUN" = "1" ]; then
            total=$((total+1))
            continue
        fi

        start_ts=$(date +%s)
        if "${cmd[@]}" 2>&1 | tee "$log_file"; then
            status="OK"
        else
            status="FAIL"
            fail=$((fail+1))
        fi
        elapsed=$(( $(date +%s) - start_ts ))

        printf "%-40s %-6s %ds\n" "$exp_name" "$status" "$elapsed" | tee -a "$SUMMARY"
        total=$((total+1))
    done
done

echo "----------------------------------------" | tee -a "$SUMMARY"
echo "done: $total runs, $fail failed"          | tee -a "$SUMMARY"
echo "summary saved to $SUMMARY"

[ "$fail" -eq 0 ] || exit 1
