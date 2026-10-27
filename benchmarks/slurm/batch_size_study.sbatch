#!/usr/bin/env bash
# Held-out accuracy by batch size and learning-rate scaling, on the published
# 1KGP split: one (setting, fold) per array task, published config otherwise.
# Larger batches mean fewer updates per epoch, so the learning rates are scaled
# by sqrt(k) or k for a batch k times the published 138.
#
#   RELEASE=/path/Dietnet_training DIETNET_VENV=/path/.venv REPO=/path/repo OUT=/path/study \
#   sbatch [--account=ACCOUNT] benchmarks/slurm/batch_size_study.sbatch
#
# Then, per setting: dietnet check --packages $OUT/<setting>/exp/packages
#SBATCH --job-name=dietnet_batch_study
#SBATCH --array=0-24
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x.%A_%a.out

set -euo pipefail
: "${RELEASE:?}" "${DIETNET_VENV:?}" "${REPO:?}" "${OUT:?}"
# "batch_size lr_scale"
SETTINGS=("138 1" "552 2" "552 4" "2208 4" "2208 16")
read -r BATCH SCALE <<< "${SETTINGS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}"
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
RUN="$OUT/batch${BATCH}_lr${SCALE}x"
mkdir -p "$RUN"
echo "task $SLURM_ARRAY_TASK_ID: batch $BATCH, lr x$SCALE, fold $FOLD on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

CONFIG="$RUN/config.yaml"
[ -f "$CONFIG" ] || "$DIETNET_VENV/bin/python" - "$RELEASE" "$BATCH" "$SCALE" "$CONFIG" <<'PY'
import sys
import yaml
release, batch, scale, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
cfg = yaml.safe_load(open(f"{release}/supporting_files/config_rep1models.yaml"))
cfg.update(batch_size=batch, lr_aux=cfg["lr_aux"] * scale, lr_main=cfg["lr_main"] * scale)
open(out + ".partial", "w").write(yaml.safe_dump(cfg, sort_keys=False))
import os
os.replace(out + ".partial", out)
PY
SEED=$(grep '^seed:' "$CONFIG" | awk '{print $2}')

start=$(date +%s)
"$DIETNET_VENV/bin/dietnet" train \
    --dataset "$RELEASE/1KGP_dataset.hdf5" \
    --partition "$REPO/reproduction/1kgp_partition_seed124.npz" \
    --embedding "$RELEASE/supporting_files/SNPs_embedding.npz" \
    --input-stats "$RELEASE/supporting_files/normalization_stats.npz" \
    --config "$CONFIG" --exp-dir "$RUN/exp" --seeds "$SEED" --folds "$FOLD" \
    --inputs ram --loader fast --workers 5
echo "trained in $(( $(date +%s) - start )) s"
