#!/usr/bin/env bash
# One published (seed, fold) per array task: 15 tasks, seeds 78-80 x folds 0-4,
# trained by upstream's train.py through `dietnet train` with the published config.
#SBATCH --job-name=dietnet_repro_train
#SBATCH --array=0-14
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=%x.%A_%a.out

set -euo pipefail
: "${RELEASE:?}" "${DIETNET_VENV:?}" "${EXP_DIR:?}" "${REPO:?}"
cd "$EXP_DIR"

REP=$(( SLURM_ARRAY_TASK_ID / 5 + 1 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
CONFIG="configs/config_rep${REP}models.yaml"
SEED=$(grep '^seed:' "$CONFIG" | awk '{print $2}')
echo "task $SLURM_ARRAY_TASK_ID: replicate $REP (seed $SEED), fold $FOLD on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$DIETNET_VENV/bin/dietnet" train \
    --dataset "$RELEASE/1KGP_dataset.hdf5" \
    --partition "$REPO/reproduction/1kgp_partition_seed124.npz" \
    --embedding embedding.npz --input-stats stats.npz \
    --config "$CONFIG" --exp-dir exp --seeds "$SEED" --folds "$FOLD"
