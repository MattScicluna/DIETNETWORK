#!/usr/bin/env bash
# Per-fold SNP means and SDs, one fold per array task; merge_stats.sbatch joins them.
#SBATCH --job-name=traits_stats
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=04:00:00
source "$HERE/common.sh"

k="$SLURM_ARRAY_TASK_ID"
"$DIETNET" compute-input-stats --dataset "data/$FIRST.h5" --partition "partition_fold$k.npz" \
    --out "stats_fold$k.npz"
