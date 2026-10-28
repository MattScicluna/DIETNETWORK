#!/usr/bin/env bash
# Per-fold SNP means and SDs by upstream's script, one fold per array task.
# The script loads the fold's training genotypes into memory (~256k x 485k int8;
# 124 GB peak at full shape), then takes 10 min to 2 h a fold (Mila, simulated):
# folds run side by side rather than one after another; merge_stats.sbatch
# joins them.
#SBATCH --job-name=traits_stats
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=04:00:00
source "$HERE/common.sh"

k="$SLURM_ARRAY_TASK_ID"
"$DIETNET" compute-input-stats --dataset "data/$FIRST.h5" --partition "partition_fold$k.npz" \
    --out "stats_fold$k.npz" --ncpus "$SLURM_CPUS_PER_TASK"
