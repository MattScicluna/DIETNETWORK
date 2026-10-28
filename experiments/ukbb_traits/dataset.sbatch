#!/usr/bin/env bash
# One dataset per label set (array task i: the i-th of LABELS), and its 2-bit
# copy for training. All are built from the same fileset, so their rows match.
#SBATCH --job-name=traits_dataset
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
source "$HERE/common.sh"

name="${LABEL_SETS[$SLURM_ARRAY_TASK_ID]%%=*}"
mkdir -p data
"$DIETNET" create-dataset --plink cohort --labels inputs/classes.tsv \
    --regression-labels "inputs/labels/$name.tsv" --out "data/$name.h5"
"$DIETNET" pack-genotypes --dataset "data/$name.h5"
