#!/usr/bin/env bash
# One dataset per label set (array task i: the i-th of LABELS), and its 2-bit
# copy for training. All are built from the same fileset, so their rows match.
# create-dataset reads the whole .bed into memory (51.6 GB for 424,768 x 485,516;
# memory-mapping it from Lustre is far slower), plus ~1 GiB blocks: 80G.
#SBATCH --job-name=traits_dataset
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
source "$HERE/common.sh"

name="${LABEL_SETS[$SLURM_ARRAY_TASK_ID]%%=*}"
mkdir -p data
"$DIETNET" create-dataset --plink cohort --labels inputs/classes.tsv \
    --regression-labels "inputs/labels/$name.tsv" --out "data/$name.h5"
"$DIETNET" pack-genotypes --dataset "data/$name.h5"
