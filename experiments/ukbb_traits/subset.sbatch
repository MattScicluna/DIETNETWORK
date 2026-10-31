#!/usr/bin/env bash
# The genotypes of the kept samples (inputs/keep.txt) as their own fileset.
#SBATCH --job-name=traits_subset
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
source "$HERE/common.sh"
module load plink/2.0.0-a.6.32

plink2 --bfile "$GENOTYPES" --keep inputs/keep.txt --make-bed --out cohort \
    --threads "$SLURM_CPUS_PER_TASK" --memory 28000
