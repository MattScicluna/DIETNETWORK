#!/usr/bin/env bash
# The genotypes of the kept samples (inputs/keep.txt) as their own fileset, SNPs
# with MAF >= $MAF (default 0.01) in those samples only. Rare variants break the
# inputs' standardisation: on the full UK Biobank array, 895 SNPs had no carrier
# among a fold's training samples (SD 1e-6, so a validation carrier's input was
# ~1e6) and 18,970 had SD <= 0.01, and every model's validation loss blew up.
#SBATCH --job-name=traits_subset
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
source "$HERE/common.sh"
module load plink/2.0.0-a.6.32

plink2 --bfile "$GENOTYPES" --keep inputs/keep.txt --maf "${MAF:-0.01}" --make-bed --out cohort \
    --threads "$SLURM_CPUS_PER_TASK" --memory 28000
