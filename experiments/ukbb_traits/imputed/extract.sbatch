#!/usr/bin/env bash
# The cohort's HapMap3+ genotypes from the imputed data, one chromosome per array
# task (1-22): kept samples (SRC_EXP's inputs/keep.txt), matched variants
# (imputed/ids/chr<c>.ids from hm3_ids.R), MAF >= $MAF (default 0.01) in those
# samples and at most 10% missing calls (hard calls at GP >= 0.9 leave some).
# Writes EXP_DIR/imputed/chr<c>.{bed,bim,fam}.
#SBATCH --job-name=traits_imputed_extract
#SBATCH --array=1-22
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
source "$HERE/common.sh"
: "${IMPUTED:?plink prefix of the imputed genotypes, {chr} for the chromosome}" "${SRC_EXP:?}"
module load StdEnv/2023 plink/2.0.0-a.6.32

c="$SLURM_ARRAY_TASK_ID"
plink2 --bfile "${IMPUTED//\{chr\}/$c}" --keep "$SRC_EXP/inputs/keep.txt" \
    --extract "imputed/ids/chr$c.ids" --maf "${MAF:-0.01}" --geno 0.1 --make-bed \
    --threads "$SLURM_CPUS_PER_TASK" --memory 28000 --out "imputed/chr$c"
