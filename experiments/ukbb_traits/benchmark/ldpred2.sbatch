#!/usr/bin/env bash
# The joint linear benchmark on the Diet Network's folds (array task k: fold k):
# LDpred2-auto (bigsnpr) on the fold's regenie GWAS (training samples only, from
# regenie_ct.sbatch), with in-sample LD from the fold's 10,000-sample LD reference,
# then the validation and test samples scored with plink2 (SCORE1_SUM).
# Results: bench/fold<k>/<label>.ldpred2.json.
# CHR=22 (say) restricts every step to one chromosome, and tags the outputs
# (<label>.ldpred2.chr22.*): a quick end-to-end test.
#SBATCH --job-name=traits_ldpred2
#SBATCH --array=0-4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
source "$HERE/common.sh"
: "${BENCH_LABELS:=height,bmi}"

k="$SLURM_ARRAY_TASK_ID"
B="bench/fold$k"
threads="$SLURM_CPUS_PER_TASK"
read -ra chr <<< "${CHR:+--chr $CHR}"
tag="${CHR:+.chr$CHR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # bigsnpr parallelises itself

module load StdEnv/2023 plink/2.0.0-a.6.32
plink2 --bfile cohort ${chr[@]+"${chr[@]}"} --keep "$B/ld.txt" --make-bed \
    --threads "$threads" --out "$SLURM_TMPDIR/ld"

module load gcc r/4.4.0
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/x86_64-pc-linux-gnu-library/4.4}"
Rscript "$HERE/benchmark/ldpred2.R" "$SLURM_TMPDIR/ld.bed" "$B" "$BENCH_LABELS" "$tag" \
    "$threads" "$SLURM_TMPDIR"

for label in ${BENCH_LABELS//,/ }; do
    p="$B/$label.ldpred2$tag"
    for split in valid test; do
        plink2 --bfile cohort ${chr[@]+"${chr[@]}"} --keep "$B/$split.txt" \
            --score "$p.betas.tsv" 1 2 3 header cols=+scoresums \
            --threads "$threads" --out "$p.$split"
    done
    "$PYTHON" "$HERE/benchmark/score_eval.py" --pheno bench/pheno.tsv --label "$label" \
        --prefix "$p" --method "regenie + LDpred2-auto" --out "$p.json"
done
