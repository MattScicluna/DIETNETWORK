#!/usr/bin/env bash
# The linear benchmark on the Diet Network's folds (array task k: fold k):
# regenie GWAS (step 1 whole-genome ridge regression, step 2 association) on the
# fold's training samples for every label set, then per label set clumping and
# thresholding: LD clumping on 10,000 training samples, scores at a range of
# p-value thresholds, the threshold chosen on the validation samples, and the
# test samples scored once. Labels are already adjusted for covariates.
# Needs bench/ from benchmark/folds.py. Results: bench/fold<k>/<label>.json.
# CHR=22 (say) restricts every step to one chromosome: a quick end-to-end test.
#SBATCH --job-name=traits_regenie_ct
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
source "$HERE/common.sh"
module load regenie/3.2.1 plink/2.0.0-a.6.32
: "${BENCH_LABELS:=height,bmi}"

k="$SLURM_ARRAY_TASK_ID"
B="bench/fold$k"
threads="$SLURM_CPUS_PER_TASK"
read -ra chr <<< "${CHR:+--chr $CHR}"

regenie --step 1 --bed cohort ${chr[@]+"${chr[@]}"} --keep "$B/train.txt" --phenoFile bench/pheno.tsv \
    --phenoColList "$BENCH_LABELS" --qt --bsize 1000 --lowmem --lowmem-prefix "$B/tmp" \
    --threads "$threads" --out "$B/step1"
regenie --step 2 --bed cohort ${chr[@]+"${chr[@]}"} --keep "$B/train.txt" --phenoFile bench/pheno.tsv \
    --phenoColList "$BENCH_LABELS" --qt --pred "$B/step1_pred.list" --bsize 400 \
    --threads "$threads" --out "$B/gwas"

THRESHOLDS="5e-8 1e-6 1e-4 1e-3 0.01 0.05 0.1 0.2 0.5 1"
for label in ${BENCH_LABELS//,/ }; do
    g="$B/gwas_$label.regenie"
    # ID and P for clumping; ID, effect allele and beta for scoring.
    awk 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; print "ID\tP"; next }
         { print $c["ID"] "\t" 10 ^ (-$c["LOG10P"]) }' "$g" > "$B/$label.p.tsv"
    awk 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
         { print $c["ID"] "\t" $c["ALLELE1"] "\t" $c["BETA"] }' "$g" > "$B/$label.beta.tsv"
    plink2 --bfile cohort ${chr[@]+"${chr[@]}"} --keep "$B/ld.txt" --clump "$B/$label.p.tsv" \
        --clump-p1 1 --clump-r2 0.1 --clump-kb 250 --clump-id-field ID --clump-p-field P \
        --threads "$threads" --out "$B/$label"
    awk 'NR > 1 { print $3 }' "$B/$label.clumps" > "$B/$label.index.txt"
    for t in $THRESHOLDS; do echo "$t 0 $t"; done > "$B/$label.ranges.txt"
    for split in valid test; do
        plink2 --bfile cohort ${chr[@]+"${chr[@]}"} --keep "$B/$split.txt" --extract "$B/$label.index.txt" \
            --score "$B/$label.beta.tsv" 1 2 3 cols=+scoresums \
            --q-score-range "$B/$label.ranges.txt" "$B/$label.p.tsv" header \
            --threads "$threads" --out "$B/$label.$split"
    done
    "$PYTHON" "$HERE/benchmark/ct_select.py" --pheno bench/pheno.tsv --label "$label" \
        --prefix "$B/$label" --thresholds $THRESHOLDS --out "$B/$label.json"
done
