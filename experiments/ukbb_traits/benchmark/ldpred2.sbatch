#!/usr/bin/env bash
# The joint linear benchmark on the Diet Network's folds (array task k: fold k):
# LDpred2-auto (bigsnpr) on the fold's regenie GWAS (training samples only, from
# regenie_ct.sbatch), with in-sample LD from the fold's 10,000-sample LD reference,
# then the validation and test samples scored with plink2 (SCORE1_SUM).
# Results: bench/fold<k>/<label>.ldpred2.json.
# CHR=22 (say) restricts every step to one chromosome, and tags the outputs
# (<label>.ldpred2.chr22.*): a quick end-to-end test.
# GWAS=plink2 uses a plain marginal GWAS instead (plink2 --glm on the training
# samples, no LOCO adjustment; gwas_marginal_<label>.regenie), tagged .plink:
# regenie's step 2 tests y minus a polygenic prediction, which is not the
# marginal association LDpred2 models.
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
: "${GWAS:=regenie}"
case "$GWAS" in
regenie) gwas_prefix=gwas tag="${CHR:+.chr$CHR}" ;;
plink2) gwas_prefix=gwas_marginal tag=".plink${CHR:+.chr$CHR}" ;;
*) echo "GWAS must be regenie or plink2, not $GWAS" >&2; exit 1 ;;
esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # bigsnpr parallelises itself

module load StdEnv/2023 plink/2.0.0-a.6.32
if [ "$GWAS" = plink2 ]; then
    plink2 --bfile cohort ${chr[@]+"${chr[@]}"} --keep "$B/train.txt" --pheno bench/pheno.tsv \
        --pheno-name "$BENCH_LABELS" --glm allow-no-covars \
        cols=chrom,pos,omitted,a1freq,test,nobs,beta,se,p --threads "$threads" \
        --out "$B/marginal"
    for label in ${BENCH_LABELS//,/ }; do
        # regenie's columns: BETA per ALLELE1 (plink2's A1), ALLELE0 the other allele.
        awk 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i
                       print "CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ N TEST BETA SE"; next }
             $c["TEST"] == "ADD" { print $c["#CHROM"], $c["POS"], $c["ID"], $c["OMITTED"], $c["A1"],
                 $c["A1_FREQ"], $c["OBS_CT"], "ADD", $c["BETA"], $c["SE"] }' \
            "$B/marginal.$label.glm.linear" > "$B/gwas_marginal_$label.regenie"
    done
fi
plink2 --bfile cohort ${chr[@]+"${chr[@]}"} --keep "$B/ld.txt" --make-bed \
    --threads "$threads" --out "$SLURM_TMPDIR/ld"

module load gcc r/4.4.0
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/x86_64-pc-linux-gnu-library/4.4}"
Rscript "$HERE/benchmark/ldpred2.R" "$SLURM_TMPDIR/ld.bed" "$B" "$BENCH_LABELS" "$tag" \
    "$threads" "$SLURM_TMPDIR" "$gwas_prefix"

for label in ${BENCH_LABELS//,/ }; do
    p="$B/$label.ldpred2$tag"
    for split in valid test; do
        plink2 --bfile cohort ${chr[@]+"${chr[@]}"} --keep "$B/$split.txt" \
            --score "$p.betas.tsv" 1 2 3 header cols=+scoresums \
            --threads "$threads" --out "$p.$split"
    done
    "$PYTHON" "$HERE/benchmark/score_eval.py" --pheno bench/pheno.tsv --label "$label" \
        --prefix "$p" --method "$GWAS GWAS + LDpred2-auto" --out "$p.json"
done
