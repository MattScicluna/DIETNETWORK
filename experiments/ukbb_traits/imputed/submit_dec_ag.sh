#!/usr/bin/env bash
# AlphaGenome as a prior on the HM3+ cohort: merge the tables, build the dec_ag embeddings,
# then train dec_ag and penalized dec_ag (lambda 0.3), 5 folds, both traits, with the
# recipe of the dec runs they are compared against. Each step waits on the one before.
#
#   source traits.env; EXP_DIR=<HM3+ run directory> bash imputed/submit_dec_ag.sh
set -euo pipefail
: "${EXP_DIR:?}" "${ACCOUNT:?}" "${GPU_ACCOUNT:?}"
export HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXP_DIR"
export EXP_DIR="$PWD" LABELS="height=height:sex,age bmi=bmi:sex,age"

export ARRAY_TABLE="${ARRAY_TABLE:-$(readlink -f alphagenome.tsv.gz)}"
export NEW_TABLE="$PWD/alphagenome/alphagenome_hm3.tsv.gz"
merge=$(sbatch --parsable --account="$ACCOUNT" --export=ALL --output=logs/%x.%j.out \
    "$HERE/imputed/merge_alphagenome.sbatch")

# The table is read whole and reduced by PCA: the array's peak was ~100 GB at 416k SNPs.
export ALPHAGENOME="$PWD/alphagenome/alphagenome_merged.tsv.gz" SPECS="dec_ag@height dec_ag@bmi"
embed=$(sbatch --parsable --account="$ACCOUNT" --export=ALL --array=0-1 --mem=480G \
    --time=12:00:00 --dependency="afterok:$merge" --output=logs/%x.%A_%a.out \
    "$HERE/embedding.sbatch")

export STAGE=0 TRAIN_LABELS="height bmi" TRAIN_SPECS=dec_ag TRAIN_CONFIGS=b138_d50 SEEDS=78
export FOLDS="0 1 2 3 4"
recipe="--bn-recalibrate 20000 --snp-sample 0.5 --chunk-rows 2048 --chunks-per-buffer 4"
recipe+=" --snp-blocks 64"
train() {
    sbatch --parsable --account="$GPU_ACCOUNT" --export=ALL --array=0-9 --mem=128G \
        --time=12:00:00 --dependency="afterok:$embed" --output=logs/%x.%A_%a.out \
        "$HERE/train.sbatch"
}
plain=$(TRAIN_FLAGS="$recipe" RUN_SUFFIX=_bn_chunk train)
penalized=$(TRAIN_FLAGS="$recipe --free-snp-weights 0.3" RUN_SUFFIX=_bn_chunk_fw0.3 train)
echo "merge $merge embed $embed plain $plain penalized $penalized"
echo "$(date -Is) dec_ag merge $merge embed $embed plain $plain penalized $penalized" \
    >> logs/jobs.txt
