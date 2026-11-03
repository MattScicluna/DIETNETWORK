#!/usr/bin/env bash
# Submit the UK Biobank trait runs. Run prepare.sh first.
#
#   source traits.env   # GENOTYPES FIELDS PAN_UK ALPHAGENOME DIETNET_VENV EXP_DIR ACCOUNT
#                       # (and GPU_ACCOUNT, if GPU jobs go to another account)
#   bash submit.sh prep    # datasets, partition, stats, embeddings; then the pilot
#   TRAIN_LABELS="height bmi" TRAIN_SPECS="freq ag freq_ag freq_random freq_shuffled mlp" \
#   TRAIN_CONFIGS=b138 SEEDS="78 79 80" FOLDS="0 1 2 3 4" bash submit.sh train
#   bash submit.sh bench   # linear benchmark (regenie, clumping/thresholding), same folds
#   bash submit.sh bench-ldpred2   # LDpred2-auto on bench's GWAS (CHR=22: one chromosome)
#   EXP_DIR=<new dir> SRC_EXP=<array run dir> IMPUTED=<prefix with {chr}> HM3_MAP=<rds> \
#   bash submit.sh imputed          # the cohort on imputed HapMap3+ variants
#
# Every step waits on the ones it reads (afterok), so `prep` can be submitted
# while the cluster is down and runs end to end when it is back. Job ids go to
# $EXP_DIR/logs/jobs.txt.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HERE
source "$HERE/common.sh"
: "${ACCOUNT:?}"

submit() {  # submit <name> [sbatch options...] <script>: prints the job id
    local name="$1" account="$ACCOUNT"
    shift
    [[ "$name" == train* ]] && account="${GPU_ACCOUNT:-$ACCOUNT}"
    local id
    id=$(sbatch --parsable --account="$account" --export=ALL \
        --output="$EXP_DIR/logs/%x.%A_%a.out" "$@")
    echo "$(date -Is) $name $id" >> logs/jobs.txt
    echo "$name: $id" >&2
    echo "$id"
}

train() {  # train [sbatch options...]: one array task per combination
    local n=1 list
    for list in "$TRAIN_LABELS" "$TRAIN_SPECS" "$TRAIN_CONFIGS" "$SEEDS" "$FOLDS"; do
        n=$((n * $(wc -w <<< "$list")))
    done
    export TRAIN_LABELS TRAIN_SPECS TRAIN_CONFIGS SEEDS FOLDS
    submit "train($TRAIN_LABELS / $TRAIN_SPECS / $TRAIN_CONFIGS)" \
        --array="0-$((n - 1))" "$@" "$HERE/train.sbatch"
}

case "${1:?prep or train}" in
prep)
    [ -s inputs/keep.txt ] || { echo "run prepare.sh first" >&2; exit 1; }
    subset=$(submit subset "$HERE/subset.sbatch")
    datasets=$(submit datasets --array="0-$((${#LABEL_SETS[@]} - 1))" \
        --dependency="afterok:$subset" "$HERE/dataset.sbatch")
    # Partition, statistics and embeddings read only the first dataset.
    partition=$(submit partition --dependency="afterok:${datasets}_0" "$HERE/partition.sbatch")
    stats=$(submit stats --dependency="afterok:$partition" "$HERE/stats.sbatch")
    merged=$(submit merge_stats --dependency="afterok:$stats" "$HERE/merge_stats.sbatch")
    embeddings=$(submit embeddings --array="0-$((${#SPEC_NAMES[@]} - 1))" \
        --dependency="afterok:$partition" "$HERE/embedding.sbatch")
    # The pilot: the first label set and spec, fold 0, both batch sizes.
    TRAIN_LABELS="$FIRST" TRAIN_SPECS="${SPEC_NAMES[0]}" TRAIN_CONFIGS="b138 b552" \
        SEEDS=78 FOLDS=0 train --dependency="afterok:$merged:${embeddings}_0" > /dev/null
    ;;
train)
    : "${TRAIN_LABELS:?}" "${TRAIN_SPECS:?}" "${TRAIN_CONFIGS:?}" "${SEEDS:?}" "${FOLDS:?}"
    train "${@:2}" > /dev/null
    ;;
bench)
    # The linear benchmark on the same folds: sample lists, then regenie and
    # clumping/thresholding per fold (further arguments go to that array).
    export BENCH_LABELS="${BENCH_LABELS:-height,bmi}"
    folds=$(submit bench_folds --job-name=traits_bench_folds --cpus-per-task=1 --mem=8G \
        --time=00:30:00 --wrap "\"$PYTHON\" \"$HERE/benchmark/folds.py\" --exp-dir \"$EXP_DIR\" \
        --labels \"$BENCH_LABELS\" --out-dir \"$EXP_DIR/bench\"")
    submit regenie_ct --dependency="afterok:$folds" "${@:2}" \
        "$HERE/benchmark/regenie_ct.sbatch" > /dev/null
    ;;
imputed)
    # The cohort on HapMap3+ variants from the imputed data, in EXP_DIR (a new run
    # directory), with SRC_EXP's benchmark folds: ids -> 22 extracts -> merge.
    : "${IMPUTED:?}" "${HM3_MAP:?map_hm3_plus.rds}" "${SRC_EXP:?the array run directory}"
    export IMPUTED SRC_EXP
    ids=$(submit imputed_ids --job-name=traits_imputed_ids --cpus-per-task=2 --mem=24G \
        --time=01:00:00 --wrap "module load StdEnv/2023 gcc r/4.4.0 && \
        Rscript \"$HERE/imputed/hm3_ids.R\" \"$HM3_MAP\" \"$IMPUTED\" imputed/ids")
    extract=$(submit imputed_extract --dependency="afterok:$ids" "$HERE/imputed/extract.sbatch")
    submit imputed_merge --dependency="afterok:$extract" "${@:2}" \
        "$HERE/imputed/merge.sbatch" > /dev/null
    ;;
bench-ldpred2)
    # LDpred2-auto on the GWAS that `bench` wrote (further arguments go to the array).
    export BENCH_LABELS="${BENCH_LABELS:-height,bmi}" ${CHR:+CHR}
    submit ldpred2 "${@:2}" "$HERE/benchmark/ldpred2.sbatch" > /dev/null
    ;;
*)
    echo "usage: submit.sh prep|train|bench|bench-ldpred2|imputed [sbatch options]" >&2
    exit 1
    ;;
esac
