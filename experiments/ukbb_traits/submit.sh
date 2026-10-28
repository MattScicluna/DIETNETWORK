#!/usr/bin/env bash
# Submit the UK Biobank trait runs. Run prepare.sh first.
#
#   source traits.env   # GENOTYPES FIELDS PAN_UK ALPHAGENOME DIETNET_VENV EXP_DIR ACCOUNT
#                       # (and GPU_ACCOUNT, if GPU jobs go to another account)
#   bash submit.sh prep    # datasets, partition, stats, embeddings; then the pilot
#   TRAIN_LABELS="height bmi" TRAIN_SPECS="freq ag freq_ag freq_random freq_shuffled mlp" \
#   TRAIN_CONFIGS=b138 SEEDS="78 79 80" FOLDS="0 1 2 3 4" bash submit.sh train
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
*)
    echo "usage: submit.sh prep|train [sbatch options for train]" >&2
    exit 1
    ;;
esac
