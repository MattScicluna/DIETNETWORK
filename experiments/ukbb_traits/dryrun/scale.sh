#!/usr/bin/env bash
# The real step scripts, as SLURM jobs, on simulated inputs of the real shape
# (simulate.py): how long each step takes and how much memory it needs, on a
# cluster without the cohort. One label set; afterwards
#
#   sacct -j <ids from $SIM_DIR/exp/logs/jobs.txt> --format=JobID,JobName,Elapsed,MaxRSS,State
#
#   SIM_DIR=/path/sim DIETNET_VENV=/path/.venv CPU_OPTS="--partition=cpu" \
#   GPU_OPTS="--partition=gpu" bash experiments/ukbb_traits/dryrun/scale.sh   # from the repo root
set -euo pipefail
: "${SIM_DIR:?}" "${DIETNET_VENV:?}"
read -ra CPU <<< "${CPU_OPTS:-}"
read -ra GPU <<< "${GPU_OPTS:-}"
SHAPE="${SHAPE:-424768 485516}"
export HERE="$(pwd)/experiments/ukbb_traits"
export GENOTYPES="$SIM_DIR/geno/ukb" FIELDS="$SIM_DIR/fields" PAN_UK="$SIM_DIR/pan" \
    ALPHAGENOME="$SIM_DIR/alphagenome.tsv.gz" EXP_DIR="$SIM_DIR/exp" DIETNET_VENV \
    LABELS="height=height:sex,age"
mkdir -p "$EXP_DIR/logs"
cd "$EXP_DIR"

submit() {  # submit <name> [sbatch options...] <script>: prints the job id
    local name="$1" id
    shift
    id=$(sbatch --parsable --export=ALL --output="$EXP_DIR/logs/%x.%A_%a.out" "$@")
    echo "$(date -Is) $name $id" >> logs/jobs.txt
    echo "$name: $id" >&2
    echo "$id"
}

# Simulate, then prepare.sh, with the fileset itself as the cohort (every
# sample is kept, so there is nothing to subset). A finished simulation is reused.
after=()
if [ ! -e "$EXP_DIR/cohort.bed" ]; then
    sim=$(submit simulate "${CPU[@]}" --job-name=traits_simulate --cpus-per-task=32 \
        --mem=200G --time=06:00:00 --wrap "set -e; \"$DIETNET_VENV/bin/python\" \
        \"$HERE/dryrun/simulate.py\" \"$SIM_DIR\" --shape $SHAPE --procs 24; \
        bash \"$HERE/prepare.sh\"; \
        for e in bed bim fam; do ln -sfn \"$GENOTYPES.\$e\" \"$EXP_DIR/cohort.\$e\"; done")
    after=(--dependency="afterok:$sim")
fi
data=$(submit dataset "${CPU[@]}" --array=0 "${after[@]}" "$HERE/dataset.sbatch")
part=$(submit partition "${CPU[@]}" --dependency="afterok:${data}_0" "$HERE/partition.sbatch")
stats=$(submit stats "${CPU[@]}" --dependency="afterok:$part" "$HERE/stats.sbatch")
merged=$(submit merge_stats "${CPU[@]}" --dependency="afterok:$stats" "$HERE/merge_stats.sbatch")
emb=$(submit embeddings "${CPU[@]}" --array=0-4 --dependency="afterok:$part" \
    "$HERE/embedding.sbatch")
export TRAIN_LABELS=height TRAIN_SPECS=freq TRAIN_CONFIGS="b138 b552" SEEDS=78 FOLDS=0
submit pilot "${GPU[@]}" --array=0-1 --dependency="afterok:$merged:${emb}_0" \
    "$HERE/train.sbatch" > /dev/null
