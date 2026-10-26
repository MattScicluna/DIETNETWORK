#!/usr/bin/env bash
# The dietnet pipeline at biobank scale: every step timed (wall time, peak RSS).
#
# Trains a throwaway one-fold model on a reference panel for a few epochs
# (accuracy is not the point; the work per step is the same), then matches a
# cohort to it. Inference is predict_gpu.sbatch. Inputs come from the environment, so no
# cluster path is committed:
#
#   TRAIN_PREFIX=/path/panel TRAIN_LABELS=/path/panel_labels.tsv \
#   TEST_PREFIX=/path/cohort DIETNET_VENV=/path/.venv RUN_DIR=/path/to/run_dir \
#   sbatch --account=ACCOUNT benchmarks/slurm/bench_predict.sbatch
#
# Outputs (the matched cohort HDF5, predictions) are cohort data and stay
# under RUN_DIR on the cluster.
#SBATCH --job-name=dietnet_bench_pipeline
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out

set -euo pipefail

: "${TRAIN_PREFIX:?}" "${TRAIN_LABELS:?}" "${TEST_PREFIX:?}" "${DIETNET_VENV:?}" "${RUN_DIR:?}"
DIETNET="$DIETNET_VENV/bin/dietnet"
# GNU time (for peak memory); not always at /usr/bin/time (Alliance clusters: CVMFS).
TIME="$(type -P time)"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

step() {
    local name="$1"; shift
    echo "== $name"
    "$TIME" -v "$@" > "$name.log" 2> "$name.time"
    grep -E "Elapsed \(wall|Maximum resident" "$name.time" | sed 's/^\s*/   /'
}

cat > config.yaml <<'YAML'
batch_size: 138
epochs: 3
input_dropout: 0.995
dropout_main: 0.1
lr_aux: 0.0006
lr_main: 0.0003
learning_rate_annealing: 0.99
nb_hidden_u_aux: [100, 100]
nb_hidden_u_main: [100]
patience: 3
seed: 78
uniform_init_limit: 0.02
YAML

echo "node $(hostname)"
"$DIETNET" info
echo "panel  $(wc -l < "$TRAIN_PREFIX.fam") samples x $(wc -l < "$TRAIN_PREFIX.bim") SNPs"
echo "cohort $(wc -l < "$TEST_PREFIX.fam") samples x $(wc -l < "$TEST_PREFIX.bim") SNPs"

step create-dataset "$DIETNET" create-dataset --plink "$TRAIN_PREFIX" --labels "$TRAIN_LABELS" --out dataset.h5
step partition "$DIETNET" partition --dataset dataset.h5 --out partition.npz
step generate-embedding "$DIETNET" generate-embedding --dataset dataset.h5 --partition partition.npz --out embedding.npz --ncpus 8
step compute-input-stats "$DIETNET" compute-input-stats --dataset dataset.h5 --partition partition.npz --out stats.npz --ncpus 8
step train "$DIETNET" train --dataset dataset.h5 --partition partition.npz --embedding embedding.npz \
    --input-stats stats.npz --config config.yaml --exp-dir exp --seeds 78 --folds 0
# Matching is CPU and I/O; inference goes to a GPU job (predict_gpu.sbatch) with
# the same work dir, which reuses the matched cohort.
step match "$DIETNET" match --model exp/packages --plink "$TEST_PREFIX" --work-dir work
tail -2 match.log
du -sh dataset.h5 work exp/packages
