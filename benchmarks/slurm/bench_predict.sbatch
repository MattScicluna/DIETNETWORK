#!/usr/bin/env bash
# Time today's `dietnet predict` on a biobank-sized cohort: the baseline the
# refactor's HDF5 intermediate has to beat.
#
# Trains a throwaway one-fold model for a few epochs (accuracy is not the
# point; the predict path does the same work either way), then predicts on
# every sample of the test cohort under /usr/bin/time -v for wall time and peak
# memory. Inputs come from the environment, so no cluster path is committed:
#
#   TRAIN_PREFIX=/path/train TRAIN_LABELS=/path/train_labels.tsv \
#   TRAIN_OUTPUTS=/dir/with/partitioned_idx.npz,embedding.npz,input_features_means.npz \
#   TEST_PREFIX=/path/cohort DIETNET_VENV=/path/.venv RUN_DIR=/path/on/scratch \
#   sbatch --account=ACCOUNT benchmarks/slurm/bench_predict.sbatch
#
# TEST_PREFIX must already be on the model's SNP set (predict runs with
# --skip-preprocess). Predictions are written under RUN_DIR; they are cohort
# data and stay on the cluster.
#SBATCH --job-name=dietnet_bench_predict
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out

set -euo pipefail

: "${TRAIN_PREFIX:?}" "${TRAIN_LABELS:?}" "${TRAIN_OUTPUTS:?}" "${TEST_PREFIX:?}"
: "${DIETNET_VENV:?}" "${RUN_DIR:?}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
DIETNET=("$DIETNET_VENV/bin/python" -u -m Dietnet.cli)

EXP="$RUN_DIR/exp"
mkdir -p "$EXP/run"
cp "$TRAIN_OUTPUTS"/{partitioned_idx.npz,embedding.npz,input_features_means.npz} "$EXP/"
cp "$TRAIN_LABELS" "$EXP/labels.tsv"
cat > "$EXP/run/config.yaml" <<'YAML'
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

echo "node $(hostname)  code $(git -C "$REPO" rev-parse --short HEAD)"
echo "== train (throwaway, 3 epochs)"
/usr/bin/time -v "${DIETNET[@]}" train --exp-path "$EXP" --exp-name run --config config.yaml \
    --plink-prefix "$TRAIN_PREFIX" --label-file labels.tsv \
    --seeds 78 --folds 0 --output-dir "$EXP/packages" 2> "$RUN_DIR/train.time"
tail -20 "$RUN_DIR/train.time"

echo "== predict on $(wc -l < "$TEST_PREFIX.fam") samples x $(wc -l < "$TEST_PREFIX.bim") SNPs"
/usr/bin/time -v "${DIETNET[@]}" predict --model "$EXP/packages" --plink-prefix "$TEST_PREFIX" \
    --output "$RUN_DIR/predictions.tsv" --seeds 78 --folds 0 --skip-preprocess \
    --temp-dir "$RUN_DIR/tmp" --batch-size 1024 2> "$RUN_DIR/predict.time"
tail -25 "$RUN_DIR/predict.time"
du -sh "$RUN_DIR/tmp" "$EXP/packages" 2>/dev/null || true
