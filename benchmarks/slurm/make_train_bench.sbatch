#!/usr/bin/env bash
# Build the training benchmark (benchmarks/make_train_bench.py) and partition it.
# BENCH_ARGS: "--source /path/matched.h5" (where that cohort lives), or
# "--simulate SAMPLES SNPS" (anywhere).
#
#   BENCH_ARGS="--simulate 486748 169829" BENCH_DIR=/path/bench DIETNET_VENV=/path/.venv \
#   REPO=/path/repo sbatch [--account=ACCOUNT] benchmarks/slurm/make_train_bench.sbatch
#SBATCH --job-name=dietnet_make_train_bench
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.out

set -euo pipefail
: "${BENCH_ARGS:?}" "${BENCH_DIR:?}" "${DIETNET_VENV:?}" "${REPO:?}"
start=$(date +%s)
# shellcheck disable=SC2086  # BENCH_ARGS is a list of arguments
"$DIETNET_VENV/bin/python" "$REPO/benchmarks/make_train_bench.py" "$BENCH_DIR" $BENCH_ARGS
echo "built in $(( $(date +%s) - start )) s"
"$DIETNET_VENV/bin/dietnet" partition --dataset "$BENCH_DIR/dataset.h5" \
    --out "$BENCH_DIR/partition.npz"
ls -la "$BENCH_DIR"
