#!/usr/bin/env bash
# Training throughput by loader, workers, missing-value fill and batch size
# (benchmarks/bench_train_throughput.py), with the dataset staged to local disk.
#
#   BENCH_DIR=/path/bench DIETNET_VENV=/path/.venv REPO=/path/repo \
#   sbatch [--account=ACCOUNT] benchmarks/slurm/bench_train_throughput.sbatch
#SBATCH --job-name=dietnet_bench_throughput
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.out

set -euo pipefail
: "${BENCH_DIR:?}" "${DIETNET_VENV:?}" "${REPO:?}"
echo "node $(hostname), $(nproc) CPUs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
start=$(date +%s)
cp "$BENCH_DIR/dataset.h5" "$SLURM_TMPDIR/dataset.h5"
echo "staged to local disk in $(( $(date +%s) - start )) s"
"$DIETNET_VENV/bin/python" "$REPO/benchmarks/bench_train_throughput.py" "$BENCH_DIR" \
    --dataset "$SLURM_TMPDIR/dataset.h5" --workers 0 4 8 \
    | grep -E "^(table|loader|train)\s|samples x"
