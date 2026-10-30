#!/usr/bin/env bash
# Training throughput by loader, workers, missing-value fill and batch size
# (benchmarks/bench_train_throughput.py), with the dataset -- and its packed copy,
# if `dietnet pack-genotypes` made one -- staged to local disk.
#
#   BENCH_DIR=/path/bench DIETNET_VENV=/path/.venv REPO=/path/repo \
#   [BATCH_SIZES="512 2048 8192"] [WORKERS="4 8"] \
#   sbatch [--account=ACCOUNT] benchmarks/slurm/bench_train_throughput.sbatch
#SBATCH --job-name=dietnet_bench_throughput
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.out

set -euo pipefail
: "${BENCH_DIR:?}" "${DIETNET_VENV:?}" "${REPO:?}"
echo "node $(hostname), $(nproc) CPUs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
start=$(date +%s)
# -p keeps modification times: the packed copy is checked against its dataset's.
cp --preserve=timestamps "$BENCH_DIR"/dataset.h5* "$SLURM_TMPDIR/"
echo "staged $(ls "$SLURM_TMPDIR" | tr '\n' ' ')to local disk in $(( $(date +%s) - start )) s"
# shellcheck disable=SC2086  # lists of numbers
"$DIETNET_VENV/bin/python" "$REPO/benchmarks/bench_train_throughput.py" "$BENCH_DIR" \
    --dataset "$SLURM_TMPDIR/dataset.h5" --batch-sizes ${BATCH_SIZES:-138 512 2048 8192} \
    --workers ${WORKERS:-0 4 8} | grep -E "^(table|loader|train)\s|samples x"
