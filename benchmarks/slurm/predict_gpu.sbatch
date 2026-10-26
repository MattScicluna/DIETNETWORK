#!/usr/bin/env bash
# Inference on a GPU, after `dietnet match` has written the matched cohort to
# WORK_DIR (a CPU job): predict finds it there and only runs the models.
#
#   MODEL_DIR=/path/packages TEST_PREFIX=/path/cohort WORK_DIR=/path/to/run_dir/work \
#   OUT=/path/to/run_dir/predictions.tsv DIETNET_VENV=/path/.venv \
#   sbatch --account=ACCOUNT benchmarks/slurm/predict_gpu.sbatch
#SBATCH --job-name=dietnet_predict_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.out

set -euo pipefail
: "${MODEL_DIR:?}" "${TEST_PREFIX:?}" "${WORK_DIR:?}" "${OUT:?}" "${DIETNET_VENV:?}"
TIME="$(type -P time)"

echo "node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$TIME" -v "$DIETNET_VENV/bin/dietnet" predict --model "$MODEL_DIR" --plink "$TEST_PREFIX" \
    --out "$OUT" --work-dir "$WORK_DIR" --device cuda --batch-size 4096 \
    2> "$OUT.time" | grep -v -E "Embedding size"
grep -E "Elapsed \(wall|Maximum resident" "$OUT.time"
