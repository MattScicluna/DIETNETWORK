#!/usr/bin/env bash
# Training time per epoch by genotype store: upstream's copy in RAM, or a memory
# map read a batch at a time, from Lustre or staged to the node's local disk.
# BENCH_DIR comes from benchmarks/make_train_bench.py + `dietnet partition`.
#
#   BENCH_DIR=/path/bench DIETNET_VENV=/path/.venv INPUTS=memmap STAGE=1 \
#   sbatch --account=ACCOUNT --mem=32G benchmarks/slurm/bench_train_inputs.sbatch
#
# INPUTS: ram | memmap.  STAGE=1 copies dataset.h5 to $SLURM_TMPDIR first.
#SBATCH --job-name=dietnet_bench_train
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

set -euo pipefail
: "${BENCH_DIR:?}" "${DIETNET_VENV:?}" "${INPUTS:?}"
TIME="$(type -P time)"
RUN="$BENCH_DIR/run_${INPUTS}_stage${STAGE:-0}_${SLURM_JOB_ID}"
mkdir -p "$RUN"

echo "node $(hostname), inputs $INPUTS, stage ${STAGE:-0}, mem ${SLURM_MEM_PER_NODE:-?} MB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
DATASET="$BENCH_DIR/dataset.h5"
if [ "${STAGE:-0}" = 1 ]; then
    df -h "$SLURM_TMPDIR" | tail -1
    start=$(date +%s)
    cp "$DATASET" "$SLURM_TMPDIR/dataset.h5"
    echo "staged to local disk in $(( $(date +%s) - start )) s"
    DATASET="$SLURM_TMPDIR/dataset.h5"
fi

cat > "$RUN/config.yaml" <<'EOF'
batch_size: 138
epochs: 2
input_dropout: 0.995
dropout_main: 0.1
lr_aux: 0.0006
lr_main: 0.0003
learning_rate_annealing: 0.99
nb_hidden_u_aux: [100, 100]
nb_hidden_u_main: [100]
patience: 100
seed: 78
uniform_init_limit: 0.02
EOF

"$TIME" -v "$DIETNET_VENV/bin/dietnet" train --dataset "$DATASET" \
    --partition "$BENCH_DIR/partition.npz" --embedding "$BENCH_DIR/embedding.npz" \
    --input-stats "$BENCH_DIR/stats.npz" --config "$RUN/config.yaml" --exp-dir "$RUN/exp" \
    --seeds 78 --folds 0 --inputs "$INPUTS" 2> "$RUN/time.txt" || true
grep -E "Elapsed \(wall|Maximum resident" "$RUN/time.txt"
grep -h -E "^dietnet:|Loaded data in|Epoch [0-9]+ of|Train step time|Epoch execution time|Traceback|Error" \
    "$RUN"/exp/logs/train_*.log | tail -20
