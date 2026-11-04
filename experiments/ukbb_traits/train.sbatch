#!/usr/bin/env bash
# One model per GPU: array task i is the i-th of every combination of
# TRAIN_LABELS x TRAIN_SPECS x TRAIN_CONFIGS x SEEDS x FOLDS (space separated;
# a spec of "mlp" trains upstream's MLP, without features). The fastest path in
# docs/scaling.md: dietnet's engine on the 2-bit copy, staged to the node's disk.
#SBATCH --job-name=traits_train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
source "$HERE/common.sh"
: "${TRAIN_LABELS:?}" "${TRAIN_SPECS:?}" "${TRAIN_CONFIGS:?}" "${SEEDS:?}" "${FOLDS:?}"

i="$SLURM_ARRAY_TASK_ID"
pick() {  # pick <list>: the element of <list> for task i, then divides i
    local -a items
    read -ra items <<< "$1"
    REPLY="${items[$((i % ${#items[@]}))]}"
    i=$((i / ${#items[@]}))
}
pick "$FOLDS"; fold="$REPLY"
pick "$SEEDS"; seed="$REPLY"
pick "$TRAIN_CONFIGS"; config="$REPLY"
pick "$TRAIN_SPECS"; spec="$REPLY"
pick "$TRAIN_LABELS"; label="$REPLY"
echo "$label / $spec / $config: seed $seed, fold $fold on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Training reads only labels and metadata from the .h5, and genotypes from the
# 2-bit copy: stage the copy and link the .h5, so the sidecar's check (the .h5's
# size and mtime) sees the file the copy was made from; the copy's own mtime is
# not checked. Not cp -p: Narval's node disk refuses Lustre's permissions, and
# cp then fails.
local_data="$SLURM_TMPDIR/$label.h5"
ln -s "$EXP_DIR/data/$label.h5" "$local_data"
time cp "data/$label.h5.packed.npy" "data/$label.h5.packed.json" "$SLURM_TMPDIR/"
# Read the copy once, in order, so it is in the page cache before training: batches
# read random rows, and from a cold node disk the first epoch on 1.29M SNPs (137 GB)
# took 96 min, the third 39 s. Needs --mem above the copy's size.
time cat "$SLURM_TMPDIR/$label.h5.packed.npy" > /dev/null

embedding="embeddings/$spec@$label.npz"  # label-specific (decile) features first
[ -e "$embedding" ] || embedding="embeddings/$spec.npz"
features=(--embedding "$embedding")
[ "$spec" = mlp ] && features=(--model mlp)
# TRAIN_FLAGS: further `dietnet train` flags (e.g. --bn-recalibrate 20000);
# RUN_SUFFIX keeps such runs apart from the plain ones (runs/<label>/<spec>/<config><suffix>).
read -ra extra <<< "${TRAIN_FLAGS:-}"
"$DIETNET" train --dataset "$local_data" --partition partition.npz "${features[@]}" \
    --input-stats stats.npz --config "$HERE/configs/$config.yaml" \
    --exp-dir "runs/$label/$spec/$config${RUN_SUFFIX:-}" --seeds "$seed" --folds "$fold" \
    --task regression --engine dietnet --inputs packed --workers 8 ${extra[@]+"${extra[@]}"}
