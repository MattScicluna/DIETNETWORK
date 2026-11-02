#!/usr/bin/env bash
# Every step of experiments/ukbb_traits, in order, on fake data around the
# kgp_mini fixture: SLURM's variables simulated, plink2 replaced by a copy (all
# samples are kept), configs cut to a few epochs, training on CPU.
#
#   bash experiments/ukbb_traits/dryrun/dryrun.sh WORK_DIR    # from the repository root
set -euo pipefail
REPO="$(pwd)" WORK="${1:?work directory (emptied)}"
rm -rf "$WORK" && mkdir -p "$WORK"
"$REPO/.venv/bin/python" "$REPO/experiments/ukbb_traits/dryrun/make_fake.py" \
    "$REPO/tests/fixtures" "$WORK/fake"

cp -r "$REPO/experiments/ukbb_traits" "$WORK/scripts"
for c in "$WORK"/scripts/configs/*.yaml; do
    sed -e 's/^epochs: .*/epochs: 3/' -e 's/^patience: .*/patience: 3/' "$c" > "$c.tmp"
    mv "$c.tmp" "$c"
done
export HERE="$WORK/scripts" GENOTYPES="$WORK/fake/geno/ukb" FIELDS="$WORK/fake/fields" \
    PAN_UK="$WORK/fake/pan" ALPHAGENOME="$WORK/fake/alphagenome.tsv.gz" \
    DIETNET_VENV="$REPO/.venv" EXP_DIR="$WORK/exp" SLURM_CPUS_PER_TASK=2
DIETNET="$DIETNET_VENV/bin/dietnet"

bash "$HERE/prepare.sh"
cmp <(cut -d' ' -f1,2 "$GENOTYPES.fam" | tr ' ' '\t' | sort) <(sort "$EXP_DIR/inputs/keep.txt")
for e in bed bim fam; do cp "$GENOTYPES.$e" "$EXP_DIR/cohort.$e"; done  # subset.sbatch

for i in 0 1 2 3; do SLURM_ARRAY_TASK_ID=$i bash "$HERE/dataset.sbatch"; done
bash "$HERE/partition.sbatch"
for k in 0 1 2 3 4; do SLURM_ARRAY_TASK_ID=$k bash "$HERE/stats.sbatch"; done
bash "$HERE/merge_stats.sbatch"
for i in 0 1 2 3 4; do SLURM_ARRAY_TASK_ID=$i bash "$HERE/embedding.sbatch"; done

cd "$EXP_DIR"
# Statistics fold by fold, merged, are what one run over every fold writes.
"$DIETNET" compute-input-stats --dataset data/height.h5 --partition partition.npz \
    --out stats_all.npz --ncpus 2
"$DIETNET_VENV/bin/python" - <<'PY'
import numpy as np
a, b = np.load("stats.npz"), np.load("stats_all.npz")
for key in ("means_by_fold", "sd_by_fold"):
    assert np.array_equal(a[key], b[key]), key
print("merged per-fold stats == one run over all folds:", a["means_by_fold"].shape)
PY

# The pilot's two tasks, then every other spec on another label set and fold.
export TRAIN_LABELS=height TRAIN_SPECS=freq TRAIN_CONFIGS="b138 b552" SEEDS=78 FOLDS=0
for i in 0 1; do SLURM_ARRAY_TASK_ID=$i SLURM_TMPDIR=$(mktemp -d) bash "$HERE/train.sbatch"; done
export TRAIN_LABELS=bmi_tdi TRAIN_SPECS="ag freq_ag freq_random freq_shuffled mlp" \
    TRAIN_CONFIGS=b552 FOLDS=1
for i in 0 1 2 3 4; do SLURM_ARRAY_TASK_ID=$i SLURM_TMPDIR=$(mktemp -d) bash "$HERE/train.sbatch"; done

# Decile features (templates filled per label set) and their training runs.
export SPECS="dec@height dec_ag@height dec@bmi dec_ag@bmi"
for i in 0 1 2 3; do SLURM_ARRAY_TASK_ID=$i bash "$HERE/embedding.sbatch"; done
test "$(cut -f2 inputs/deciles/height.tsv | tail -n +2 | sort -u | wc -l)" -eq 10
export TRAIN_LABELS="height bmi" TRAIN_SPECS="dec dec_ag" TRAIN_CONFIGS=b138 FOLDS=2
for i in 0 1 2 3; do SLURM_ARRAY_TASK_ID=$i SLURM_TMPDIR=$(mktemp -d) bash "$HERE/train.sbatch"; done
# Extra flags and a separate run directory (batch-norm recalibration).
TRAIN_LABELS=height TRAIN_SPECS=dec TRAIN_FLAGS="--bn-recalibrate 50" RUN_SUFFIX=_bn \
    SLURM_ARRAY_TASK_ID=0 SLURM_TMPDIR=$(mktemp -d) bash "$HERE/train.sbatch"
grep -q "batch-norm statistics recomputed on 50" runs/height/dec/b138_bn/logs/*.log
"$DIETNET_VENV/bin/python" -c "
import numpy as np
for name, width in [('dec@height', 30), ('dec_ag@bmi', 30 + 64)]:
    shape = np.load(f'embeddings/{name}.npz')['emb'].shape
    assert shape[-1] == width, (name, shape)
print('decile features: 30 columns, 94 with AlphaGenome')
"
for d in runs/*/*/*; do
    echo "== $d"
    "$DIETNET" check --packages "$d/packages" | tail -1
done
