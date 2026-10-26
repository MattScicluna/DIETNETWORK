#!/usr/bin/env bash
# Per-fold embedding and input statistics for the published split, recomputed
# with dietnet (upstream's scripts) and checked against the Zenodo release.
#SBATCH --job-name=dietnet_repro_prep
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out

set -euo pipefail
: "${RELEASE:?}" "${DIETNET_VENV:?}" "${EXP_DIR:?}" "${REPO:?}"
DIETNET="$DIETNET_VENV/bin/dietnet"
mkdir -p "$EXP_DIR"
cd "$EXP_DIR"

DATASET="$RELEASE/1KGP_dataset.hdf5"
PARTITION="$REPO/reproduction/1kgp_partition_seed124.npz"

"$DIETNET" generate-embedding --dataset "$DATASET" --partition "$PARTITION" --out embedding.npz --ncpus 8
"$DIETNET" compute-input-stats --dataset "$DATASET" --partition "$PARTITION" --out stats.npz --ncpus 8

"$DIETNET_VENV/bin/python" - "$RELEASE" <<'PY'
import sys
import numpy as np
release = sys.argv[1]
ours_e = np.load("embedding.npz")["emb"]
theirs_e = np.load(f"{release}/supporting_files/SNPs_embedding.npz")["emb"]
ours_s = np.load("stats.npz")
theirs_s = np.load(f"{release}/supporting_files/normalization_stats.npz")
checks = {
    "embedding": np.abs(ours_e - theirs_e).max(),
    "means": np.abs(ours_s["means_by_fold"] - theirs_s["means_by_fold"]).max(),
    "sds": np.abs(ours_s["sd_by_fold"] - theirs_s["sd_by_fold"]).max(),
}
print(checks)
if any(v > 1e-9 for v in checks.values()):
    sys.exit("recomputed per-fold inputs differ from the published ones")
PY

# The published configs, one per seed (they differ only in seed).
mkdir -p configs
cp "$RELEASE"/supporting_files/config_rep*models.yaml configs/
echo "prep done"
