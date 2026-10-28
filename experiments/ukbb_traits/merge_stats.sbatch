#!/usr/bin/env bash
# stats.npz from the per-fold files, as compute-input-stats over all folds writes it.
#SBATCH --job-name=traits_merge_stats
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
source "$HERE/common.sh"

"$PYTHON" - "$NB_FOLDS" <<'PY'
import sys
import numpy as np
folds = [np.load(f"stats_fold{k}.npz") for k in range(int(sys.argv[1]))]
merged = {key: np.concatenate([f[key] for f in folds]) for key in ("means_by_fold", "sd_by_fold")}
for key, value in merged.items():
    if not np.isfinite(value).all():
        sys.exit(f"{key}: non-finite values")
np.savez("stats.npz", **merged)
print({key: value.shape for key, value in merged.items()})
PY
