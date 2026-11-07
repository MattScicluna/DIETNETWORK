#!/usr/bin/env bash
# Five folds (one class, so a plain random split), and a one-fold copy of the
# partition per fold, so that the statistics can be computed fold by fold in
# parallel by upstream's script.
#SBATCH --job-name=traits_partition
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
source "$HERE/common.sh"

"$DIETNET" partition --dataset "data/$FIRST.h5" --out partition.npz --nb-folds "$NB_FOLDS"
"$PYTHON" - "$NB_FOLDS" <<'PY'
import sys
import numpy as np
p = np.load("partition.npz", allow_pickle=True)
for k in range(int(sys.argv[1])):
    np.savez(f"partition_fold{k}.npz", **{key: p[key][k : k + 1] for key in p.files})
PY
