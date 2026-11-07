"""Another run's folds on this run's dataset, by sample id.

The imputed cohort lacks a few hundred of the array cohort's samples, so a new
random partition would give different folds. This keeps the array run's folds:
each split of each fold keeps its samples in the same order, minus those absent
from the dataset, as row indices into this dataset. Writes the partition and one
single-fold copy per fold (``<out stem>_fold<k>.npz``, as partition.sbatch does).

    python partition_from.py --src ../exp_maf01/partition.npz --dataset data/height.h5 \
        --out partition.npz
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    src = np.load(args.src, allow_pickle=True)
    with h5py.File(args.dataset, "r") as f:
        samples = np.array(f["samples"]).astype(str)
    row = {s: i for i, s in enumerate(samples)}

    folds = src["folds_samples"]
    indexes = np.empty(folds.shape, dtype=object)
    kept = np.empty(folds.shape, dtype=object)
    for k in range(folds.shape[0]):
        for s in range(folds.shape[1]):
            # Stored as bytes (the datasets' |S ids): decode, as the dataset's are.
            source = np.asarray(folds[k, s]).astype(str)
            ids = [x for x in source if x in row]
            if len(source) - len(ids) > 0.01 * len(source):
                raise SystemExit(
                    f"fold {k} split {s}: {len(source) - len(ids)} of {len(source)} samples "
                    f"are not in {args.dataset} (e.g. {source[:3].tolist()} vs "
                    f"{samples[:3].tolist()}): not the same cohort?"
                )
            kept[k, s] = np.array(ids)
            indexes[k, s] = np.array([row[x] for x in ids], dtype=np.int64)
        sizes = [len(x) for x in indexes[k]]
        dropped = [len(folds[k, s]) - sizes[s] for s in range(folds.shape[1])]
        print(f"fold {k}: {sizes} (dropped {dropped})")

    np.savez(args.out, folds_indexes=indexes, folds_samples=kept)
    for k in range(folds.shape[0]):
        np.savez(
            args.out.with_name(f"{args.out.stem}_fold{k}.npz"),
            folds_indexes=indexes[k : k + 1],
            folds_samples=kept[k : k + 1],
        )


if __name__ == "__main__":
    main()
