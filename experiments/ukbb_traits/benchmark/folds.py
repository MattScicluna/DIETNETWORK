"""Sample lists and phenotypes for the linear benchmarks, on the Diet Network's folds.

From EXP_DIR's partition (row indices into the datasets' ``samples``), the
cohort's .fam (FIDs) and the adjusted labels, writes under OUT_DIR:

- ``fold<k>/{train,valid,test}.txt``: FID IID of each split of fold k;
- ``fold<k>/ld.txt``: 10,000 training samples (seeded), the LD reference for clumping;
- ``pheno.tsv``: FID IID and one column per label set.

    python folds.py --exp-dir EXP_DIR --labels height,bmi --out-dir EXP_DIR/bench
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--labels", required=True, help="label sets, comma-separated")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ld-samples", type=int, default=10_000)
    args = parser.parse_args()
    labels = args.labels.split(",")

    with h5py.File(args.exp_dir / f"data/{labels[0]}.h5", "r") as f:
        samples = np.array(f["samples"]).astype(str)
    fam = pd.read_csv(args.exp_dir / "cohort.fam", sep=r"\s+", header=None, dtype=str)
    fid = dict(zip(fam[1], fam[0]))
    ids = pd.DataFrame({"FID": [fid[s] for s in samples], "IID": samples})

    pheno = ids.copy()
    for label in labels:
        values = pd.read_csv(args.exp_dir / f"inputs/labels/{label}.tsv", sep="\t", dtype={0: str})
        values = values.set_index(values.columns[0]).iloc[:, 0]
        pheno[label] = values.loc[samples].to_numpy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pheno.to_csv(args.out_dir / "pheno.tsv", sep="\t", index=False)

    folds = np.load(args.exp_dir / "partition.npz", allow_pickle=True)["folds_indexes"]
    rng = np.random.default_rng(0)
    for k, fold in enumerate(folds):
        out = args.out_dir / f"fold{k}"
        out.mkdir(exist_ok=True)
        for name, rows in zip(("train", "valid", "test"), fold):
            ids.iloc[np.sort(np.asarray(rows))].to_csv(
                out / f"{name}.txt", sep="\t", index=False, header=False
            )
        train = np.asarray(fold[0])
        ld = rng.choice(train, min(args.ld_samples, len(train)), replace=False)
        ids.iloc[np.sort(ld)].to_csv(out / "ld.txt", sep="\t", index=False, header=False)
        print(
            f"fold {k}: "
            + ", ".join(f"{n} {len(r):,}" for n, r in zip(("train", "valid", "test"), fold))
        )


if __name__ == "__main__":
    main()
