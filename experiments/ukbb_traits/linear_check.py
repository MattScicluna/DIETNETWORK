"""Is there signal for the label in the genotypes? A marginal linear score, as a check.

For one fold: each SNP's marginal effect on the label over the training rows
(genotypes standardised with the fold's statistics, missing values at the
mean), then the test rows scored with the top-k SNPs by |effect|, and the
correlation of that score with the label. Fit on training rows only. A clearly
positive test r says labels and genotypes line up and carry signal, so a model
that stays at r ~ 0 is failing to learn it.

    python linear_check.py --dataset data/height.h5 --partition partition.npz \
        --stats stats.npz --fold 0 --top 1000,10000,100000,0
"""

import argparse

import h5py
import numpy as np

from dietnet import genotypes


def blocks(store, rows, mean, sd, size):
    """Standardised float32 genotypes of ``rows``, a block at a time."""
    for start in range(0, len(rows), size):
        chunk = rows[start : start + size]
        g = genotypes.read_rows(store, chunk).astype(np.float32)
        g = np.where(g < 0, mean, g)
        yield chunk, (g - mean) / sd


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--top", default="1000,10000,100000,0", help="k values; 0 = all SNPs")
    parser.add_argument("--block", type=int, default=2048)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as f:
        y = np.array(f["regression_labels"], dtype=np.float64)
    train, _, test = (np.sort(np.asarray(r)) for r in
                      np.load(args.partition, allow_pickle=True)["folds_indexes"][args.fold])  # fmt: skip
    stats = np.load(args.stats)
    mean = stats["means_by_fold"][args.fold].astype(np.float32)
    sd = stats["sd_by_fold"][args.fold].astype(np.float32)
    store = genotypes.open_store(args.dataset, genotypes.choose(args.dataset, "auto"))

    y_train = y[train] - y[train].mean()
    xty = np.zeros(len(mean), dtype=np.float64)
    for chunk, x in blocks(store, train, mean, sd, args.block):
        xty += x.T.astype(np.float64) @ y_train[np.searchsorted(train, chunk)]
    effect = xty / len(train)  # covariance of each standardised SNP with the label

    order = np.argsort(-np.abs(effect))
    ks = [int(k) or len(effect) for k in args.top.split(",")]
    scores = {k: np.zeros(len(test)) for k in ks}
    weights = {k: np.where(np.isin(np.arange(len(effect)), order[:k]), effect, 0.0) for k in ks}
    for chunk, x in blocks(store, test, mean, sd, args.block):
        at = np.searchsorted(test, chunk)
        for k in ks:
            scores[k][at] = x.astype(np.float64) @ weights[k]

    print(f"fold {args.fold}: {len(train):,} training rows, {len(test):,} test rows")
    print(f"largest |effect| {np.abs(effect).max():.4f}; median {np.median(np.abs(effect)):.5f}")
    for k in ks:
        r = np.corrcoef(scores[k], y[test])[0, 1]
        print(f"top {k:>7,} SNPs: test r = {r:.4f}")


if __name__ == "__main__":
    main()
