"""A training benchmark at real scale, for timing only.

Writes OUT_DIR/dataset.h5 in the dataset schema (contiguous ``inputs``) with
random class labels, and a random per-fold embedding and flat normalisation
statistics next to it. The genotypes are either copied from SOURCE, e.g. a
cohort matched by ``dietnet match`` (where the cohort lives), or simulated
with ``--simulate SAMPLES SNPS`` (anywhere: no cohort data involved). Either
way reading them costs what real ones would; the labels and features are
noise, so the model learns nothing and its accuracy means nothing. Partition
with `dietnet partition` afterwards.

    python benchmarks/make_train_bench.py OUT_DIR --source SOURCE.h5
    python benchmarks/make_train_bench.py OUT_DIR --simulate 486748 169829
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

FOLDS = 5


def _simulate(dst, n_samples, n_snps, block, rng) -> None:
    """Genotypes drawn per SNP from binomial(2, p), p in [0.05, 0.5], ~1% missing."""
    freq = rng.uniform(0.05, 0.5, n_snps)
    inputs = dst.create_dataset("inputs", shape=(n_samples, n_snps), dtype=np.int8)
    for start in range(0, n_samples, block):
        rows = min(block, n_samples - start)
        g = rng.binomial(2, freq, size=(rows, n_snps)).astype(np.int8)
        g[rng.random((rows, n_snps)) < 0.01] = -1
        inputs[start : start + rows] = g
    dst.create_dataset("samples", data=np.array([f"sim{i}" for i in range(n_samples)], "S"))
    dst.create_dataset("snp_names", data=np.array([f"snp{j}" for j in range(n_snps)], "S"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="copy genotypes from this dataset")
    source.add_argument("--simulate", type=int, nargs=2, metavar=("SAMPLES", "SNPS"))
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--block", type=int, default=4096, help="samples written at a time")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    with h5py.File(args.out_dir / "dataset.h5", "w") as dst:
        if args.source:
            with h5py.File(args.source, "r") as src:
                n_samples, n_snps = src["inputs"].shape
                inputs = dst.create_dataset("inputs", shape=(n_samples, n_snps), dtype=np.int8)
                for start in range(0, n_samples, args.block):
                    inputs[start : start + args.block] = src["inputs"][start : start + args.block]
                dst.create_dataset("samples", data=src["samples"][:])
                dst.create_dataset("snp_names", data=src["snp_names"][:])
        else:
            n_samples, n_snps = args.simulate
            _simulate(dst, n_samples, n_snps, args.block, rng)
        names = np.array([f"class{i}" for i in range(args.classes)], dtype="S")
        dst.create_dataset("class_label_names", data=names)
        dst.create_dataset(
            "class_labels", data=rng.integers(0, args.classes, n_samples).astype(np.int64)
        )

    # Upstream's embedding is per fold: genotype frequency per class, 3 per class.
    emb = rng.random((FOLDS, n_snps, 3 * args.classes), dtype=np.float32)
    np.savez(args.out_dir / "embedding.npz", emb=emb)
    np.savez(
        args.out_dir / "stats.npz",
        means_by_fold=np.ones((FOLDS, n_snps), dtype=np.float32),
        sd_by_fold=np.ones((FOLDS, n_snps), dtype=np.float32),
    )
    print(f"{n_samples:,} samples x {n_snps:,} SNPs -> {args.out_dir}")


if __name__ == "__main__":
    main()
