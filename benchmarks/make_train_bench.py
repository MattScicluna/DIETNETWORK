"""A training benchmark from any contiguous genotype matrix, for timing only.

Copies ``inputs`` (and ``samples``, ``snp_names``) from SOURCE, e.g. a cohort
matched by ``dietnet match``, into OUT_DIR/dataset.h5 in the dataset schema,
with random class labels; writes a random per-fold embedding and flat
normalisation statistics next to it. Real genotypes at real scale, so reading
them costs what it will; the labels and features are noise, so the model
learns nothing and its accuracy means nothing. Partition with `dietnet
partition` afterwards.

    python benchmarks/make_train_bench.py SOURCE.h5 OUT_DIR [--classes 4]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

FOLDS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--block", type=int, default=4096, help="samples copied at a time")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    with h5py.File(args.source, "r") as src, h5py.File(args.out_dir / "dataset.h5", "w") as dst:
        n_samples, n_snps = src["inputs"].shape
        inputs = dst.create_dataset("inputs", shape=(n_samples, n_snps), dtype=np.int8)
        for start in range(0, n_samples, args.block):
            inputs[start : start + args.block] = src["inputs"][start : start + args.block]
        dst.create_dataset("samples", data=src["samples"][:])
        dst.create_dataset("snp_names", data=src["snp_names"][:])
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
