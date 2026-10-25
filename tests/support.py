"""Test datasets and helpers shared across the suite.

Imported by name (``from support import ...``); pytest puts tests/ on the path.
The datasets themselves are described in tests/fixtures/build_fixtures.py.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The HussinLab/DIETNETWORK commit the parity tests are pinned to. Moving it is a
# deliberate act: re-run the parity suite and record what changed in
# docs/refactor-plan.md.
UPSTREAM_SHA = "5481556e472140c6d3a67363c1286f22109fb1ce"


@dataclass(frozen=True)
class Dataset:
    """One committed fixture: PLINK prefix, labels, and upstream's HDF5 of it."""

    name: str
    plink_prefix: Path
    labels: Path
    upstream_hdf5: Path

    def upstream_arrays(self):
        """Genotypes, class labels and sample ids exactly as upstream stored them."""
        with h5py.File(self.upstream_hdf5, "r") as f:
            return (
                np.array(f["inputs"], dtype=np.int8),
                np.array(f["class_labels"], dtype=np.int64),
                np.array(f["samples"]).astype(str),
            )


DATASETS = {
    name: Dataset(
        name=name,
        plink_prefix=FIXTURES / name,
        labels=FIXTURES / f"{name}_labels.tsv",
        upstream_hdf5=FIXTURES / f"{name}_upstream.hdf5",
    )
    for name in ("kgp_mini", "synthetic")
}


def stratified_folds(labels, n_folds=5, seed=0):
    """A fixed partition shared by both implementations: [(train, valid, test), ...].

    Partitioning is an input to everything compared here, so both sides get the
    same one rather than each making their own.
    """
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(labels), dtype=int)
    for c in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == c))
        fold_of[members] = np.arange(len(members)) % n_folds

    folds = []
    for k in range(n_folds):
        test = np.flatnonzero(fold_of == k)
        rest = rng.permutation(np.flatnonzero(fold_of != k))
        n_train = int(0.75 * len(rest))
        folds.append((np.sort(rest[:n_train]), np.sort(rest[n_train:]), test))
    return folds
