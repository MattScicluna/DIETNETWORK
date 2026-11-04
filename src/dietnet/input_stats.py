"""Per-fold SNP means and SDs, as upstream's compute_input_features_statistics.py
computes them, from genotype counts streamed a block of samples at a time.

Upstream loads a fold's training genotypes whole (int8, samples x SNPs: ~330 GB
for 254k samples x 1.29M SNPs) and then, per SNP over the non-missing calls,

    mean = sum(x) / n
    sd   = sqrt(sum((x - mean)^2) / (n - 1)) + 1e-6

Both follow from how many calls are 0, 1 and 2 (missing, -1, excluded), which
:func:`dietnet.genotypes.count_by_fold` gathers in a few GB whatever the number
of samples. The means are the same numbers as upstream's (an integer sum over an
integer count); the SDs differ only by floating-point rounding (the sum of
squares is taken over three counts rather than over every sample).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dietnet import genotypes


def compute(dataset, partition, inputs: str = "auto", block: int = 2048) -> dict:
    """``means_by_fold`` and ``sd_by_fold`` (folds x SNPs, float64), as upstream writes."""
    folds = np.load(partition, allow_pickle=True)["folds_indexes"]
    store = genotypes.open_store(dataset, genotypes.choose(dataset, inputs))
    counts, _ = genotypes.count_by_fold(store, [fold[0] for fold in folds], block=block)
    c = counts[:, 0].astype(np.float64)  # folds x SNPs x (n0, n1, n2)
    n = c.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):  # n <= 1: nan, as upstream
        means = (c[..., 1] + 2 * c[..., 2]) / n
        squares = sum(c[..., g] * (g - means) ** 2 for g in range(3))
        sds = np.sqrt(squares / (n - 1)) + 1e-6
    return {"means_by_fold": means, "sd_by_fold": sds}


def write(dataset, partition, out, inputs: str = "auto", block: int = 2048) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **compute(dataset, partition, inputs, block))
    return out
