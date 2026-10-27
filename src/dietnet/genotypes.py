"""Where training reads genotypes from: a store of int8 rows, one per sample.

Upstream's train.py copies the whole ``inputs`` matrix (every sample x every
SNP) into RAM before training, then draws its minibatches from that copy. For
a biobank that copy is hundreds of GB. Training only ever needs one batch of
rows at a time, so the copy is an implementation detail, and replaceable.

A *store* is anything that gives those rows by sample index -- ``store[i]`` or
``store[indices]`` -- as int8 arrays:

- ``ram``: upstream's copy in memory.
- ``memmap``: a memory map of the dataset file (:func:`dietnet.dataset.memmap_inputs`).
  Rows are read when a batch asks for them; the OS keeps what it can in its
  page cache, which every process on a node shares.

Both give the same rows, so the model trained from either is the same. The
store is the seam for what comes next: a packed (2-bit) backend if reading is
the bottleneck, and a data-parallel loop in which each GPU reads its own
batches from one shared store.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dietnet.dataset import memmap_inputs

MODES = ("ram", "memmap")

# "auto" maps inputs larger than this (the threshold the 2022 fork used for its
# PLINK genotype cache); below it a copy in RAM costs little and is fastest.
AUTO_MEMMAP_ABOVE_BYTES = 2 * 2**30


def choose(dataset, mode: str = "auto") -> str:
    """The store for ``dataset``: ``mode`` itself, or for "auto", ``memmap`` when
    ``inputs`` is over :data:`AUTO_MEMMAP_ABOVE_BYTES` and can be mapped
    (contiguous and uncompressed, as ``dietnet create-dataset`` writes it)."""
    if mode not in (*MODES, "auto"):
        raise ValueError(f"inputs mode must be one of {['auto', *MODES]}, not {mode!r}")
    if mode == "ram":
        return mode
    try:
        inputs = memmap_inputs(dataset)
    except ValueError as error:
        if mode == "memmap":
            raise ValueError(f"cannot memory-map the genotypes: {error}") from None
        return "ram"
    if mode == "memmap" or inputs.nbytes > AUTO_MEMMAP_ABOVE_BYTES:
        return "memmap"
    return "ram"


def open_store(dataset: str | Path, mode: str) -> np.ndarray:
    """The genotype store for ``dataset`` in ``mode`` (one of :data:`MODES`)."""
    if mode == "memmap":
        return memmap_inputs(dataset)
    if mode == "ram":
        with h5py.File(dataset, "r") as f:
            return np.array(f["inputs"], dtype=np.int8)
    raise ValueError(f"inputs mode must be one of {MODES}, not {mode!r}")
