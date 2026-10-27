"""Where training reads genotypes from: a store of int8 rows, one per sample.

Upstream's train.py copies the whole ``inputs`` matrix (every sample x every
SNP) into RAM before training, then draws its minibatches from that copy. For
a biobank that copy is hundreds of GB. Training only ever needs one batch of
rows at a time, so the copy is an implementation detail, and replaceable.

A *store* is anything that gives those rows by sample index -- ``store[i]`` or
``store[indices]`` -- as int8 arrays (dosage 0, 1, 2; -1 missing):

- ``ram``: upstream's copy in memory.
- ``memmap``: a memory map of the dataset file (:func:`dietnet.dataset.memmap_inputs`).
  Rows are read when a batch asks for them; the OS keeps what it can in its
  page cache, which every process on a node shares.
- ``packed``: a memory map of a 2-bit copy (:func:`pack`), a quarter of the
  bytes, decoded per batch. Training from local disk is limited by how many
  bytes it reads, so this is the fastest store when the data does not fit in
  memory.

All give the same rows, so the model trained from any of them is the same.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

from dietnet.dataset import memmap_inputs

MODES = ("ram", "memmap", "packed")

# "auto" leaves inputs smaller than this in RAM (the threshold the 2022 fork used
# for its PLINK genotype cache); a copy in RAM then costs little and is fastest.
AUTO_MEMMAP_ABOVE_BYTES = 2 * 2**30

# 2-bit codes: dosage 0, 1, 2 as themselves, missing (-1) as 3. Four genotypes per
# byte, the first in the lowest bits; rows padded to whole bytes.
_DECODE = np.array(
    [[(byte >> (2 * k)) & 3 for k in range(4)] for byte in range(256)], dtype=np.int8
)
_DECODE[_DECODE == 3] = -1


def packed_path(dataset) -> Path:
    """Where :func:`pack` writes the 2-bit copy of ``dataset``."""
    return Path(f"{dataset}.packed.npy")


def _meta_path(dataset) -> Path:
    return Path(f"{dataset}.packed.json")


def _source(dataset) -> dict:
    """What a pack was built from: if the dataset changes, the pack is stale."""
    stat = os.stat(dataset)
    with h5py.File(dataset, "r") as f:
        n_samples, n_snps = f["inputs"].shape
    return {
        "n_samples": int(n_samples),
        "n_snps": int(n_snps),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def encode(rows: np.ndarray) -> np.ndarray:
    """int8 genotype rows (samples x SNPs) as 2-bit packed uint8 (samples x ceil(SNPs/4))."""
    rows = np.atleast_2d(rows)
    n, m = rows.shape
    codes = np.zeros((n, -(-m // 4) * 4), dtype=np.uint8)
    codes[:, :m] = np.where(rows < 0, 3, rows)
    codes = codes.reshape(n, -1, 4)
    return codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)


def decode(packed: np.ndarray, n_snps: int) -> np.ndarray:
    """Packed rows back to int8 rows (samples x n_snps)."""
    packed = np.atleast_2d(packed)
    rows = _DECODE[packed].reshape(len(packed), -1)
    return rows if rows.shape[1] == n_snps else np.ascontiguousarray(rows[:, :n_snps])


def pack(dataset, block: int = 4096) -> Path:
    """Write the 2-bit copy of ``dataset``'s inputs next to it; return its path.

    Reads the dataset ``block`` samples at a time, so memory stays small. The
    sidecar ``.packed.json`` records the dataset it came from.
    """
    dataset = Path(dataset)
    source = _source(dataset)
    n, m = source["n_samples"], source["n_snps"]
    out = packed_path(dataset)
    tmp = out.with_name(out.name + ".partial")
    packed = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, -(-m // 4)))
    with h5py.File(dataset, "r") as f:
        for start in range(0, n, block):
            packed[start : start + block] = encode(f["inputs"][start : start + block])
    packed.flush()
    del packed
    os.replace(tmp, out)
    _meta_path(dataset).write_text(json.dumps(source))
    return out


def is_packed(dataset) -> bool:
    """Whether ``dataset`` has a 2-bit copy built from it as it is now."""
    meta = _meta_path(dataset)
    if not (packed_path(dataset).exists() and meta.exists()):
        return False
    return json.loads(meta.read_text()) == _source(dataset)


class PackedStore:
    """int8 rows decoded from the 2-bit copy, read through a memory map.

    ``store[i]`` is one row, ``store[indices]`` a block of rows, as for a numpy
    array; :meth:`packed_rows` gives the undecoded bytes, for decoding on a GPU.
    """

    def __init__(self, dataset):
        if not is_packed(dataset):
            raise ValueError(
                f"{dataset} has no up-to-date packed copy; run `dietnet pack-genotypes`"
            )
        self.packed = np.load(packed_path(dataset), mmap_mode="r")
        self.n_snps = json.loads(_meta_path(dataset).read_text())["n_snps"]
        self.shape = (len(self.packed), self.n_snps)
        self.dtype = np.dtype(np.int8)

    def __len__(self) -> int:
        return self.shape[0]

    def packed_rows(self, rows) -> np.ndarray:
        return np.asarray(self.packed[rows])

    def __getitem__(self, rows) -> np.ndarray:
        if np.ndim(rows) == 0:
            return decode(self.packed[int(rows)], self.n_snps)[0]
        return decode(self.packed_rows(rows), self.n_snps)


def choose(dataset, mode: str = "auto") -> str:
    """The store for ``dataset``: ``mode`` itself, checked, or for "auto": ``packed``
    if an up-to-date pack exists, else ``memmap`` when ``inputs`` is over
    :data:`AUTO_MEMMAP_ABOVE_BYTES` and can be mapped (contiguous and
    uncompressed, as ``dietnet create-dataset`` writes it), else ``ram``."""
    if mode not in (*MODES, "auto"):
        raise ValueError(f"inputs mode must be one of {['auto', *MODES]}, not {mode!r}")
    if mode == "ram":
        return mode
    if mode == "packed" or (mode == "auto" and is_packed(dataset)):
        if not is_packed(dataset):
            raise ValueError(
                f"cannot read packed genotypes: {dataset} has no up-to-date packed copy; "
                "run `dietnet pack-genotypes` first"
            )
        return "packed"
    try:
        inputs = memmap_inputs(dataset)
    except ValueError as error:
        if mode == "memmap":
            raise ValueError(f"cannot memory-map the genotypes: {error}") from None
        return "ram"
    if mode == "memmap" or inputs.nbytes > AUTO_MEMMAP_ABOVE_BYTES:
        return "memmap"
    return "ram"


def open_store(dataset: str | Path, mode: str):
    """The genotype store for ``dataset`` in ``mode`` (one of :data:`MODES`)."""
    if mode == "packed":
        return PackedStore(dataset)
    if mode == "memmap":
        return memmap_inputs(dataset)
    if mode == "ram":
        with h5py.File(dataset, "r") as f:
            return np.array(f["inputs"], dtype=np.int8)
    raise ValueError(f"inputs mode must be one of {MODES}, not {mode!r}")
