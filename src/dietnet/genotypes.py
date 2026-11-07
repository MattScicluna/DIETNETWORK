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
DECODE_TABLE = np.array(
    [[(byte >> (2 * k)) & 3 for k in range(4)] for byte in range(256)], dtype=np.int8
)
DECODE_TABLE[DECODE_TABLE == 3] = -1


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
    rows = DECODE_TABLE[packed].reshape(len(packed), -1)
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


class PackedReader:
    """Rows of a 2-bit copy read with plain file reads: ``reader[rows]`` is
    ``packed[rows]``, one read per run of consecutive rows.

    Indexing the memory map instead turns each 322 KB row (1.29M SNPs) into ~80
    page faults of 4 KB, each a round trip: on Lustre, validation's ~105k scattered
    rows went at ~3 MB/s. Opened lazily, once per process (loader workers fork).
    """

    def __init__(self, path, n_rows, row_bytes, offset):
        self.path, self.n_rows, self.row_bytes, self.offset = Path(path), n_rows, row_bytes, offset
        self._file, self._pid = None, None

    @classmethod
    def of(cls, path) -> PackedReader:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)  # public readers: numpy versions differ
            read = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
            }.get(version)
            if read is None:
                raise ValueError(f"{path}: unsupported .npy format version {version}")
            shape, _, _ = read(f)
            return cls(path, shape[0], shape[1], f.tell())

    def __getstate__(self):
        return {**self.__dict__, "_file": None, "_pid": None}

    def _handle(self):
        if self._file is None or self._pid != os.getpid():
            self._file, self._pid = open(self.path, "rb", buffering=0), os.getpid()
        return self._file

    def __getitem__(self, rows) -> np.ndarray:
        return self.read(rows)

    def _read_into(self, f, view, first_row):
        f.seek(self.offset + int(first_row) * self.row_bytes)
        done = 0
        while done < len(view):
            n = f.readinto(view[done:])
            if not n:
                raise EOFError(f"{self.path}: short read at row {first_row}")
            done += n

    def read(self, rows, max_gap: int = 0) -> np.ndarray:
        """``packed[rows]``. Rows within ``max_gap`` of each other (in the given order,
        ascending) are read as one span and the rows between discarded: a few large
        reads instead of many small ones, which is what a parallel file system serves
        fast, for at most ``max_gap`` rows of extra bytes per gap."""
        scalar = np.ndim(rows) == 0
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        if len(rows) and (rows.min() < 0 or rows.max() >= self.n_rows):
            raise IndexError(f"rows out of range 0..{self.n_rows - 1}")
        out = np.empty((len(rows), self.row_bytes), dtype=np.uint8)
        if not len(rows):
            return out
        f = self._handle()
        step = np.diff(rows)
        breaks = np.flatnonzero((step < 1) | (step > max_gap + 1)) + 1
        for start, stop in zip(np.r_[0, breaks], np.r_[breaks, len(rows)]):
            first, last = int(rows[start]), int(rows[stop - 1])
            if last - first + 1 == stop - start:  # consecutive: straight into out
                self._read_into(f, memoryview(out[start:stop]).cast("B"), first)
                continue
            span = np.empty((last - first + 1, self.row_bytes), dtype=np.uint8)
            self._read_into(f, memoryview(span).cast("B"), first)
            out[start:stop] = span[rows[start:stop] - first]
        return out[0] if scalar else out


class CachedRows:
    """Some rows of a genotype source held in memory: ``cache[rows]`` for any of them.

    For the rows every epoch reads again (validation, the BN recalibration subset,
    the test set): one pass over the file, then no disk at all. ``read`` is the
    source's reader, called once with the sorted rows (e.g. ``PackedReader.read``
    with a ``max_gap``, so the pass is a few large reads).
    """

    def __init__(self, read, rows):
        self.rows = np.unique(np.asarray(rows, dtype=np.int64))
        self.data = np.asarray(read(self.rows))

    def __getitem__(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        at = np.searchsorted(self.rows, rows)
        if np.any(at >= len(self.rows)) or np.any(
            self.rows[np.minimum(at, len(self.rows) - 1)] != rows
        ):
            raise KeyError("rows not in the cache")
        return self.data[at]


class PackedStore:
    """int8 rows decoded from the 2-bit copy, read with file reads (:class:`PackedReader`).

    ``store[i]`` is one row, ``store[indices]`` a block of rows, as for a numpy
    array; :meth:`packed_rows` gives the undecoded bytes, for decoding on a GPU.
    """

    def __init__(self, dataset):
        if not is_packed(dataset):
            raise ValueError(
                f"{dataset} has no up-to-date packed copy; run `dietnet pack-genotypes`"
            )
        self.dataset = Path(dataset)
        self.n_snps = json.loads(_meta_path(dataset).read_text())["n_snps"]
        self._packed, self._reader = None, None
        self.shape = (len(self.packed), self.n_snps)
        self.dtype = np.dtype(np.int8)

    @property
    def packed(self) -> np.ndarray:
        if self._packed is None:  # opened where it is used: a loader worker maps its own
            self._packed = np.load(packed_path(self.dataset), mmap_mode="r")
        return self._packed

    def __getstate__(self):
        return {**self.__dict__, "_packed": None}

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def reader(self) -> PackedReader:
        """The undecoded rows, read with file reads rather than through the memory map."""
        if getattr(self, "_reader", None) is None:
            self._reader = PackedReader.of(packed_path(self.dataset))
        return self._reader

    def packed_rows(self, rows) -> np.ndarray:
        return self.reader[rows]

    def __getitem__(self, rows) -> np.ndarray:
        if np.ndim(rows) == 0:
            return decode(self.reader[int(rows)][None], self.n_snps)[0]
        return decode(self.packed_rows(rows), self.n_snps)


class MemmapStore:
    """int8 rows read through a memory map of the dataset's ``inputs``.

    Indexes like the array it wraps; unlike an ``np.memmap``, it pickles as its
    path, so a loader worker maps the file itself instead of being sent a copy.
    """

    def __init__(self, dataset):
        self.dataset = Path(dataset)
        self._x = None
        self.shape, self.dtype = self.x.shape, self.x.dtype

    @property
    def x(self) -> np.ndarray:
        if self._x is None:
            self._x = memmap_inputs(self.dataset)
        return self._x

    def __getstate__(self):
        return {**self.__dict__, "_x": None}

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows) -> np.ndarray:
        return self.x[rows]


def read_rows(store, indices) -> np.ndarray:
    """``store``'s rows ``indices``, in that order, read in one pass in file order.

    Batches come from a shuffled sampler; reading them sorted turns random reads
    into forward ones, and the rows are then put back in the sampler's order.
    """
    indices = np.asarray(indices)
    order = np.argsort(indices, kind="stable")
    rows = np.asarray(store[indices[order]])
    out = np.empty_like(rows)
    out[order] = rows
    return out


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
        return MemmapStore(dataset)
    if mode == "ram":
        with h5py.File(dataset, "r") as f:
            return np.array(f["inputs"], dtype=np.int8)
    raise ValueError(f"inputs mode must be one of {MODES}, not {mode!r}")


def count_by_fold(store, train_rows, klass=None, n_classes=1, block=2048):
    """Genotype counts over each fold's training rows, per class, a block of rows at
    a time: memory is a block plus the counts, whatever the number of samples.

    ``train_rows``: one array of row indices per fold. ``klass``: each row's class
    (0..n_classes-1), or None for one class. Returns ``counts`` (folds x classes x
    SNPs x 3, int64: how many rows have genotype 0, 1, 2; missing, -1, counts for
    none) and ``rows`` (folds x classes: training rows, missing genotypes or not).
    Rows are grouped by class and by the folds they train in, and each group is
    read once, so a row shared by four folds' training sets is read once, not four
    times.
    """
    n_rows, n_snps = store.shape
    n_folds = len(train_rows)
    member = np.zeros((n_folds, n_rows), dtype=bool)
    for f, rows in enumerate(train_rows):
        member[f, np.asarray(rows)] = True
    klass = np.zeros(n_rows, dtype=np.int64) if klass is None else np.asarray(klass)

    pattern = (member * (1 << np.arange(n_folds))[:, None]).sum(axis=0)
    rows = np.flatnonzero(pattern)
    keys, group = np.unique(klass[rows] * (1 << n_folds) + pattern[rows], return_inverse=True)
    group_counts = np.zeros((len(keys), n_snps, 3), dtype=np.int64)
    for start in range(0, len(rows), block):
        g = np.asarray(store[rows[start : start + block]])
        at = group[start : start + block]
        for k in np.unique(at):
            sel = g[at == k]
            for value in range(3):
                group_counts[k, :, value] += (sel == value).sum(axis=0)

    sizes = np.bincount(group, minlength=len(keys))
    key_class, key_pattern = keys >> n_folds, keys & ((1 << n_folds) - 1)
    counts = np.zeros((n_folds, n_classes, n_snps, 3), dtype=np.int64)
    n = np.zeros((n_folds, n_classes), dtype=np.int64)
    for f in range(n_folds):
        for c in range(n_classes):
            mine = (key_class == c) & ((key_pattern >> f) & 1).astype(bool)
            n[f, c] = sizes[mine].sum()
            counts[f, c] = group_counts[mine].sum(axis=0)
    return counts, n
