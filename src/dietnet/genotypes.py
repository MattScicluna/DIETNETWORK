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
- ``packed``: a 2-bit copy (:class:`PackedFile`), a quarter of the bytes, decoded per
  batch. Training from disk is limited by how many bytes it reads, so this is the
  fastest store when the data does not fit in memory. The copy is laid out in blocks of
  SNPs, each block a contiguous samples x bytes region: whole rows read every block,
  and training that keeps a share of the SNPs per buffer reads only its blocks.

All give the same rows, so the model trained from any of them is the same.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from dietnet._io import sync_and_drop, write_at
from dietnet.dataset import memmap_inputs

MODES = ("ram", "memmap", "packed")

# "auto" leaves inputs smaller than this in RAM (the threshold the 2022 fork used
# for its PLINK genotype cache); a copy in RAM then costs little and is fastest.
AUTO_MEMMAP_ABOVE_BYTES = 2 * 2**30

# SNP blocks of the packed copy: training's --snp-blocks with the same number reads whole
# blocks only (any other number is read through the blocks that cover its ranges).
PACKED_BLOCKS = 64

# 2-bit codes: dosage 0, 1, 2 as themselves, missing (-1) as 3. Four genotypes per
# byte, the first in the lowest bits; rows padded to whole bytes.
DECODE_TABLE = np.array(
    [[(byte >> (2 * k)) & 3 for k in range(4)] for byte in range(256)], dtype=np.int8
)
DECODE_TABLE[DECODE_TABLE == 3] = -1


def packed_path(dataset) -> Path:
    """Where the 2-bit copy of ``dataset`` is (:class:`PackedWriter`)."""
    return Path(f"{dataset}.packed.bin")


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


def block_edges(n_snps: int, n_blocks: int) -> np.ndarray:
    """SNP edges of ``n_blocks`` contiguous blocks (``n_blocks + 1`` values): multiples of 4
    but the last, so that every block is whole bytes of a 2-bit row."""
    edges = np.linspace(0, n_snps, n_blocks + 1).astype(np.int64) // 4 * 4
    edges[-1] = n_snps
    return edges


def packed_stripes(n_snps: int, max_snps: int) -> list[tuple[int, int]]:
    """Consecutive runs of whole blocks of the packed copy, each of at most ``max_snps``
    SNPs (or one block, if a block alone is more): the pieces a .bed is read in to write
    the copy (:meth:`dietnet.plink.Bed.snp_stripes`)."""
    edges = block_edges(n_snps, PACKED_BLOCKS)
    runs, first = [], 0
    for b in range(1, PACKED_BLOCKS + 1):
        if b == PACKED_BLOCKS or int(edges[b + 1]) - int(edges[first]) > max_snps:
            if edges[b] > edges[first]:
                runs.append((int(edges[first]), int(edges[b])))
            first = b
    return runs


class PackedWriter:
    """Writes the 2-bit copy of a dataset being built (``out``), in bounded memory.

    ``write(start, snp, rows)`` takes int8 rows from sample ``start``, SNPs from ``snp``
    to the end of a block (a piece covers whole blocks); each block's bytes of those rows
    are one write at their place in the block's region, so the file fills sequentially
    within every region and nothing is read back. ``finish()``, once the dataset file is
    in place, moves the copy in and records what it was built from.
    """

    def __init__(self, out, n_samples: int, n_snps: int):
        self.out, self.n_samples, self.n_snps = Path(out), n_samples, n_snps
        self.edges = block_edges(n_snps, PACKED_BLOCKS)
        self.byte_edges = -(-self.edges // 4)
        self.widths = np.diff(self.byte_edges)
        self.offsets = np.r_[0, np.cumsum(self.widths * n_samples)[:-1]]
        self.tmp = packed_path(self.out).with_name(packed_path(self.out).name + ".partial")
        with open(self.tmp, "wb") as f:
            f.truncate(int(self.widths.sum()) * n_samples)
        self._fd = os.open(self.tmp, os.O_WRONLY)

    def write(self, start: int, snp: int, rows: np.ndarray) -> None:
        packed = encode(rows)
        lo = int(self.byte_edges[np.searchsorted(self.edges, snp)])
        for b in range(PACKED_BLOCKS):
            b0, b1 = int(self.byte_edges[b]), int(self.byte_edges[b + 1])
            if b1 <= lo or b0 >= lo + packed.shape[1] or b1 == b0:
                continue
            piece = np.ascontiguousarray(packed[:, b0 - lo : b1 - lo])
            write_at(self._fd, piece, int(self.offsets[b] + start * self.widths[b]))
        sync_and_drop(self._fd)

    def finish(self) -> Path:
        os.close(self._fd)
        os.replace(self.tmp, packed_path(self.out))
        meta = {"source": _source(self.out), "edges": self.edges.tolist(),
                "byte_edges": self.byte_edges.tolist(),
                "offsets": [int(o) for o in self.offsets]}  # fmt: skip
        _meta_path(self.out).write_text(json.dumps(meta))
        return packed_path(self.out)


def pack(dataset, block: int = 4096) -> Path:
    """Write the 2-bit copy of a dataset that has int8 ``inputs``, ``block`` samples at a
    time; return its path. (``create-dataset`` writes it directly from PLINK.)"""
    dataset = Path(dataset)
    require_inputs(dataset, "pack-genotypes")
    with h5py.File(dataset, "r") as f:
        inputs = f["inputs"]
        writer = PackedWriter(dataset, *inputs.shape)
        for start in range(0, inputs.shape[0], block):
            writer.write(start, 0, inputs[start : start + block])
    return writer.finish()


def inputs_stored(dataset) -> bool:
    """Whether ``dataset`` stores its genotypes as int8 ``inputs``, as upstream's datasets
    do; those of ``create-dataset`` are only in the 2-bit copy."""
    with h5py.File(dataset, "r") as f:
        return bool(f.attrs.get("inputs_stored", True))


def require_inputs(dataset, what: str) -> None:
    """Refuse ``what`` on a dataset whose genotypes are only in the 2-bit copy."""
    if not inputs_stored(dataset):
        raise ValueError(
            f"{what} reads the int8 genotypes, but {dataset} has them only in its packed "
            "copy: read them with --inputs packed"
        )


def is_packed(dataset) -> bool:
    """Whether ``dataset`` has a 2-bit copy built from it as it is now."""
    meta = _meta_path(dataset)
    if not (packed_path(dataset).exists() and meta.exists()):
        return False
    return json.loads(meta.read_text()).get("source") == _source(dataset)


def _read_rows(f, offset: int, row_bytes: int, rows: np.ndarray, max_gap: int) -> np.ndarray:
    """Rows ``rows`` (ascending runs) of a samples x ``row_bytes`` region at ``offset``.
    Rows within ``max_gap`` of each other are read as one span and the rows between
    discarded: a few large reads instead of many small ones, which is what a parallel file
    system serves fast."""
    out = np.empty((len(rows), row_bytes), dtype=np.uint8)
    if not len(rows) or not row_bytes:
        return out
    step = np.diff(rows)
    breaks = np.flatnonzero((step < 1) | (step > max_gap + 1)) + 1
    for start, stop in zip(np.r_[0, breaks], np.r_[breaks, len(rows)]):
        first, last = int(rows[start]), int(rows[stop - 1])
        span = np.empty((last - first + 1, row_bytes), dtype=np.uint8)
        f.seek(offset + first * row_bytes)
        view, done = memoryview(span).cast("B"), 0
        while done < len(view):
            n = f.readinto(view[done:])
            if not n:
                raise EOFError(f"{f.name}: short read at row {first}")
            done += n
        out[start:stop] = span[rows[start:stop] - first]
    return out


class PackedFile:
    """Rows of a dataset's 2-bit copy, read with plain file reads: ``read(rows)`` is their
    bytes; ``read(rows, ranges=...)`` only those of the SNP ranges (starts multiples of 4),
    side by side, from the blocks that cover them.

    Indexing a memory map instead turns each row into page faults of 4 KB, each a round
    trip: on Lustre, validation's ~105k scattered rows went at ~3 MB/s. The file is opened
    lazily, once per process (loader workers fork).
    """

    def __init__(self, dataset):
        if not is_packed(dataset):
            raise ValueError(
                f"{dataset} has no up-to-date packed copy; run `dietnet pack-genotypes`"
            )
        meta = json.loads(_meta_path(dataset).read_text())
        self.path = packed_path(dataset)
        self.n_rows, self.n_snps = meta["source"]["n_samples"], meta["source"]["n_snps"]
        self.byte_edges = np.asarray(meta["byte_edges"])
        self.offsets = np.asarray(meta["offsets"])
        self.row_bytes = int(self.byte_edges[-1])
        self._file, self._pid = None, None

    def __getstate__(self):
        return {**self.__dict__, "_file": None, "_pid": None}

    def _handle(self):
        if self._file is None or self._pid != os.getpid():
            self._file, self._pid = open(self.path, "rb", buffering=0), os.getpid()
        return self._file

    def __getitem__(self, rows) -> np.ndarray:
        return self.read(rows)

    def read(self, rows, max_gap: int = 0, ranges=None) -> np.ndarray:
        scalar = np.ndim(rows) == 0
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        if len(rows) and (rows.min() < 0 or rows.max() >= self.n_rows):
            raise IndexError(f"rows out of range 0..{self.n_rows - 1}")
        spans = [(0, self.row_bytes)] if ranges is None else [
            (int(a) // 4, -(-int(b) // 4)) for a, b in ranges
        ]  # fmt: skip
        if ranges is not None and any(int(a) % 4 for a, _ in ranges):
            raise ValueError("SNP ranges must start on a byte: a multiple of 4")
        be, f, parts = self.byte_edges, self._handle(), []
        for lo, hi in spans:
            for b in np.flatnonzero((be[:-1] < hi) & (be[1:] > lo) & (be[1:] > be[:-1])):
                block = _read_rows(f, int(self.offsets[b]), int(be[b + 1] - be[b]), rows, max_gap)
                parts.append(block[:, max(lo, be[b]) - be[b] : min(hi, be[b + 1]) - be[b]])
        out = np.concatenate(parts, axis=1) if parts else np.empty((len(rows), 0), np.uint8)
        return out[0] if scalar else out


class CachedRows:
    """The rows every epoch reads again (validation, the batch-norm recalibration subset,
    the test set), read once into a file in ``directory`` and memory-mapped: one pass over
    the genotype file, then pages the OS keeps in memory while it can.

    The file is named by ``ident`` (what the rows are of: :func:`dataset_ident`) and the
    rows themselves, so a later run asking for the same rows of the same data maps it
    instead of reading them again; it is put in place only once complete. ``read`` is
    called ``batch`` sorted rows at a time (e.g. ``PackedFile.read`` with a ``max_gap``)
    and each batch is written and synced, so memory stays bounded while the file fills.
    """

    def __init__(self, read, rows, directory, ident: str, batch: int = 512):
        self.rows = np.unique(np.asarray(rows, dtype=np.int64))
        key = hashlib.sha256(ident.encode() + self.rows.tobytes()).hexdigest()[:24]
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"cached_rows_{key}.npy"
        self.reused = self.path.exists()
        if not self.reused:
            self._write(read, batch)
        self.data = np.load(self.path, mmap_mode="r")
        if len(self.data) != len(self.rows):
            raise ValueError(f"{self.path}: {len(self.data)} rows cached, {len(self.rows)} asked")

    def _write(self, read, batch: int) -> None:
        handle, tmp = tempfile.mkstemp(
            suffix=".partial", prefix="cached_rows_", dir=self.path.parent
        )
        os.close(handle)
        first = np.asarray(read(self.rows[:batch]))
        header = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=first.dtype, shape=(len(self.rows), *first.shape[1:])
        )
        offset, row_bytes = header.offset, first[0].nbytes if len(first) else 0
        del header
        fd = os.open(tmp, os.O_WRONLY)
        try:
            for start in range(0, len(self.rows), batch):
                block = first if start == 0 else np.asarray(read(self.rows[start : start + batch]))
                write_at(fd, np.ascontiguousarray(block), offset + start * row_bytes)
                os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self.path)

    def __getitem__(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        at = np.searchsorted(self.rows, rows)
        if np.any(at >= len(self.rows)) or np.any(
            self.rows[np.minimum(at, len(self.rows) - 1)] != rows
        ):
            raise KeyError("rows not in the cache")
        return self.data[at]


def dataset_ident(dataset) -> str:
    """What identifies a dataset's genotypes for caching: its shape, size and mtime."""
    return json.dumps(_source(dataset), sort_keys=True)


class PackedStore:
    """int8 rows decoded from the 2-bit copy (:class:`PackedFile`).

    ``store[i]`` is one row, ``store[indices]`` a block of rows, as for a numpy array;
    ``reader`` gives the undecoded bytes, for decoding on a GPU.
    """

    def __init__(self, dataset):
        self.dataset = Path(dataset)
        self.reader = PackedFile(dataset)
        self.n_snps = self.reader.n_snps
        self.shape = (self.reader.n_rows, self.n_snps)
        self.dtype = np.dtype(np.int8)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows) -> np.ndarray:
        if np.ndim(rows) == 0:
            return decode(self.reader[int(rows)][None], self.n_snps)[0]
        return decode(self.reader[rows], self.n_snps)


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
    if mode != "packed" and not (mode == "auto" and is_packed(dataset)):
        require_inputs(dataset, f"--inputs {mode}")
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
