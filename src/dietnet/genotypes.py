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
import tempfile
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
    require_inputs(dataset, "pack-genotypes")
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
    return finish_pack(dataset, tmp, source)


def _sync_and_drop(fd: int) -> None:
    """Write ``fd``'s dirty pages out and drop its cached pages: on a cluster both count
    against the job's memory limit (Linux; elsewhere only synced)."""
    os.fsync(fd)
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def _pwrite_all(fd: int, data, offset: int) -> None:
    view = memoryview(data).cast("B")
    while len(view):
        n = os.pwrite(fd, view, offset)
        view, offset = view[n:], offset + n


def _pread_all(fd: int, size: int, offset: int) -> np.ndarray:
    out = np.empty(size, dtype=np.uint8)
    view, done = memoryview(out), 0
    while done < size:
        n = os.preadv(fd, [view[done:]], offset + done)
        if not n:
            raise EOFError(f"expected {size} bytes at offset {offset}, got {done}")
        done += n
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
    return out


def pack_bed(bed, path, load_bytes: int, block_bytes: int) -> None:
    """Write the 2-bit copy of a PLINK fileset's genotypes (samples x SNPs) to ``path``,
    from :meth:`dietnet.plink.Bed.snp_stripes`, in bounded memory and sequential I/O only.

    Writing each stripe's bytes into every row of the final file would be strided writes
    that start mid-page, which a shared filesystem serves with a read per row. Instead:
    each stripe goes to its own file, sample-major and appended in order; then the final
    rows are assembled a block at a time from the matching range of every stripe file and
    written contiguously. Takes the final file's size again in temporary space, removed at
    the end.
    """
    path = Path(path)
    parts, fd = [], None  # (file, bytes per row) per stripe
    ins, out = [], None
    try:
        for start, _, block in bed.snp_stripes(load_bytes, block_bytes):
            if start == 0:  # a new stripe
                if fd is not None:
                    _sync_and_drop(fd)
                    os.close(fd)
                part = path.with_name(f"{path.name}.stripe{len(parts)}")
                fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
                parts.append((part, -(-block.shape[1] // 4)))
            _pwrite_all(fd, encode(block), os.lseek(fd, 0, os.SEEK_END))
            _sync_and_drop(fd)
        if fd is not None:
            _sync_and_drop(fd)
            os.close(fd)
            fd = None

        n, width = bed.n_samples, sum(w for _, w in parts)
        header = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n, width))
        offset = header.offset
        del header
        out = os.open(path, os.O_WRONLY)
        ins = [os.open(part, os.O_RDONLY) for part, _ in parts]
        rows = max(1, block_bytes // max(width, 1))
        for r0 in range(0, n, rows):
            r1 = min(r0 + rows, n)
            pieces = [_pread_all(f, (r1 - r0) * w, r0 * w).reshape(r1 - r0, w)
                      for f, (_, w) in zip(ins, parts)]  # fmt: skip
            _pwrite_all(out, np.concatenate(pieces, axis=1), offset + r0 * width)
            _sync_and_drop(out)
    finally:
        for f in [fd, out, *ins]:
            if f is not None:
                os.close(f)
        for part, _ in parts:
            part.unlink(missing_ok=True)


def finish_pack(dataset, written: Path, source: dict | None = None) -> Path:
    """Put a fully written 2-bit copy of ``dataset`` in place and record what it was built
    from (``source``, by default the dataset as it is now)."""
    out = packed_path(dataset)
    os.replace(written, out)
    _meta_path(dataset).write_text(json.dumps(source or _source(dataset)))
    return out


def inputs_stored(dataset) -> bool:
    """Whether ``dataset`` stores its genotypes as int8 ``inputs``: not when built with
    ``create-dataset --packed-only``, which keeps them only in the 2-bit copy."""
    with h5py.File(dataset, "r") as f:
        return bool(f.attrs.get("inputs_stored", True))


def require_inputs(dataset, what: str) -> None:
    """Refuse ``what`` on a dataset whose genotypes are only in the 2-bit copy."""
    if not inputs_stored(dataset):
        raise ValueError(
            f"{what} reads the int8 genotypes, but {dataset} has them only in its packed "
            "copy (create-dataset --packed-only): read them with --inputs packed"
        )


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


def block_edges(n_snps: int, n_blocks: int) -> np.ndarray:
    """SNP edges of ``n_blocks`` contiguous blocks (``n_blocks + 1`` values): multiples of 4
    but the last, so that every block is whole bytes of a 2-bit row."""
    edges = np.linspace(0, n_snps, n_blocks + 1).astype(np.int64) // 4 * 4
    edges[-1] = n_snps
    return edges


def blocks_path(dataset, n_blocks: int) -> Path:
    """Where :func:`pack_blocks` writes the blocked 2-bit copy of ``dataset``."""
    return Path(f"{dataset}.packed{n_blocks}b.bin")


def _blocks_meta_path(dataset, n_blocks: int) -> Path:
    return Path(f"{blocks_path(dataset, n_blocks)}.json")


def pack_blocks(dataset, n_blocks: int, block_bytes: int = 2**30) -> Path:
    """Write the 2-bit copy again, cut into ``n_blocks`` blocks of SNPs (:func:`block_edges`),
    each block stored as its own contiguous samples x bytes region.

    Training with ``--snp-blocks`` holds a share of the blocks per buffer; from this copy it
    reads only those blocks, one contiguous span of rows per block, instead of whole rows
    of which it keeps that share. Built from the row-wise copy (:func:`pack`), read in
    order a block of rows at a time; each block's slice of those rows is one write.
    """
    dataset = Path(dataset)
    if not is_packed(dataset):
        raise ValueError(f"{dataset} has no up-to-date packed copy; run `dietnet pack-genotypes`")
    reader = PackedReader.of(packed_path(dataset))
    n, row_bytes = reader.n_rows, reader.row_bytes
    n_snps = json.loads(_meta_path(dataset).read_text())["n_snps"]
    edges = block_edges(n_snps, n_blocks)
    byte_edges = np.minimum(-(-edges // 4), row_bytes)
    widths = np.diff(byte_edges)
    offsets = np.r_[0, np.cumsum(widths * n)[:-1]]
    out = blocks_path(dataset, n_blocks)
    tmp = out.with_name(out.name + ".partial")
    with open(tmp, "wb") as f:
        f.truncate(int(widths.sum()) * n)
    fd = os.open(tmp, os.O_WRONLY)
    try:
        rows_per = max(1, block_bytes // max(row_bytes, 1))
        for r0 in range(0, n, rows_per):
            r1 = min(r0 + rows_per, n)
            rows = reader.read(np.arange(r0, r1))
            for b in range(n_blocks):
                part = np.ascontiguousarray(rows[:, byte_edges[b] : byte_edges[b + 1]])
                _pwrite_all(fd, part, int(offsets[b] + r0 * widths[b]))
            os.fsync(fd)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    os.replace(tmp, out)
    meta = {"source": _source(dataset), "n_samples": int(n), "n_snps": int(n_snps),
            "edges": edges.tolist(), "byte_edges": byte_edges.tolist(),
            "offsets": [int(o) for o in offsets]}  # fmt: skip
    _blocks_meta_path(dataset, n_blocks).write_text(json.dumps(meta))
    return out


def is_blocked(dataset, n_blocks: int) -> bool:
    """Whether ``dataset`` has a blocked copy of ``n_blocks`` blocks built from it as it is."""
    meta = _blocks_meta_path(dataset, n_blocks)
    if not (blocks_path(dataset, n_blocks).exists() and meta.exists()):
        return False
    return json.loads(meta.read_text())["source"] == _source(dataset)


class BlockedReader:
    """Rows of a blocked 2-bit copy (:func:`pack_blocks`), only of the blocks asked for:
    ``read(rows, blocks)`` is the packed rows' bytes of those blocks, side by side, as
    slicing whole packed rows would give."""

    def __init__(self, dataset, n_blocks: int):
        if not is_blocked(dataset, n_blocks):
            raise ValueError(f"{dataset} has no up-to-date copy in {n_blocks} blocks")
        meta = json.loads(_blocks_meta_path(dataset, n_blocks).read_text())
        self.edges = np.asarray(meta["edges"])
        widths = np.diff(meta["byte_edges"])
        path = blocks_path(dataset, n_blocks)
        self.readers = [PackedReader(path, meta["n_samples"], int(w), int(o))
                        for w, o in zip(widths, meta["offsets"])]  # fmt: skip

    def block_of(self, ranges) -> np.ndarray:
        """The block numbers of SNP ranges that are whole blocks."""
        starts = np.asarray(ranges)[:, 0]
        at = np.searchsorted(self.edges, starts)
        if not np.array_equal(self.edges[at], starts) or not np.array_equal(
            self.edges[at + 1], np.asarray(ranges)[:, 1]
        ):
            raise ValueError("SNP ranges are not this copy's blocks")
        return at

    def read(self, rows, blocks, max_gap: int = 0) -> np.ndarray:
        return np.concatenate([self.readers[b].read(rows, max_gap) for b in blocks], axis=1)


class CachedRows:
    """Some rows of a genotype source held in memory: ``cache[rows]`` for any of them.

    For the rows every epoch reads again (validation, the BN recalibration subset,
    the test set): one pass over the file, then no disk at all. ``read`` is the
    source's reader, called once with the sorted rows (e.g. ``PackedReader.read``
    with a ``max_gap``, so the pass is a few large reads).

    With ``directory``, the rows are held in a file there instead, memory-mapped: a node's
    local disk, when they would not fit in the job's memory (at 9.1M SNPs, validation and
    test are ~190 GB each, packed). They are read ``batch`` rows at a time and written
    sequentially, each batch synced, so memory stays bounded while the file fills; its
    pages are then clean, cached while memory allows and dropped when it does not.
    """

    def __init__(self, read, rows, directory=None, batch: int = 512, key: str | None = None):
        self.rows = np.unique(np.asarray(rows, dtype=np.int64))
        self.path, self.reused = None, False
        if directory is None:
            self.data = np.asarray(read(self.rows))
            return
        # With a key (what the rows are of, and which), the file is named by it and reused
        # by later runs; it is put in place only once complete.
        final = Path(directory) / f"cached_rows_{key}.npy" if key else None
        if final is not None and final.exists():
            self.path, self.reused = final, True
            self.data = np.load(final, mmap_mode="r")
            if len(self.data) != len(self.rows):
                raise ValueError(f"{final}: {len(self.data)} rows cached, {len(self.rows)} asked")
            return
        handle, path = tempfile.mkstemp(suffix=".partial", prefix="cached_rows_", dir=directory)
        os.close(handle)
        self.path = Path(path)
        first = np.asarray(read(self.rows[:batch]))
        header = np.lib.format.open_memmap(
            self.path, mode="w+", dtype=first.dtype, shape=(len(self.rows), *first.shape[1:])
        )
        offset, row_bytes = header.offset, first[0].nbytes if len(first) else 0
        del header
        fd = os.open(self.path, os.O_WRONLY)
        try:
            for start in range(0, len(self.rows), batch):
                block = first if start == 0 else np.asarray(read(self.rows[start : start + batch]))
                _pwrite_all(fd, np.ascontiguousarray(block), offset + start * row_bytes)
                os.fsync(fd)
        finally:
            os.close(fd)
        if final is not None:
            os.replace(self.path, final)
            self.path = final
        else:
            renamed = self.path.with_suffix(".npy")
            os.replace(self.path, renamed)
            self.path = renamed
        self.data = np.load(self.path, mmap_mode="r")

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
