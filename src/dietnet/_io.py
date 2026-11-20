"""File I/O that keeps a cluster job's memory bounded.

On a cluster the pages the kernel caches for a job's reads and writes are charged to the
job's memory limit: read whole, a 19 GiB .bed counted twice (array and cache) and the job
was killed; written through a memory map, a 64 GiB stripe's dirty pages were. These read
into arrays and write from them with plain calls, dropping the pages behind them (Linux;
where ``posix_fadvise`` does not exist, only the sync happens).
"""

from __future__ import annotations

import os

import numpy as np

CHUNK = 2**28


def _drop(fd: int, offset: int = 0, size: int = 0) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)


def read_uncached(path, offset: int, size: int, chunk: int = CHUNK) -> np.ndarray:
    """``size`` bytes of ``path`` from ``offset``, read without keeping them cached."""
    out = np.empty(size, dtype=np.uint8)
    view = memoryview(out)
    with open(path, "rb", buffering=0) as f:
        f.seek(offset)
        done = 0
        while done < size:
            n = f.readinto(view[done : done + chunk])
            if not n:
                raise EOFError(f"{path}: expected {size} bytes after offset {offset}, got {done}")
            _drop(f.fileno(), offset + done, n)
            done += n
    return out


def write_at(fd: int, data, offset: int) -> None:
    """All of ``data`` (an array or buffer) written to ``fd`` at ``offset``."""
    view = memoryview(data).cast("B")
    while len(view):
        n = os.pwrite(fd, view, offset)
        view, offset = view[n:], offset + n


def sync_and_drop(fd: int) -> None:
    """Write ``fd``'s dirty pages out, then drop its cached pages."""
    os.fsync(fd)
    _drop(fd)
