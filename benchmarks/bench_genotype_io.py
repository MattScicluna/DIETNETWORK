"""How fast can each on-disk format hand the trainer its genotype matrix?

Settles a design question in docs/refactor-plan.md: which intermediate the
pipeline should build once from PLINK and read in every later step. Candidates:

* ``bed-pyplink``: today's path below 2 GB -- pyplink, one marker at a time,
  repeated by every command because nothing is cached.
* ``bed-numpy``: the same .bed decoded in bulk with a lookup table, to separate
  "PLINK is slow" from "pyplink's per-marker loop is slow".
* ``npy`` / ``npy-memmap``: today's cache above 2 GB.
* ``hdf5-contiguous``: what upstream's create_dataset.py writes.
* ``hdf5-chunked-rows``, ``-lzf``, ``-gzip``: sample-major chunks, the layout a
  minibatch read wants, with and without compression.

Two access patterns, the ones the pipeline has: load the whole matrix (training
keeps it in RAM), and read random minibatches of samples straight from disk
(what a dataset too big for RAM would need).

Page cache: the operating system keeps recently read files in memory, so the
second read of anything is a memory copy. Each measurement reports the first
run separately from the median. On a cluster, run it as a batch job on the node
type that will do the work, with ``--workdir`` on the filesystem you mean to use
(``$SLURM_TMPDIR`` for node-local disk, scratch for Lustre) -- laptop numbers do
not transfer.

At biobank scale (UK Biobank: ~487k samples, an 80+ GB int8 matrix) only one
full copy of the matrix is held at a time besides the reference, so ask for
about 2.2x the matrix in memory, pass ``--repeats 1`` and pick ``--formats``:
the compressed and chunked layouts were already ruled out at 1KGP scale.

Usage::

    python benchmarks/bench_genotype_io.py --plink-prefix DATA/1000G... --out results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

BATCH = 138  # the 1KGP config's batch size
N_BATCHES = 50
VERIFY_ROWS = 256

ALL_FORMATS = (
    "bed-pyplink",
    "bed-numpy",
    "npy",
    "hdf5-contiguous",
    "hdf5-chunked-rows",
    "hdf5-chunked-rows-lzf",
    "hdf5-chunked-rows-gzip",
)

# Two bits per genotype, low bits first. 00 hom A1, 01 missing, 10 het, 11 hom A2.
BED_CODES = np.array([2, -1, 1, 0], dtype=np.int8)


def count_lines(path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def decode_bed(prefix: str, block: int = 4096) -> np.ndarray:
    """Whole .bed to (samples x SNPs) int8 dosages of A1, -1 missing.

    Vectorised, a block of SNPs at a time into one preallocated array, so peak
    memory is the output plus one block rather than several full-size copies.
    """
    n_samples = count_lines(f"{prefix}.fam")
    n_snps = count_lines(f"{prefix}.bim")
    bytes_per_snp = (n_samples + 3) // 4

    packed = np.memmap(f"{prefix}.bed", dtype=np.uint8, mode="r")
    if packed[:3].tolist() != [0x6C, 0x1B, 0x01]:
        raise ValueError("not a SNP-major .bed file")
    packed = packed[3:].reshape(n_snps, bytes_per_snp)

    out = np.empty((n_samples, n_snps), dtype=np.int8)
    for start in range(0, n_snps, block):
        chunk = np.asarray(packed[start : start + block])
        codes = np.stack([(chunk >> shift) & 3 for shift in (0, 2, 4, 6)], axis=-1)
        dosages = BED_CODES[codes].reshape(len(chunk), -1)[:, :n_samples]
        out[:, start : start + len(chunk)] = dosages.T
    return out


def read_pyplink(prefix: str) -> np.ndarray:
    from pyplink import PyPlink

    with PyPlink(prefix) as bed:
        out = np.empty((bed.get_nb_samples(), bed.get_nb_markers()), dtype=np.int8)
        for j, (_, genotypes) in enumerate(bed):
            out[:, j] = genotypes
    return out


def peak_rss_gib() -> float:
    """Peak resident memory of this process so far. ru_maxrss is KiB on Linux, bytes on macOS."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**30 if sys.platform == "darwin" else 2**20)


def random_batches(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_samples, BATCH, replace=False)) for _ in range(N_BATCHES)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plink-prefix", required=True)
    parser.add_argument("--workdir", type=Path, default=None, help="where to write the formats")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--formats",
        default=",".join(ALL_FORMATS),
        help=f"comma-separated subset of: {', '.join(ALL_FORMATS)}",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = set(formats) - set(ALL_FORMATS)
    if unknown:
        parser.error(f"unknown formats: {sorted(unknown)}")

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="dietnet_bench_"))
    workdir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    reference = decode_bed(args.plink_prefix)
    first_decode = time.perf_counter() - start
    n_samples, n_snps = reference.shape
    batches = random_batches(n_samples)
    rng = np.random.default_rng(1)
    check_rows = np.sort(rng.choice(n_samples, min(VERIFY_ROWS, n_samples), replace=False))
    print(
        f"{n_samples} samples x {n_snps} SNPs ({reference.nbytes / 2**30:.1f} GiB int8); "
        f"first decode {first_decode:.1f}s",
        flush=True,
    )

    files = {"bed": Path(f"{args.plink_prefix}.bed")}
    build = {}

    def make(name, writer):
        start = time.perf_counter()
        files[name] = writer()
        build[name] = time.perf_counter() - start
        print(f"built          {name:24s} {build[name]:7.1f}s", flush=True)

    if "npy" in formats:
        make("npy", lambda: (np.save(workdir / "g.npy", reference), workdir / "g.npy")[1])

    row_chunks = (min(64, n_samples), n_snps)
    h5_layouts = {
        "hdf5-contiguous": {},
        "hdf5-chunked-rows": dict(chunks=row_chunks),
        "hdf5-chunked-rows-lzf": dict(chunks=row_chunks, compression="lzf"),
        "hdf5-chunked-rows-gzip": dict(chunks=row_chunks, compression="gzip", compression_opts=4),
    }
    for name, layout in h5_layouts.items():
        if name not in formats:
            continue

        def write(name=name, layout=layout):
            path = workdir / f"{name}.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("inputs", data=reference, **layout)
            return path

        make(name, write)

    def h5_full(name):
        def read():
            with h5py.File(files[name], "r") as f:
                return f["inputs"][:]

        return read

    def h5_batches(name):
        def read():
            with h5py.File(files[name], "r") as f:
                d = f["inputs"]
                return [d[idx] for idx in batches]

        return read

    def memmap_batches():
        m = np.load(files["npy"], mmap_mode="r")
        return [np.asarray(m[idx]) for idx in batches]

    full = {}
    if "bed-pyplink" in formats:
        full["bed-pyplink"] = lambda: read_pyplink(args.plink_prefix)
    if "bed-numpy" in formats:
        full["bed-numpy"] = lambda: decode_bed(args.plink_prefix)
    if "npy" in formats:
        full["npy"] = lambda: np.load(files["npy"])
    full.update({n: h5_full(n) for n in files if n.startswith("hdf5")})

    batched = {}
    if "npy" in formats:
        batched["npy-memmap"] = memmap_batches
    batched.update({n: h5_batches(n) for n in files if n.startswith("hdf5")})

    results = {
        "platform": f"{platform.system()} {platform.machine()}",
        "host": platform.node(),
        "workdir": str(workdir),
        "n_samples": n_samples,
        "n_snps": n_snps,
        "batch": BATCH,
        "n_batches": N_BATCHES,
        "first_decode_seconds": first_decode,
        "build_seconds": build,
        "size_gib": {n: os.path.getsize(p) / 2**30 for n, p in files.items()},
        "full_load": {},
        "random_batches": {},
    }

    def report(kind, name, times):
        results[kind][name] = {"first": times[0], "median": statistics.median(times)}
        label = "full load" if kind == "full_load" else f"{N_BATCHES} batches"
        print(
            f"{label:14s} {name:24s} first {times[0]:8.2f}s  "
            f"median {statistics.median(times):8.2f}s  peak RSS {peak_rss_gib():6.1f} GiB",
            flush=True,
        )

    for name, fn in full.items():
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            matrix = fn()
            times.append(time.perf_counter() - start)
            # Sampled rows, not the whole matrix: a full comparison would need a
            # third full-size copy at biobank scale.
            assert matrix.shape == reference.shape, name
            assert np.array_equal(matrix[check_rows], reference[check_rows]), name
            del matrix
            gc.collect()
        report("full_load", name, times)

    for name, fn in batched.items():
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            got = fn()
            times.append(time.perf_counter() - start)
        assert all(np.array_equal(g, reference[idx]) for g, idx in zip(got, batches)), name
        report("random_batches", name, times)

    for name, gib in results["size_gib"].items():
        print(f"size           {name:24s} {gib:8.2f} GiB  build {build.get(name, 0):7.1f}s")

    results["peak_rss_gib"] = peak_rss_gib()
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
