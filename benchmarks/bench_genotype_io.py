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
second read of anything is a memory copy. Each measurement is the median of
``--repeats`` runs and the first, cold-ish run is reported separately. On a
shared cluster filesystem run it on the real data, on the node type that will
train -- laptop SSD numbers do not transfer to Lustre.

Usage::

    python benchmarks/bench_genotype_io.py --plink-prefix DATA/1000G... --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

BATCH = 138  # the 1KGP config's batch size
N_BATCHES = 50


def decode_bed(prefix: str) -> np.ndarray:
    """Whole .bed to (samples x SNPs) int8 dosages of A1, -1 missing. Vectorised."""
    n_samples = sum(1 for _ in open(f"{prefix}.fam"))
    n_snps = sum(1 for _ in open(f"{prefix}.bim"))
    bytes_per_snp = (n_samples + 3) // 4

    raw = np.fromfile(f"{prefix}.bed", dtype=np.uint8)
    if raw[:3].tolist() != [0x6C, 0x1B, 0x01]:
        raise ValueError("not a SNP-major .bed file")
    packed = raw[3:].reshape(n_snps, bytes_per_snp)

    # Two bits per genotype, low bits first. 00 hom A1, 01 missing, 10 het, 11 hom A2.
    table = np.array([2, -1, 1, 0], dtype=np.int8)
    codes = np.stack([(packed >> shift) & 3 for shift in (0, 2, 4, 6)], axis=-1)
    dosages = table[codes].reshape(n_snps, -1)[:, :n_samples]
    return np.ascontiguousarray(dosages.T)


def read_pyplink(prefix: str) -> np.ndarray:
    from pyplink import PyPlink

    with PyPlink(prefix) as bed:
        out = np.empty((bed.get_nb_samples(), bed.get_nb_markers()), dtype=np.int8)
        for j, (_, genotypes) in enumerate(bed):
            out[:, j] = genotypes
    return out


def timed(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return times, result


def random_batches(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_samples, BATCH, replace=False)) for _ in range(N_BATCHES)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plink-prefix", required=True)
    parser.add_argument("--workdir", type=Path, default=None, help="where to write the formats")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-pyplink", action="store_true", help="it takes minutes at scale")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="dietnet_bench_"))
    workdir.mkdir(parents=True, exist_ok=True)

    reference = decode_bed(args.plink_prefix)
    n_samples, n_snps = reference.shape
    batches = random_batches(n_samples)
    print(f"{n_samples} samples x {n_snps} SNPs ({reference.nbytes / 2**20:.0f} MiB int8)")

    files = {"bed": Path(f"{args.plink_prefix}.bed")}
    build = {}

    def make(name, writer):
        start = time.perf_counter()
        files[name] = writer()
        build[name] = time.perf_counter() - start

    make("npy", lambda: (np.save(workdir / "g.npy", reference), workdir / "g.npy")[1])

    def h5(name, **kwargs):
        def write():
            path = workdir / f"{name}.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("inputs", data=reference, **kwargs)
            return path

        make(name, write)

    row_chunks = (min(64, n_samples), n_snps)
    h5("hdf5-contiguous")
    h5("hdf5-chunked-rows", chunks=row_chunks)
    h5("hdf5-chunked-rows-lzf", chunks=row_chunks, compression="lzf")
    h5("hdf5-chunked-rows-gzip", chunks=row_chunks, compression="gzip", compression_opts=4)

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

    full = {
        "bed-numpy": lambda: decode_bed(args.plink_prefix),
        "npy": lambda: np.load(files["npy"]),
        **{n: h5_full(n) for n in files if n.startswith("hdf5")},
    }
    if not args.skip_pyplink:
        full = {"bed-pyplink": lambda: read_pyplink(args.plink_prefix), **full}

    batched = {
        "npy-memmap": memmap_batches,
        **{n: h5_batches(n) for n in files if n.startswith("hdf5")},
    }

    results = {
        "platform": f"{platform.system()} {platform.machine()}",
        "n_samples": n_samples,
        "n_snps": n_snps,
        "batch": BATCH,
        "n_batches": N_BATCHES,
        "build_seconds": build,
        "size_mib": {n: os.path.getsize(p) / 2**20 for n, p in files.items()},
        "full_load": {},
        "random_batches": {},
    }

    for name, fn in full.items():
        times, matrix = timed(fn, args.repeats)
        assert np.array_equal(matrix, reference), f"{name} returned different genotypes"
        results["full_load"][name] = {"first": times[0], "median": statistics.median(times)}
        print(
            f"full load      {name:24s} first {times[0]:7.2f}s  median {statistics.median(times):7.2f}s"
        )

    for name, fn in batched.items():
        times, got = timed(fn, args.repeats)
        assert all(np.array_equal(g, reference[idx]) for g, idx in zip(got, batches)), name
        results["random_batches"][name] = {"first": times[0], "median": statistics.median(times)}
        print(
            f"{N_BATCHES} batches     {name:24s} first {times[0]:7.2f}s  median {statistics.median(times):7.2f}s"
        )

    for name, mib in results["size_mib"].items():
        print(f"size           {name:24s} {mib:8.1f} MiB  build {build.get(name, 0):6.2f}s")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
