"""Inputs of the real runs' shape, simulated, to time and size every step anywhere.

Writes to OUT_DIR the same files the real runs read (README.md, "Running it"),
with no cohort data involved: ``geno/ukb.{bed,bim,fam}`` (SAMPLES x SNPS,
binomial(2, p) genotypes, ~0.4% missing), ``fields/`` and ``pan/`` for those
samples, and ``alphagenome.tsv.gz`` (SNPS x TRACKS, matched to the ``.bim`` by
position and alleles, ~15% of SNPs strand-ambiguous as on a real array). Every
sample is kept, so ``cohort`` is the fileset itself and no subset is needed.
Reading these costs what the real files would; what is learnt from them means
nothing.

    python simulate.py OUT_DIR --shape 424768 485516 --tracks 3849 --procs 32
"""

import argparse
import gzip
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

MAGIC = bytes([0x6C, 0x1B, 0x01])  # PLINK 1 .bed, SNP-major
# 2-bit codes by dosage of A1: 2 -> 00, missing -> 01, 1 -> 10, 0 -> 11.
CODE = np.array([0b11, 0b10, 0b00, 0b01], dtype=np.uint8)  # index: dosage, 3 = missing
SNP_BLOCK = 512


def _bed_block(path, n_samples, n_snps, start, seed):
    rng = np.random.default_rng(seed)
    stop = min(start + SNP_BLOCK, n_snps)
    p = rng.uniform(0.05, 0.5, stop - start)[:, None]
    r = rng.integers(0, 256, size=(stop - start, n_samples), dtype=np.uint8)
    dosage = (r >= np.round(255 * (1 - p) ** 2)).astype(np.uint8)
    dosage += r >= np.round(255 * (1 - p**2))
    dosage[r == 255] = 3
    codes = CODE[dosage].reshape(stop - start, -1, 4)
    packed = codes[..., 0] | codes[..., 1] << 2 | codes[..., 2] << 4 | codes[..., 3] << 6
    width = n_samples // 4
    bed = np.memmap(path, dtype=np.uint8, mode="r+", offset=3, shape=(n_snps, width))
    bed[start:stop] = packed
    bed.flush()


def _table_block(path, snps, n_tracks, seed):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((len(snps), n_tracks)).astype(np.float32) * 0.01
    tracks = pd.DataFrame(values, columns=[f"T|t{t}" for t in range(n_tracks)])
    rows = pd.concat([snps.reset_index(drop=True), tracks], axis=1)
    with gzip.open(path, "wt", compresslevel=6) as f:
        rows.to_csv(f, sep="\t", index=False, header=False, float_format="%.8g")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--shape", type=int, nargs=2, metavar=("SAMPLES", "SNPS"), required=True)
    parser.add_argument("--tracks", type=int, default=3849)
    parser.add_argument("--procs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    (n_samples, n_snps), out = args.shape, args.out_dir
    if n_samples % 4:
        raise SystemExit("SAMPLES must be a multiple of 4 (whole .bed bytes)")
    for d in ("geno", "fields", "pan", "parts"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    ids = np.arange(1000001, 1000001 + n_samples)
    pd.DataFrame({0: ids, 1: ids, 2: 0, 3: 0, 4: 0, 5: -9}).to_csv(
        out / "geno/ukb.fam", sep=" ", header=False, index=False
    )
    chrom = np.sort(rng.integers(1, 23, n_snps))
    pos = np.zeros(n_snps, dtype=np.int64)
    for c in range(1, 23):
        here = chrom == c
        pos[here] = np.sort(rng.choice(200_000_000, here.sum(), replace=False)) + 1
    pairs = np.array([("A", "C"), ("A", "G"), ("C", "T"), ("G", "T"), ("A", "T"), ("C", "G")])
    kind = rng.choice(6, n_snps, p=[0.2125, 0.2125, 0.2125, 0.2125, 0.075, 0.075])
    flip = rng.random(n_snps) < 0.5
    a1 = np.where(flip, pairs[kind, 1], pairs[kind, 0])
    a2 = np.where(flip, pairs[kind, 0], pairs[kind, 1])
    ids_snp = [f"{c}:{p}:{x}:{y}" for c, p, x, y in zip(chrom, pos, a1, a2)]
    pd.DataFrame({0: chrom, 1: ids_snp, 2: 0, 3: pos, 4: a1, 5: a2}).to_csv(
        out / "geno/ukb.bim", sep="\t", header=False, index=False
    )

    bed = out / "geno/ukb.bed"
    with open(bed, "wb") as f:
        f.write(MAGIC)
        f.truncate(3 + n_snps * (n_samples // 4))
    starts = range(0, n_snps, SNP_BLOCK)
    with ProcessPoolExecutor(args.procs) as pool:
        list(pool.map(_bed_block, *zip(*[(bed, n_samples, n_snps, s, s) for s in starts])))
    print("wrote", bed, bed.stat().st_size)

    values = {
        50: rng.normal(170, 9, n_samples),
        21001: rng.normal(27, 4, n_samples),
        31: rng.integers(0, 2, n_samples).astype(float),
        21022: rng.integers(40, 70, n_samples).astype(float),
        22189: rng.normal(-1, 3, n_samples),
    }
    for field, v in values.items():
        pd.DataFrame(
            {"variable_id": field, "sample_id": ids, "instance": 0, "index": 0, "value": v}
        ).to_csv(out / f"fields/{field}.csv.gz", index=False)
    pan_ids = ids + 5_000_000
    bridge = pd.DataFrame({"ours": ids, "pan": pan_ids})
    bridge.to_csv(out / "pan/ukb20168bridge31063.txt", sep=" ", header=False, index=False)
    related = np.where(rng.random(n_samples) < 0.15, "true", "false")
    pd.DataFrame({"s": pan_ids, "PC1": 0.0, "pop": "EUR", "related": related}).to_csv(
        out / "pan/all_pops_non_eur_pruned_within_pop_pc_covs.tsv", sep="\t", index=False
    )

    # AlphaGenome-shaped table: REF is the .bim's A2, as in the real one.
    snps = pd.DataFrame({"chrom": [f"chr{c}" for c in chrom], "pos": pos, "ref": a2, "alt": a1})
    header = "\t".join([*snps.columns, *[f"T|t{t}" for t in range(args.tracks)]]) + "\n"
    with gzip.open(out / "parts/0000_header.gz", "wt") as f:
        f.write(header)
    chunks = np.array_split(np.arange(n_snps), args.procs * 4)
    jobs = [(out / f"parts/{i + 1:04d}.gz", snps.iloc[c], args.tracks, 10_000 + i)
            for i, c in enumerate(chunks)]  # fmt: skip
    with ProcessPoolExecutor(args.procs) as pool:
        list(pool.map(_table_block, *zip(*jobs)))
    with open(out / "alphagenome.tsv.gz", "wb") as table:  # gzip members concatenate
        for part in sorted((out / "parts").iterdir()):
            with open(part, "rb") as f:
                shutil.copyfileobj(f, table)
    shutil.rmtree(out / "parts")
    print("wrote", out / "alphagenome.tsv.gz", (out / "alphagenome.tsv.gz").stat().st_size)


if __name__ == "__main__":
    main()
