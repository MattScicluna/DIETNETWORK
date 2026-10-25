"""Read PLINK 1 binary filesets (.bed/.bim/.fam) without third-party readers.

Dosages count the .bim A1 allele (0, 1, 2) with -1 for a missing call: what
``plink --recode A`` writes, and so what upstream's HDF5 datasets hold
(tests/parity checks it). The .bed is decoded in blocks of samples with a lookup
table, so a biobank-sized file is read in bounded memory and written out
sample-major, one contiguous run of rows per block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

MAGIC = (0x6C, 0x1B, 0x01)  # SNP-major
# Two bits per genotype, low bits first: 00 hom A1, 01 missing, 10 het, 11 hom A2.
CODES = np.array([2, -1, 1, 0], dtype=np.int8)
# Every byte value -> its four samples' dosages: one table lookup per byte.
BYTE_TO_DOSAGES = CODES[np.stack([(np.arange(256) >> s) & 3 for s in (0, 2, 4, 6)], axis=-1)]
# A block of samples holds at most this many dosages (int8 bytes), decoded a
# chunk of SNPs at a time.
BLOCK_BYTES = 2**32
SNP_CHUNK = 8192
# Packed .bed files up to this size are read into memory once (see Bed._packed).
LOAD_BYTES = 64 * 2**30


def _prefix(prefix) -> str:
    prefix = str(prefix)
    return prefix[:-4] if prefix.endswith((".bed", ".bim", ".fam")) else prefix


def read_fam(prefix) -> pd.DataFrame:
    return pd.read_csv(
        f"{_prefix(prefix)}.fam",
        sep=r"\s+",
        header=None,
        names=["fid", "iid", "father", "mother", "sex", "phenotype"],
        dtype=str,
    )


def read_bim(prefix) -> pd.DataFrame:
    bim = pd.read_csv(
        f"{_prefix(prefix)}.bim",
        sep=r"\s+",
        header=None,
        names=["chrom", "snp", "cm", "pos", "a1", "a2"],
        dtype={"chrom": str, "snp": str, "a1": str, "a2": str},
    )
    bim["chrom"] = bim["chrom"].str.removeprefix("chr")
    return bim


@dataclass(frozen=True)
class Bed:
    """A PLINK fileset opened for block reads."""

    prefix: str
    n_samples: int
    n_snps: int

    @classmethod
    def open(cls, prefix) -> "Bed":
        prefix = _prefix(prefix)
        with open(f"{prefix}.fam", "rb") as f:
            n_samples = sum(1 for _ in f)
        with open(f"{prefix}.bim", "rb") as f:
            n_snps = sum(1 for _ in f)
        bed = cls(prefix, n_samples, n_snps)
        header = np.fromfile(f"{prefix}.bed", dtype=np.uint8, count=3)
        if tuple(header) != MAGIC:
            raise ValueError(f"{prefix}.bed is not a SNP-major PLINK 1 .bed file")
        expected = 3 + n_snps * bed.bytes_per_snp
        actual = Path(f"{prefix}.bed").stat().st_size
        if actual != expected:
            raise ValueError(
                f"{prefix}.bed has {actual} bytes; {n_samples} samples x {n_snps} SNPs need {expected}"
            )
        return bed

    @property
    def bytes_per_snp(self) -> int:
        return (self.n_samples + 3) // 4

    def _packed(self, load_bytes: int) -> np.ndarray:
        """The packed genotypes, SNPs x bytes: read into memory if they fit
        ``load_bytes``, else memory-mapped.

        Every block of samples needs a band of bytes from every SNP's row, so a
        memory map re-reads the file (in scattered pages) once per block; on a
        shared filesystem that was 2 h for UK Biobank. Read once, sequentially,
        the same .bed is a quarter of the decoded matrix.
        """
        path = f"{self.prefix}.bed"
        if self.n_snps * self.bytes_per_snp <= load_bytes:
            raw = np.fromfile(path, dtype=np.uint8, offset=3)
        else:
            raw = np.memmap(path, dtype=np.uint8, mode="r", offset=3)
        return raw.reshape(self.n_snps, self.bytes_per_snp)

    def sample_blocks(
        self, rows=None, block_bytes: int = BLOCK_BYTES, load_bytes: int = LOAD_BYTES
    ) -> Iterator[tuple[int, np.ndarray]]:
        """(first sample, samples x len(rows) int8 dosages) for consecutive blocks of samples.

        ``rows`` are .bim rows in the order wanted (default: all, in order). The
        output is sample-major, like upstream's ``inputs``, so each block is
        written as one contiguous run of rows. A block never holds more than
        about ``block_bytes`` dosages.
        """
        packed = self._packed(load_bytes)
        rows = np.arange(self.n_snps) if rows is None else np.asarray(rows, dtype=np.int64)
        order = np.argsort(rows, kind="stable")  # read in file order, place in requested order
        ordered = rows[order]
        # Four samples per byte: blocks start on a byte boundary.
        per_block = max(4, block_bytes // max(len(rows), 1) // 4 * 4)

        for s0 in range(0, self.n_samples, per_block):
            s1 = min(s0 + per_block, self.n_samples)
            b0, b1 = s0 // 4, (s1 + 3) // 4
            out = np.empty((s1 - s0, len(rows)), dtype=np.int8)
            for c0 in range(0, len(rows), SNP_CHUNK):
                chunk = np.asarray(packed[ordered[c0 : c0 + SNP_CHUNK], b0:b1])
                dosages = BYTE_TO_DOSAGES[chunk].reshape(len(chunk), -1)[:, : s1 - s0]
                out[:, order[c0 : c0 + len(chunk)]] = dosages.T
            yield s0, out
