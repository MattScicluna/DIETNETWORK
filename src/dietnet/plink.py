"""Read PLINK 1 binary filesets (.bed/.bim/.fam) without third-party readers.

Dosages count the .bim A1 allele (0, 1, 2) with -1 for a missing call: what
``plink --recode A`` writes, and so what upstream's HDF5 datasets hold
(tests/parity checks it). The .bed is decoded in blocks of SNPs with a lookup
table, so a biobank-sized file never needs more than one block of scratch
memory beyond the output.
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
BLOCK = 4096


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

    def _packed(self) -> np.ndarray:
        raw = np.memmap(f"{self.prefix}.bed", dtype=np.uint8, mode="r", offset=3)
        return raw.reshape(self.n_snps, self.bytes_per_snp)

    def _decode(self, packed_rows: np.ndarray) -> np.ndarray:
        """Packed SNP rows -> samples x SNPs int8 dosages."""
        codes = np.stack([(packed_rows >> shift) & 3 for shift in (0, 2, 4, 6)], axis=-1)
        dosages = CODES[codes].reshape(len(packed_rows), -1)[:, : self.n_samples]
        return np.ascontiguousarray(dosages.T)

    def snp_blocks(self, block: int = BLOCK) -> Iterator[tuple[int, np.ndarray]]:
        """(first SNP, samples x block int8 dosages) for consecutive SNP blocks."""
        packed = self._packed()
        for start in range(0, self.n_snps, block):
            yield start, self._decode(np.asarray(packed[start : start + block]))

    def read_snps(self, indices) -> np.ndarray:
        """samples x len(indices) dosages for SNPs at arbitrary .bim rows, in that order."""
        indices = np.asarray(indices, dtype=np.int64)
        return self._decode(np.asarray(self._packed()[indices]))

    def read(self) -> np.ndarray:
        """The whole samples x SNPs int8 matrix."""
        out = np.empty((self.n_samples, self.n_snps), dtype=np.int8)
        for start, block in self.snp_blocks():
            out[:, start : start + block.shape[1]] = block
        return out
