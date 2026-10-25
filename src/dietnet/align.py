"""Match a new cohort's SNPs to a trained model's, by position and allele.

A dosage counts one allele, and which one PLINK 1.9 picks (A1, by default the
minor allele) depends on the cohort's own frequencies. In the 1KGP training
data a quarter of SNPs count the reference allele and the rest the alternate.
So the same SNP can be coded x in one cohort and 2 - x in another; matching by
id alone would feed the model mirror-coded genotypes for those SNPs, silently.

Per model SNP, against the cohort's .bim:

* same A1/A2                      -> copy
* A1 and A2 swapped               -> flip (2 - x; missing stays missing)
* the same on the other strand    -> copy or flip, except for A/T and C/G SNPs,
  whose strand cannot be told from the alleles: those are kept only when A1
  matches outright
* otherwise, or absent            -> missing

Missing SNPs then follow the published method (Rochefort-Boulanger, Scicluna
et al. 2026; upstream's ``match_input_features``): they are set to -1, which
imputation and standardisation turn into 0, and the observed ones are scaled
by D / (D - |C|), stored as ``scale`` in the matched dataset.

The matched dataset has upstream's schema for it (``inputs``, ``samples``,
``snp_names``, ``scale``), so upstream's test_independent_dataset.py reads it
with ``--matched-test-dataset``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from dietnet.plink import Bed, read_bim, read_fam

COMPLEMENT = str.maketrans("ACGT", "TGCA")
COLUMN_BLOCK = 4096


def _complement(alleles: pd.Series) -> pd.Series:
    return alleles.str.translate(COMPLEMENT)


@dataclass
class Alignment:
    """Where each model SNP comes from in the cohort, and whether to flip it."""

    source: np.ndarray  # cohort .bim row per model SNP, -1 if missing
    flip: np.ndarray  # bool per model SNP
    counts: dict = field(default_factory=dict)

    @property
    def n_model(self) -> int:
        return len(self.source)

    @property
    def n_matched(self) -> int:
        return int((self.source >= 0).sum())

    @property
    def scale(self) -> float:
        """D / (D - |C|): the published rescaling of observed SNPs."""
        if self.n_matched == 0:
            raise ValueError("no SNP of the model was found in the cohort")
        return self.n_model / self.n_matched


def align(model_snps: pd.DataFrame, cohort_prefix) -> Alignment:
    """Align a cohort's .bim to ``model_snps`` (from :func:`dietnet.dataset.snp_table`)."""
    cohort = read_bim(cohort_prefix).reset_index(names="row")
    model = model_snps.reset_index(drop=True).reset_index(names="col")

    by_position = {"chrom", "pos"} <= set(model.columns) and model[
        ["chrom", "pos"]
    ].notna().all().all()
    if by_position:
        model["key"] = model["chrom"].astype(str) + ":" + model["pos"].astype(str)
        cohort["key"] = cohort["chrom"] + ":" + cohort["pos"].astype(str)
    else:
        model["key"] = model["snp"]
        cohort["key"] = cohort["snp"]

    if "a1" not in model.columns or model["a1"].isna().all():
        # Nothing records which allele the model counts (a dataset whose SNP
        # names lack plink's _A1 suffix): match on the key alone and say so.
        pairs = model[["col", "key"]].merge(cohort[["row", "key"]], on="key")
        pairs = pairs.sort_values(["col", "row"]).drop_duplicates("col")
        source = np.full(len(model), -1, dtype=np.int64)
        source[pairs["col"].to_numpy()] = pairs["row"].to_numpy()
        counts = dict(
            alleles_unchecked=len(pairs),
            matched_by="position" if by_position else "id",
            model_snps=len(model),
            missing=int((source < 0).sum()),
        )
        return Alignment(source=source, flip=np.zeros(len(model), dtype=bool), counts=counts)

    model["a2"] = model["a2"] if "a2" in model.columns else None
    pairs = model.merge(cohort, on="key", suffixes=("_m", "_c"))
    a1m, a2m, a1c, a2c = pairs["a1_m"], pairs["a2_m"], pairs["a1_c"], pairs["a2_c"]
    known_a2 = a2m.notna()
    ambiguous = known_a2 & (a1m == _complement(a2m))

    same = (a1c == a1m) & (~known_a2 | (a2c == a2m))
    swapped = (a2c == a1m) & (~known_a2 | (a1c == a2m))
    strand_same = ~ambiguous & (_complement(a1c) == a1m) & (~known_a2 | (_complement(a2c) == a2m))
    strand_swapped = (
        ~ambiguous & (_complement(a2c) == a1m) & (~known_a2 | (_complement(a1c) == a2m))
    )

    pairs["kind"] = np.select(
        [same, swapped, strand_same, strand_swapped],
        ["same", "swapped", "strand", "strand_swapped"],
        default="",
    )
    usable = pairs[pairs["kind"] != ""]
    # A position can hold several cohort variants (multi-allelic sites split
    # into biallelic rows); keep the first whose alleles fit.
    usable = usable.sort_values(["col", "row"]).drop_duplicates("col")

    source = np.full(len(model), -1, dtype=np.int64)
    flip = np.zeros(len(model), dtype=bool)
    source[usable["col"].to_numpy()] = usable["row"].to_numpy()
    flip[usable["col"].to_numpy()] = usable["kind"].isin(["swapped", "strand_swapped"]).to_numpy()

    counts = usable["kind"].value_counts().to_dict()
    counts.update(
        matched_by="position" if by_position else "id",
        model_snps=len(model),
        found_with_other_alleles=int(pairs.loc[pairs["kind"] == "", "col"].nunique()),
        ambiguous_kept=int((ambiguous & same)[pairs["kind"] != ""].sum()),
        missing=int((source < 0).sum()),
    )
    return Alignment(source=source, flip=flip, counts=counts)


def fingerprint(model_snps: pd.DataFrame, cohort_prefix) -> str:
    """Identifies an (model SNP set, cohort) pair, to reuse a matched dataset."""
    digest = hashlib.sha256()
    digest.update("\n".join(model_snps["name"]).encode())
    for ext in (".bim", ".fam"):
        digest.update(Path(f"{cohort_prefix}{ext}").read_bytes())
    bed = Path(f"{cohort_prefix}.bed")
    digest.update(str(bed.stat().st_size).encode())
    return digest.hexdigest()[:16]


def write_matched(cohort_prefix, model_snps: pd.DataFrame, out, alignment=None) -> Path:
    """Write the cohort on the model's SNPs, in upstream's matched-dataset schema."""
    alignment = alignment or align(model_snps, cohort_prefix)
    bed = Bed.open(cohort_prefix)
    samples = read_fam(cohort_prefix)["iid"].to_numpy()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    with h5py.File(tmp, "w") as f:
        inputs = f.create_dataset("inputs", shape=(bed.n_samples, alignment.n_model), dtype=np.int8)
        for start in range(0, alignment.n_model, COLUMN_BLOCK):
            cols = slice(start, min(start + COLUMN_BLOCK, alignment.n_model))
            source, flip = alignment.source[cols], alignment.flip[cols]
            block = np.full((bed.n_samples, len(source)), -1, dtype=np.int8)
            present = np.flatnonzero(source >= 0)
            if len(present):
                rows = source[present]
                # Read in .bim order (sequential on disk), then put back in model order.
                order = np.argsort(rows)
                dosages = np.empty((bed.n_samples, len(rows)), dtype=np.int8)
                dosages[:, order] = bed.read_snps(rows[order])
                flipped = flip[present]
                observed = dosages >= 0
                dosages[:, flipped] = np.where(
                    observed[:, flipped], 2 - dosages[:, flipped], dosages[:, flipped]
                )
                block[:, present] = dosages
            inputs[:, cols] = block

        f.create_dataset("samples", data=samples.astype("S"))
        f.create_dataset("snp_names", data=model_snps["name"].to_numpy().astype("S"))
        f.create_dataset("scale", data=np.array([alignment.scale]))
        f.attrs["alignment"] = json.dumps(alignment.counts)
        f.attrs["source_plink_prefix"] = str(Path(bed.prefix).resolve())
    tmp.replace(out)
    return out
