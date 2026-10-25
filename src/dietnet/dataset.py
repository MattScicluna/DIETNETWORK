"""Build the HDF5 dataset upstream's scripts read, straight from PLINK.

Upstream makes it from a genotype text table (``plink --recode A``, then
``create_dataset.py``). This writes the same file from the .bed directly:
same datasets, same dtypes, same values (tests/parity), without the text
round-trip, which does not scale to a biobank.

Upstream schema, read by partition_data.py, generate_embedding.py,
compute_input_features_statistics.py and train.py:

``inputs``             int8, samples x SNPs, contiguous; dosage of the .bim A1, -1 missing
``samples``            bytes, sample ids (.fam IID), in .fam order
``snp_names``          bytes, ``<.bim id>_<A1>``: the ``plink --recode A`` column
                       names, so the counted allele travels with the name (as in
                       the published 1KGP dataset: ``chr1:858952:G:A_A``)
``class_labels``       int64 index into ``class_label_names`` (classification)
``class_label_names``  bytes, sorted unique labels
``regression_labels``  float64 (regression)

Added here, ignored by upstream: ``snps/{chrom,pos,a1,a2}`` (what matching a new
cohort by position and allele needs) and provenance attributes.

The layout is contiguous and uncompressed on purpose: chunked or compressed
layouts make the random minibatch reads training does 100-300x slower
(docs/refactor-plan.md, section 2).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from dietnet import __version__
from dietnet.plink import Bed, read_bim, read_fam


def read_labels(path) -> pd.Series:
    """Sample id -> label from a TSV with a header row (upstream's format)."""
    table = pd.read_csv(path, sep="\t", dtype=str)
    if table.shape[1] < 2:
        raise ValueError(f"{path}: expected two tab-separated columns, sample id and label")
    labels = table.set_index(table.columns[0])[table.columns[1]]
    duplicated = labels.index[labels.index.duplicated()]
    if len(duplicated):
        raise ValueError(f"{path}: duplicated sample ids, e.g. {list(duplicated[:3])}")
    return labels


def _ordered(labels: pd.Series, samples: np.ndarray, path) -> np.ndarray:
    missing = pd.Index(samples).difference(labels.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} samples in the PLINK files have no label in {path}, "
            f"e.g. {list(missing[:3])}"
        )
    return labels.loc[samples].to_numpy()


def _fingerprint(prefix: str) -> str:
    """Cheap identity of the source .bed: size plus a hash of its first and last MiB."""
    path = Path(f"{prefix}.bed")
    digest = hashlib.sha256(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        digest.update(f.read(2**20))
        f.seek(max(0, path.stat().st_size - 2**20))
        digest.update(f.read())
    return digest.hexdigest()


def build(
    plink_prefix,
    out,
    class_labels=None,
    regression_labels=None,
    snp_block: int = 4096,
) -> Path:
    """Write ``out`` (upstream's HDF5 schema) from a PLINK fileset."""
    bed = Bed.open(plink_prefix)
    fam = read_fam(bed.prefix)
    bim = read_bim(bed.prefix)
    samples = fam["iid"].to_numpy()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    with h5py.File(tmp, "w") as f:
        inputs = f.create_dataset("inputs", shape=(bed.n_samples, bed.n_snps), dtype=np.int8)
        for start, block in bed.snp_blocks(snp_block):
            inputs[:, start : start + block.shape[1]] = block

        names = (bim["snp"] + "_" + bim["a1"]).to_numpy()
        f.create_dataset("snp_names", data=names.astype("S"))
        f.create_dataset("samples", data=samples.astype("S"))

        if class_labels is not None:
            ordered = _ordered(read_labels(class_labels), samples, class_labels)
            names = np.sort(np.unique(ordered))
            f.create_dataset("class_labels", data=np.searchsorted(names, ordered).astype(np.int64))
            f.create_dataset("class_label_names", data=names.astype("S"))

        if regression_labels is not None:
            ordered = _ordered(read_labels(regression_labels), samples, regression_labels)
            f.create_dataset("regression_labels", data=ordered.astype("float64"))

        snps = f.create_group("snps")
        snps.create_dataset("chrom", data=bim["chrom"].to_numpy().astype("S"))
        snps.create_dataset("pos", data=bim["pos"].to_numpy().astype(np.int64))
        snps.create_dataset("a1", data=bim["a1"].to_numpy().astype("S"))
        snps.create_dataset("a2", data=bim["a2"].to_numpy().astype("S"))

        f.attrs["source_plink_prefix"] = str(Path(bed.prefix).resolve())
        f.attrs["source_bed_fingerprint"] = _fingerprint(bed.prefix)
        f.attrs["built_by"] = f"dietnet {__version__}"
        f.attrs["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp.replace(out)
    return out


def snp_table(path) -> pd.DataFrame:
    """One row per SNP: ``name`` as stored, ``snp`` id, ``a1`` counted allele, and
    ``chrom``, ``pos``, ``a2`` where they can be known.

    Files built here carry them in ``snps/``. Files made by upstream's
    create_dataset.py, including the published 1KGP dataset, only have
    ``snp_names``; those are parsed: ``<id>_<A1>`` gives the counted allele, and
    an id of the form ``chr:pos:ref:alt`` gives position and other allele.
    """
    with h5py.File(path, "r") as f:
        names = np.array(f["snp_names"]).astype(str)
        stored = None
        if "snps" in f:
            g = f["snps"]
            stored = {k: np.array(g[k]) for k in ("chrom", "pos", "a1", "a2")}

    table = pd.DataFrame({"name": names})
    split = table["name"].str.rsplit("_", n=1, expand=True)
    has_allele = split.shape[1] == 2 and split[1].fillna("").str.fullmatch(r"[ACGT]+").all()
    table["snp"] = split[0] if has_allele else table["name"]
    if stored is not None:
        table["chrom"] = stored["chrom"].astype(str)
        table["pos"] = stored["pos"].astype(np.int64)
        table["a1"] = stored["a1"].astype(str)
        table["a2"] = stored["a2"].astype(str)
        return table

    table["a1"] = split[1] if has_allele else None
    parts = table["snp"].str.split(":", expand=True)
    if parts.shape[1] == 4 and parts[1].str.fullmatch(r"\d+").all():
        table["chrom"] = parts[0].str.removeprefix("chr")
        table["pos"] = parts[1].astype(np.int64)
        if has_allele:
            ref, alt = parts[2], parts[3]
            table["a2"] = np.where(table["a1"] == alt, ref, np.where(table["a1"] == ref, alt, None))
    return table


def class_label_names(path) -> list[str]:
    with h5py.File(path, "r") as f:
        return list(np.array(f["class_label_names"]).astype(str))


def memmap_inputs(path) -> np.ndarray:
    """``inputs`` as a read-only memory map: no copy into RAM, pages read on demand.

    Possible because the dataset is contiguous and uncompressed; HDF5 then
    stores it as one run of bytes at a fixed offset.
    """
    with h5py.File(path, "r") as f:
        ds = f["inputs"]
        offset = ds.id.get_offset()
        if offset is None or ds.chunks is not None or ds.compression is not None:
            raise ValueError(f"{path}: 'inputs' is not contiguous and uncompressed")
        shape, dtype = ds.shape, ds.dtype
    # Copy-on-write: rows read as writable arrays (torch warns on read-only ones)
    # while the file itself is never modified.
    return np.memmap(path, dtype=dtype, mode="c", offset=offset, shape=shape)
