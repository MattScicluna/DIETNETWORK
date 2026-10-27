"""AlphaGenome variant-effect scores for a dataset's SNPs, as an auxiliary-network feature table.

AlphaGenome (Avsec et al.) predicts many genomic tracks from DNA sequence, and
scores a variant by how its ALT allele changes them relative to REF. Through
the hosted API (``pip install 'dietnetwork[alphagenome]'``, key in
ALPHAGENOME_API_KEY), :func:`score` turns every SNP of a dataset into a row
of such effects:

    chrom  pos  ref  alt  <output_type>|<track_name>  ...

one signed ALT-minus-REF ``raw_score`` per track. Gene-level scorers report
one score per nearby gene; the gene with the largest effect on the track is
kept. The table is what ``build-embedding``'s ``precomputed`` provider reads,
with ``orient: effect``: it flips the sign wherever the dataset's dosage
counts REF rather than ALT, and ``pca: k`` reduces the thousands of tracks to
a block the size of the rest of the embedding.

Scoring is resumable: rows are appended a chunk at a time, and a rerun skips
SNPs already in the file. Hundreds of thousands of API calls will be
interrupted at some point.

REF and ALT come from SNP ids of the form ``chr:pos:REF:ALT`` (the 1KGP
data's), since a .bim's A1/A2 say which allele is counted, not which is the
reference. Coordinates must be GRCh38, which is what AlphaGenome uses.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import pandas as pd

from dietnet import dataset as ds

API_KEY_ENV = "ALPHAGENOME_API_KEY"
KEYS = ["chrom", "pos", "ref", "alt"]
DEFAULT_SCORERS = ("ATAC", "DNASE", "CHIP_TF", "CHIP_HISTONE", "CAGE", "RNA_SEQ")
SEQUENCE_LENGTHS = ("16KB", "100KB", "500KB", "1MB")
# gRPC statuses the service returns for a call that succeeds when repeated: about 2% of
# calls fail with INTERNAL, at random (2026-10-26).
TRANSIENT = ("INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED")
RETRIES = 6
RETRY_WAIT = 2.0  # seconds, doubled after each failed attempt


def _retrying(call):
    """``call()``, repeated after a transient gRPC error; other errors raise at once."""
    for attempt in range(RETRIES + 1):
        try:
            return call()
        except Exception as error:
            code = getattr(error, "code", None)
            status = getattr(code(), "name", None) if callable(code) else None
            if status not in TRANSIENT or attempt == RETRIES:
                raise
            time.sleep(RETRY_WAIT * 2**attempt)


def variants_of(source) -> pd.DataFrame:
    """chrom (``chr``-prefixed), 1-based pos, ref, alt for every SNP in a dataset, or
    in a PLINK ``.bim``: only its SNP ids are read, so no genotypes are needed."""
    if Path(source).suffix == ".bim":
        ids = pd.read_csv(source, sep=r"\s+", header=None, usecols=[1], dtype=str)[1]
    else:
        ids = ds.snp_table(source)["snp"]
    parts = ids.str.split(":", expand=True)
    if parts.shape[1] != 4 or not parts[1].str.fullmatch(r"\d+").all():
        raise ValueError(
            "SNP ids are not chr:pos:REF:ALT, so REF and ALT are unknown; "
            "AlphaGenome needs them (a .bim's A1/A2 say which allele is counted, not which is REF)"
        )
    chrom = "chr" + parts[0].str.removeprefix("chr")
    return pd.DataFrame(
        {"chrom": chrom, "pos": parts[1].astype(int), "ref": parts[2], "alt": parts[3]}
    )


class Backend(Protocol):
    def score(self, variants: pd.DataFrame) -> pd.DataFrame:
        """Tidy scores for ``variants``: one row per (variant, track[, gene]) with
        ``chrom``, ``pos``, ``ref``, ``alt``, ``output_type``, ``track_name``,
        ``raw_score``."""


class AlphaGenomeAPI:
    """The hosted AlphaGenome model, through its Python client."""

    def __init__(
        self,
        scorers=DEFAULT_SCORERS,
        sequence_length: str = "1MB",
        max_workers: int = 5,
        api_key: str | None = None,
    ):
        try:
            from alphagenome.data import genome
            from alphagenome.models import dna_client, variant_scorers
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("pip install 'dietnetwork[alphagenome]'") from error

        api_key = api_key or os.environ.get(API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"set {API_KEY_ENV} to an AlphaGenome API key")
        if sequence_length not in SEQUENCE_LENGTHS:
            raise ValueError(f"sequence_length must be one of {SEQUENCE_LENGTHS}")

        self._genome, self._scorers_lib = genome, variant_scorers
        self._client = dna_client.create(api_key)
        self._length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[f"SEQUENCE_LENGTH_{sequence_length}"]
        recommended = variant_scorers.RECOMMENDED_VARIANT_SCORERS
        unknown = set(scorers) - set(recommended)
        if unknown:
            raise ValueError(f"unknown scorers {sorted(unknown)}; one of {sorted(recommended)}")
        self._scorers = [recommended[name] for name in scorers]
        self._max_workers = max_workers

    def score(self, variants: pd.DataFrame) -> pd.DataFrame:
        objects = [
            self._genome.Variant(
                chromosome=row.chrom, position=int(row.pos),
                reference_bases=row.ref, alternate_bases=row.alt,
            )
            for row in variants.itertuples()
        ]  # fmt: skip

        # One request per variant, not the client's score_variants: that fails the whole
        # batch when any one request does, and with ~2% of requests failing transiently a
        # chunk of 200 almost never succeeds.
        def one(variant):
            return _retrying(
                lambda: self._client.score_variant(
                    interval=variant.reference_interval.resize(self._length),
                    variant=variant,
                    variant_scorers=self._scorers,
                )
            )

        with ThreadPoolExecutor(self._max_workers) as pool:
            results = list(pool.map(one, objects))
        tidy = self._scorers_lib.tidy_scores(results)
        keys = {str(v): row for v, row in zip(objects, variants.itertuples())}
        rows = tidy["variant_id"].astype(str).map(keys)
        for key in KEYS:
            tidy[key] = [getattr(r, key) for r in rows]
        return tidy


def features_of(tidy: pd.DataFrame) -> pd.DataFrame:
    """One row per variant, one ``<output_type>|<track_name>`` column of raw scores.

    Where a track has a score per gene, the one with the largest magnitude is
    kept (signed).
    """
    tidy = tidy.assign(track=tidy["output_type"].astype(str) + "|" + tidy["track_name"].astype(str))
    tidy = tidy.assign(magnitude=tidy["raw_score"].abs())
    strongest = tidy.sort_values("magnitude").drop_duplicates(KEYS + ["track"], keep="last")
    wide = strongest.pivot_table(index=KEYS, columns="track", values="raw_score", aggfunc="first")
    wide.columns.name = None
    return wide.reset_index()


def _parts(out: Path) -> Path:
    return out.with_name(out.name + ".parts")


def score(dataset, out, backend: Backend, chunk: int = 200, progress=None) -> Path:
    """Score every SNP of ``dataset`` (or a ``.bim``), then write the feature table ``out``.

    Each chunk's scores go to their own file under ``<out>.parts/`` as soon as
    they arrive; a rerun scores only the SNPs no part holds yet. When all are
    scored the parts are combined: the union of their tracks, with 0 (no
    effect) where a variant has no score for a track -- a gene-level scorer
    says nothing about a variant with no gene in range.
    """
    out = Path(out)
    parts = _parts(out)
    parts.mkdir(parents=True, exist_ok=True)
    variants = variants_of(dataset)

    done = [pd.read_csv(p, sep="\t", usecols=KEYS) for p in sorted(parts.glob("part-*.tsv"))]
    todo = variants
    if done:
        seen = pd.concat(done).astype({"pos": int})
        todo = variants.merge(seen, on=KEYS, how="left", indicator=True)
        todo = todo[todo["_merge"] == "left_only"][KEYS]
    todo = todo.reset_index(drop=True)

    first = len(done)
    for i, start in enumerate(range(0, len(todo), chunk)):
        part = features_of(backend.score(todo.iloc[start : start + chunk]))
        tmp = parts / f"part-{first + i:06d}.tsv.partial"
        part.to_csv(tmp, sep="\t", index=False)
        tmp.replace(tmp.with_suffix(""))
        if progress:
            progress(min(start + chunk, len(todo)), len(todo))

    _combine(sorted(parts.glob("part-*.tsv")), variants, out)
    return out


def _combine(paths: list[Path], variants: pd.DataFrame, out: Path) -> None:
    """Write the parts as one table, holding one part in memory at a time.

    A biobank array is ~500,000 SNPs x ~4,000 tracks: 15 GB as float64, and
    reading every part at once (then copying it to fill, merge and sort) needs
    several times that. The tracks are the union of the parts' headers; rows
    follow the dataset's SNP order, parts sorted by their first SNP. Rows of
    SNPs the dataset no longer has are left out.
    """
    tracks = sorted(
        {c for p in paths for c in pd.read_csv(p, sep="\t", nrows=0).columns} - set(KEYS)
    )
    rank = {key: i for i, key in enumerate(variants[KEYS].itertuples(index=False, name=None))}

    def ranks(frame):
        keys = frame[KEYS].astype({"chrom": str, "pos": int}).itertuples(index=False, name=None)
        return pd.Series([rank.get(k, -1) for k in keys], index=frame.index)

    firsts = {p: ranks(pd.read_csv(p, sep="\t", usecols=KEYS, nrows=1)).iloc[0] for p in paths}
    tmp = out.with_name(out.name + ".partial")
    with open(tmp, "w") as handle:
        handle.write("\t".join(KEYS + tracks) + "\n")
        for path in sorted(paths, key=firsts.get):
            part = pd.read_csv(path, sep="\t")
            part = part.assign(_row=ranks(part))
            part = part[part["_row"] >= 0].sort_values("_row")
            part = part.reindex(columns=KEYS + tracks).fillna({t: 0.0 for t in tracks})
            part.to_csv(handle, sep="\t", index=False, header=False)
    tmp.replace(out)
