"""Per-SNP features for the auxiliary network, from several sources.

Upstream's DietNetwork reads its auxiliary-network input from ``embedding.npz``:
a SNPs x features matrix, one per fold (3-D) or shared by all folds (2-D),
whose columns it L2-normalises itself. Anything that can be written as such a
matrix can feed the model, without touching upstream's code. This module
builds one from *blocks*, described in a YAML spec::

    blocks:
      - provider: class_genotype_freq          # upstream's embedding (per fold)
      - provider: precomputed                  # any per-SNP table, e.g. AlphaGenome scores
        path: alphagenome_scores.tsv
        orient: effect                         # values are ALT - REF effects
        pca: 64                                # reduce to 64 components
      - provider: random                       # control: same width, no information
        columns: 64
        seed: 0

Providers:

``class_genotype_freq``
    Genotype frequencies per class and fold: upstream's generate_embedding.py,
    run unmodified. Alone, the result is exactly what ``dietnet
    generate-embedding`` writes.
``precomputed``
    A table with ``chrom``, ``pos``, ``ref``, ``alt`` and numeric feature
    columns (TSV, optionally gzipped, or parquet), matched to the dataset's SNPs
    by position and alleles, on either strand. With ``orient: effect`` a value
    describes ALT relative to REF, so it is negated where the dataset counts
    REF (plink 1.9 makes the minor allele A1, which is REF for a quarter of the
    1KGP SNPs). SNPs the table lacks get zeros and are marked in an added
    ``<name>:missing`` column.
``random``
    Standard-normal columns: the control for "more columns" rather than "more
    information".
``shuffled``
    Another block with its SNP rows permuted (``of``: that block's index): the
    control for "this information" rather than "information of this shape".

Transforms per block, applied after the provider: ``center: true`` subtracts
each column's mean over SNPs; ``pca: k`` keeps the first k principal
components over SNPs (computed per fold for a per-fold block). Upstream's
per-column L2 normalisation undoes any per-column scaling, so these are the
transforms that matter: centering, and controlling how many columns -- and so
how much weight -- a block brings.

Every column must vary: an all-zero column would be divided by a zero norm
inside the model and turn it to NaN, so building refuses one.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dietnet import dataset as ds
from dietnet import upstream

COMPLEMENT = str.maketrans("ACGT", "TGCA")


@dataclass
class Block:
    """A provider's output: SNPs x columns, or folds x SNPs x columns."""

    values: np.ndarray
    columns: list[str]
    info: dict = field(default_factory=dict)

    @property
    def fold_dependent(self) -> bool:
        return self.values.ndim == 3


@dataclass
class Context:
    dataset: Path
    partition: Path
    snps: pd.DataFrame
    n_folds: int
    built: list[Block]


def _class_genotype_freq(ctx: Context, spec: dict) -> Block:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "embedding.npz"
        upstream.run(
            "generate_embedding.py",
            ["--exp-path", tmp, "--dataset", ctx.dataset, "--partition", ctx.partition,
             "--out", out.stem, "--ncpus", spec.get("ncpus", 1)],
            log=Path(tmp) / "generate_embedding.log",
        )  # fmt: skip
        values = np.load(out)["emb"]
    classes = ds.class_label_names(ctx.dataset)[: values.shape[-1] // 3]
    columns = [f"{c}:genotype{g}" for c in classes for g in range(3)]
    return Block(values, columns, {"fold_dependent": True})


def _read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def _precomputed(ctx: Context, spec: dict) -> Block:
    table = _read_table(spec["path"])
    keys = ["chrom", "pos", "ref", "alt"]
    missing = [k for k in keys if k not in table.columns]
    if missing:
        raise ValueError(f"{spec['path']}: missing columns {missing}")
    features = spec.get("columns") or [c for c in table.columns if c not in keys]
    orient = spec.get("orient", "none")
    if orient not in ("effect", "none"):
        raise ValueError(f"orient must be 'effect' or 'none', not {orient!r}")

    snps = ctx.snps.reset_index(drop=True).reset_index(names="row")
    if not {"chrom", "pos", "a1", "a2"} <= set(snps.columns) or snps["a2"].isna().any():
        raise ValueError("the dataset's SNPs lack positions or alleles to match a table against")
    table = table.assign(chrom=table["chrom"].astype(str).str.removeprefix("chr"))
    snps = snps.assign(chrom=snps["chrom"].astype(str).str.removeprefix("chr"))
    pairs = snps.merge(table, on=["chrom", "pos"])

    comp = {k: pairs[k].str.translate(COMPLEMENT) for k in ("ref", "alt")}
    ambiguous = pairs["ref"] == comp["alt"]
    counts_alt = (pairs["a1"] == pairs["alt"]) & (pairs["a2"] == pairs["ref"])
    counts_ref = (pairs["a1"] == pairs["ref"]) & (pairs["a2"] == pairs["alt"])
    counts_alt |= ~ambiguous & (pairs["a1"] == comp["alt"]) & (pairs["a2"] == comp["ref"])
    counts_ref |= ~ambiguous & (pairs["a1"] == comp["ref"]) & (pairs["a2"] == comp["alt"])
    usable = counts_alt | counts_ref
    if orient == "effect":
        # An A/T or C/G SNP fits both orientations (a swap looks like the other
        # strand), so the sign of an effect cannot be known: treated as missing.
        usable &= ~ambiguous
    pairs = pairs[usable].drop_duplicates("row")
    sign = np.where(counts_ref[pairs.index] & (orient == "effect"), -1.0, 1.0)

    name = spec.get("name", Path(spec["path"]).stem)
    values = np.zeros((len(snps), len(features)))
    values[pairs["row"].to_numpy()] = pairs[features].to_numpy(dtype=float) * sign[:, None]
    covered = np.zeros(len(snps), dtype=bool)
    covered[pairs["row"].to_numpy()] = True
    if np.isnan(values).any():
        raise ValueError(f"{spec['path']}: NaN feature values; fill or drop them first")

    columns = [f"{name}:{c}" for c in features]
    if not covered.all():
        values = np.column_stack([values, (~covered).astype(float)])
        columns.append(f"{name}:missing")
    info = {
        "path": str(spec["path"]),
        "orient": orient,
        "coverage": float(covered.mean()),
        "negated": int((sign < 0).sum()),
        "sha256": hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest(),
    }
    return Block(values, columns, info)


def _random(ctx: Context, spec: dict) -> Block:
    k, seed = int(spec["columns"]), int(spec.get("seed", 0))
    values = np.random.default_rng(seed).standard_normal((len(ctx.snps), k))
    return Block(values, [f"random{seed}:{i}" for i in range(k)], {"seed": seed})


def _shuffled(ctx: Context, spec: dict) -> Block:
    of, seed = int(spec["of"]), int(spec.get("seed", 0))
    if not 0 <= of < len(ctx.built):
        raise ValueError(f"shuffled: 'of' must name an earlier block (0..{len(ctx.built) - 1})")
    source = ctx.built[of]
    perm = np.random.default_rng(seed).permutation(len(ctx.snps))
    values = source.values[..., perm, :]
    return Block(values, [f"shuffled{seed}:{c}" for c in source.columns], {"of": of, "seed": seed})


PROVIDERS = {
    "class_genotype_freq": _class_genotype_freq,
    "precomputed": _precomputed,
    "random": _random,
    "shuffled": _shuffled,
}


def _pca(values: np.ndarray, k: int) -> np.ndarray:
    centred = values - values.mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    return u[:, :k] * s[:k]


def _transform(block: Block, spec: dict) -> Block:
    values = block.values
    if spec.get("center"):
        values = values - values.mean(axis=-2, keepdims=True)
    if "pca" in spec:
        k = int(spec["pca"])
        if k > values.shape[-1]:
            raise ValueError(f"pca: {k} components from {values.shape[-1]} columns")
        if values.ndim == 3:
            values = np.stack([_pca(v, k) for v in values])
        else:
            values = _pca(values, k)
        stem = block.columns[0].split(":")[0]
        return Block(values, [f"{stem}:pc{i}" for i in range(k)], {**block.info, "pca": k})
    return Block(values, block.columns, block.info)


def build(spec, dataset, partition, out) -> tuple[np.ndarray, dict]:
    """Write ``out`` (upstream's embedding.npz) and ``<out>.json`` from a spec."""
    spec = yaml.safe_load(Path(spec).read_text()) if not isinstance(spec, dict) else spec
    dataset, partition, out = Path(dataset), Path(partition), Path(out)
    n_folds = len(np.load(partition, allow_pickle=True)["folds_indexes"])
    ctx = Context(dataset, partition, ds.snp_table(dataset), n_folds, [])

    described = []
    for i, block_spec in enumerate(spec["blocks"]):
        provider = block_spec.get("provider")
        if provider not in PROVIDERS:
            raise ValueError(
                f"block {i}: unknown provider {provider!r}; one of {sorted(PROVIDERS)}"
            )
        block = _transform(PROVIDERS[provider](ctx, block_spec), block_spec)
        ctx.built.append(block)
        described.append({"provider": provider, "spec": block_spec, **block.info})

    if any(b.fold_dependent for b in ctx.built):
        values = np.concatenate(
            [b.values if b.fold_dependent else np.broadcast_to(b.values, (n_folds, *b.values.shape))
             for b in ctx.built],
            axis=-1,
        )  # fmt: skip
    else:
        values = np.concatenate([b.values for b in ctx.built], axis=-1)

    columns = [c for b in ctx.built for c in b.columns]
    flat = values.reshape(-1, values.shape[-2], values.shape[-1])
    constant = sorted(
        {columns[j] for v in flat for j in np.flatnonzero(np.abs(v).sum(axis=0) == 0)}
    )
    if constant:
        raise ValueError(
            f"all-zero feature columns would be NaN in the model: {constant[:5]}"
            + (" ..." if len(constant) > 5 else "")
        )

    start = 0
    for entry, block in zip(described, ctx.built):
        entry["columns"] = [start, start + len(block.columns)]
        start += len(block.columns)
    meta = {
        "blocks": described,
        "column_names": columns,
        "fold_dependent": values.ndim == 3,
        "shape": list(values.shape),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=values)
    Path(f"{out}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return values, meta
