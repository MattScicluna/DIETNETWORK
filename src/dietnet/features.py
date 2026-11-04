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
``genotype_freq``
    Genotype frequencies per fold over all its training samples, classes
    pooled: the same definition without the split by class, for regression
    tasks (which have no classes) or as a baseline. Read a block of samples at a
    time from a genotype store (``inputs``: as for ``train``), so it scales to
    a biobank. With ``classes`` (a sample id / class TSV), per class instead:
    ``class_genotype_freq``'s values, streamed, for classes the dataset does
    not hold (deciles of a trait, say).
``precomputed``
    A table with ``chrom``, ``pos``, ``ref``, ``alt`` and numeric feature
    columns (TSV, optionally gzipped, or parquet), matched to the dataset's SNPs
    by position and alleles, on either strand. A ``path`` containing ``{fold}``
    names one table per fold (``gwas_fold{fold}.tsv``: statistics from each
    fold's training samples), read and matched fold by fold into a per-fold
    block. With ``orient: effect`` a value
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

A block with ``include: false`` is built, so a later ``shuffled`` block can
copy it, but left out of the matrix: a shuffled control without the original.

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

import h5py
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


def _genotype_freq(ctx: Context, spec: dict) -> Block:
    """Genotype frequencies over each fold's training samples, classes pooled or
    per class.

    Upstream's definition (generate_embedding.py): ``count(genotype == g) / n``
    for g in 0, 1, 2, missing genotypes counted in ``n``. Pooled by default, for
    tasks without classes such as regression; with ``classes:`` (a TSV of
    sample id and class, as for ``--labels``) per class, in upstream's column
    order, so that classes can be defined apart from the dataset (deciles of a
    trait, say). Rows are grouped by class and by the folds they train in, and
    each group is counted once, a block at a time from a genotype store, so it
    scales to a biobank.
    """
    from dietnet import genotypes

    folds = np.load(ctx.partition, allow_pickle=True)["folds_indexes"]
    store = genotypes.open_store(
        ctx.dataset, genotypes.choose(ctx.dataset, spec.get("inputs", "auto"))
    )
    n_rows, n_snps = store.shape
    if "classes" in spec:
        with h5py.File(ctx.dataset, "r") as h5:
            samples = np.array(h5["samples"]).astype(str)
        labels = ds._ordered(ds.read_labels(spec["classes"]), samples, spec["classes"])
        names, klass = np.unique(labels, return_inverse=True)
    else:
        names, klass = np.array(["all"]), None

    counts, n = genotypes.count_by_fold(
        store, [fold[0] for fold in folds], klass, len(names), int(spec.get("block", 2048))
    )
    values = np.zeros((len(folds), n_snps, 3 * len(names)))
    for f in range(len(folds)):
        for c in range(len(names)):
            if n[f, c] == 0:
                raise ValueError(
                    f"genotype_freq: class {names[c]!r} has no training samples in fold {f}"
                )
            values[f, :, 3 * c : 3 * c + 3] = counts[f, c] / n[f, c]
    columns = [f"{c}:genotype{value}" for c in names for value in range(3)]
    info = {"fold_dependent": True, "training_samples": n.sum(axis=1).tolist()}
    if "classes" in spec:
        info["classes"] = {"path": str(spec["classes"]), "names": names.tolist()}
    return Block(values, columns, info)


def _read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def _match_table(ctx: Context, spec: dict, path: Path):
    """One table matched to the dataset's SNPs: values (SNPs x features, signed as
    ``orient`` says), which SNPs it covers, the feature names and a description."""
    table = _read_table(path)
    keys = ["chrom", "pos", "ref", "alt"]
    missing = [k for k in keys if k not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
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

    values = np.zeros((len(snps), len(features)))
    values[pairs["row"].to_numpy()] = pairs[features].to_numpy(dtype=float) * sign[:, None]
    covered = np.zeros(len(snps), dtype=bool)
    covered[pairs["row"].to_numpy()] = True
    if np.isnan(values).any():
        raise ValueError(f"{path}: NaN feature values; fill or drop them first")
    info = {
        "path": str(path),
        "coverage": float(covered.mean()),
        "negated": int((sign < 0).sum()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    return values, covered, list(features), info


def _precomputed(ctx: Context, spec: dict) -> Block:
    """A fold-independent table or, with ``{fold}`` in ``path``, one table per fold
    (statistics from each fold's training samples, say) stacked into a
    folds x SNPs x features block."""
    template = str(spec["path"])
    per_fold = "{fold}" in template
    if per_fold:
        paths = [Path(template.replace("{fold}", str(f))) for f in range(ctx.n_folds)]
    else:
        paths = [Path(template)]
    absent = [str(p) for p in paths if not p.exists()]
    if absent:
        raise FileNotFoundError(f"precomputed: no table at {', '.join(absent)}")
    matched = [_match_table(ctx, spec, p) for p in paths]
    features = matched[0][2]
    for p, (_, _, names, _) in zip(paths, matched):
        if names != features:
            raise ValueError(f"{p}: feature columns {names} differ from {paths[0]}'s {features}")

    name = spec.get("name", Path(template.replace("{fold}", "")).stem)
    values = np.stack([m[0] for m in matched])
    covered = np.stack([m[1] for m in matched])
    columns = [f"{name}:{c}" for c in features]
    if not covered.all():
        complete = [str(p) for p, c in zip(paths, covered) if c.all()]
        if complete:
            # The missing column would be all zeros there: NaN in the model.
            raise ValueError(
                f"precomputed: {', '.join(complete)} cover every SNP but other folds' "
                "tables do not; give every fold's table the same SNPs"
            )
        values = np.concatenate([values, (~covered).astype(float)[..., None]], axis=-1)
        columns.append(f"{name}:missing")
    orient = {"orient": spec.get("orient", "none")}
    if per_fold:
        info = {**orient, "fold_dependent": True, "folds": [m[3] for m in matched]}
        return Block(values, columns, info)
    return Block(values[0], columns, {**matched[0][3], **orient})


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
    "genotype_freq": _genotype_freq,
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
    # Resolved: upstream's scripts run from their own directory, where a path
    # relative to the caller's would not be found.
    dataset, partition, out = Path(dataset).resolve(), Path(partition).resolve(), Path(out)
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

    kept = [b for b, s in zip(ctx.built, spec["blocks"]) if s.get("include", True)]
    if not kept:
        raise ValueError("every block has include: false; nothing to write")
    if any(b.fold_dependent for b in kept):
        values = np.concatenate(
            [b.values if b.fold_dependent else np.broadcast_to(b.values, (n_folds, *b.values.shape))
             for b in kept],
            axis=-1,
        )  # fmt: skip
    else:
        values = np.concatenate([b.values for b in kept], axis=-1)

    columns = [c for b in kept for c in b.columns]
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
        if not entry["spec"].get("include", True):
            entry["columns"] = None  # built for a later block, not in the matrix
            continue
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
