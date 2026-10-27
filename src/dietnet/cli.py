"""``dietnet``: the published Diet Network, from PLINK files to predictions.

Pipeline (each step writes one file the next reads)::

    dietnet create-dataset      PLINK + labels   -> dataset.h5       (this package)
    dietnet partition           dataset.h5       -> partition.npz    (upstream partition_data.py)
    dietnet generate-embedding  + partition      -> embedding.npz    (upstream generate_embedding.py)
      or build-embedding --spec                  (several feature sources, see dietnet.features;
                                                  dietnet alphagenome makes one of them)
    dietnet compute-input-stats + partition      -> input_stats.npz  (upstream compute_input_features_statistics.py)
    dietnet train               all of the above -> model packages   (upstream train.py)
    dietnet match               packages + PLINK -> matched cohort   (optional, CPU: ahead of predict)
    dietnet predict             packages + PLINK -> predictions.tsv  (upstream model and inference step)
    dietnet check               predictions + labels -> accuracy

``dietnet published`` fetches the published models as packages instead of
training; ``dietnet info`` shows which upstream commit is running.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from dietnet import __version__


def _ints(ctx, param, value):
    """Accept --seeds 78 79, --seeds 78,79 or repeated flags."""
    out = []
    for item in value or ():
        out += [int(v) for v in str(item).replace(",", " ").split()]
    return tuple(out)


def _npz(path: Path) -> Path:
    """numpy appends .npz to names without it; upstream passes names through."""
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


@click.group()
@click.version_option(__version__)
def main():
    """Diet Network ancestry prediction (Rochefort-Boulanger, Scicluna et al., AJHG 2026)."""


@main.command("create-dataset")
@click.option("--plink", "plink_prefix", required=True, help="PLINK prefix (.bed/.bim/.fam).")
@click.option("--labels", type=click.Path(exists=True), help="TSV: sample id, class label.")
@click.option("--regression-labels", type=click.Path(exists=True), help="TSV: sample id, value.")
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Output .h5.")
def create_dataset(plink_prefix, labels, regression_labels, out):
    """Build upstream's HDF5 dataset from PLINK, without the text round-trip."""
    from dietnet import dataset

    if not labels and not regression_labels:
        raise click.UsageError("give --labels, --regression-labels or both")
    dataset.build(plink_prefix, out, class_labels=labels, regression_labels=regression_labels)
    click.echo(f"wrote {out}")


@main.command()
@click.option(
    "--dataset", "dataset_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Output .npz.")
@click.option("--nb-folds", default=5, show_default=True)
@click.option("--seed", default=23, show_default=True)
@click.option("--train-valid-ratio", default=0.75, show_default=True, help="Unbalanced split only.")
@click.option(
    "--balanced/--unbalanced",
    default=True,
    show_default=True,
    help="Balance class labels across splits, as in the paper.",
)
def partition(dataset_path, out, nb_folds, seed, train_valid_ratio, balanced):
    """Split samples into folds (upstream partition_data.py)."""
    from dietnet import upstream

    out = _npz(out)
    args = ["--exp-path", out.parent.resolve(), "--dataset", dataset_path.resolve(),
            "--out", out.stem, "--nb-folds", nb_folds, "--seed", seed,
            "--train-valid-ratio", train_valid_ratio]  # fmt: skip
    if balanced:
        args.append("--balance-class")
    upstream.run("partition_data.py", args)


def _per_fold_step(script, dataset_path, partition_path, out, ncpus):
    from dietnet import upstream

    out = _npz(out)
    upstream.run(script, [
        "--exp-path", out.parent.resolve(), "--dataset", dataset_path.resolve(),
        "--partition", partition_path.resolve(), "--out", out.stem, "--ncpus", ncpus,
    ])  # fmt: skip


_dataset = click.option(
    "--dataset", "dataset_path", required=True, type=click.Path(exists=True, path_type=Path)
)
_partition = click.option(
    "--partition", "partition_path", required=True, type=click.Path(exists=True, path_type=Path)
)
_out = click.option("--out", required=True, type=click.Path(path_type=Path), help="Output .npz.")
_ncpus = click.option("--ncpus", default=1, show_default=True)


@main.command("generate-embedding")
@_dataset
@_partition
@_out
@_ncpus
def generate_embedding(dataset_path, partition_path, out, ncpus):
    """Per-fold genotype x class frequency embedding (upstream generate_embedding.py)."""
    from dietnet import training

    _per_fold_step("generate_embedding.py", dataset_path, partition_path, out, ncpus)
    try:
        training.check_inputs(_npz(out), None, dataset_path, partition_path)
    except training.InputError as error:
        raise click.ClickException(str(error)) from None


@main.command("build-embedding")
@_dataset
@_partition
@click.option(
    "--spec",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="YAML list of feature blocks (see dietnet.features).",
)
@_out
def build_embedding(dataset_path, partition_path, spec, out):
    """SNP features for the auxiliary network from several sources (writes embedding.npz)."""
    from dietnet import features, training

    out = _npz(out)
    try:
        values, meta = features.build(spec, dataset_path, partition_path, out)
        training.check_inputs(out, None, dataset_path, partition_path)
    except (ValueError, training.InputError) as error:
        raise click.ClickException(str(error)) from None
    for block in meta["blocks"]:
        start, stop = block["columns"]
        click.echo(f"  {block['provider']:20s} columns {start}-{stop - 1}")
    click.echo(f"wrote {out} {tuple(values.shape)} and {out}.json")


@main.command("alphagenome")
@click.option("--dataset", "dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--bim",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Score a PLINK .bim's SNPs instead of a dataset's (needs only the SNP ids).",
)
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Feature table (.tsv).")
@click.option(
    "--scorers",
    default=",".join(("ATAC", "DNASE", "CHIP_TF", "CHIP_HISTONE", "CAGE", "RNA_SEQ")),
    show_default=True,
    help="AlphaGenome recommended scorers, comma-separated.",
)
@click.option(
    "--sequence-length",
    type=click.Choice(["16KB", "100KB", "500KB", "1MB"]),
    default="1MB",
    show_default=True,
)
@click.option("--chunk", default=200, show_default=True, help="SNPs per saved part.")
@click.option("--max-workers", default=5, show_default=True, help="Parallel API requests.")
def alphagenome_scores(dataset_path, bim, out, scorers, sequence_length, chunk, max_workers):
    """AlphaGenome variant effects for every SNP (hosted API; key in ALPHAGENOME_API_KEY).

    Writes a table for build-embedding's precomputed provider (orient: effect).
    Resumable: rerun the same command after an interruption.
    """
    from dietnet import alphagenome

    if (dataset_path is None) == (bim is None):
        raise click.UsageError("give exactly one of --dataset or --bim")
    try:
        backend = alphagenome.AlphaGenomeAPI(
            scorers=[s.strip() for s in scorers.split(",") if s.strip()],
            sequence_length=sequence_length,
            max_workers=max_workers,
        )
        alphagenome.score(
            dataset_path or bim,
            out,
            backend,
            chunk=chunk,
            progress=lambda n, total: click.echo(f"scored {n}/{total}", err=True),
        )
    except (ImportError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from None
    click.echo(f"wrote {out}")


@main.command("compute-input-stats")
@_dataset
@_partition
@_out
@_ncpus
def compute_input_stats(dataset_path, partition_path, out, ncpus):
    """Per-fold SNP means and SDs (upstream compute_input_features_statistics.py)."""
    _per_fold_step("compute_input_features_statistics.py", dataset_path, partition_path, out, ncpus)


@main.command()
@_dataset
@_partition
@click.option(
    "--embedding",
    type=click.Path(exists=True, path_type=Path),
    help="Auxiliary-network input (required for the Diet Network; unused by the MLP).",
)
@click.option("--input-stats", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--exp-dir", required=True, type=click.Path(path_type=Path), help="Runs, logs, packages."
)
@click.option("--seeds", multiple=True, callback=_ints, required=True, help="e.g. 78,79,80")
@click.option("--folds", multiple=True, callback=_ints, required=True, help="e.g. 0,1,2,3,4")
@click.option(
    "--task", type=click.Choice(["classification", "regression"]), default="classification"
)
@click.option("--normalize/--no-normalize", default=True, show_default=True)
@click.option(
    "--model",
    type=click.Choice(["dietnet", "mlp"]),
    default="dietnet",
    show_default=True,
    help="Upstream's Diet Network, or its MLP baseline (no auxiliary network).",
)
def train(
    dataset_path,
    partition_path,
    embedding,
    input_stats,
    config,
    exp_dir,
    seeds,
    folds,
    task,
    normalize,
    model,
):
    """Train each (seed, fold) with upstream train.py; package the results."""
    from dietnet import training

    try:
        packages = training.train(
            dataset=dataset_path, partition=partition_path, embedding=embedding,
            input_stats=input_stats, config=config, exp_dir=exp_dir, seeds=seeds,
            folds=folds, task=task, normalize=normalize,
            model={"dietnet": "Dietnet", "mlp": "Mlp"}[model],
        )  # fmt: skip
    except (training.InputError, ValueError) as error:
        raise click.ClickException(str(error)) from None
    for pkg in packages:
        click.echo(f"packaged {pkg.path}")


_model = click.option(
    "--model",
    "model_dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="A package or a directory of packages.",
)
_plink = click.option("--plink", "plink_prefix", required=True, help="Cohort PLINK prefix.")
_seeds = click.option("--seeds", multiple=True, callback=_ints, help="Only these seeds.")
_folds = click.option("--folds", multiple=True, callback=_ints, help="Only these folds.")


# The published method's advice (Rochefort-Boulanger, Scicluna et al. 2026):
# do not rely on predictions from fewer model SNPs than this.
MIN_SNPS = 1000


def _warn_if_sparse(counts: dict) -> None:
    found = counts.get("model_snps", 0) - counts.get("missing", 0)
    if found < MIN_SNPS:
        click.echo(
            f"WARNING: only {found} of the model's {counts.get('model_snps')} SNPs were found in "
            f"the cohort; predictions from fewer than {MIN_SNPS} are not reliable.",
            err=True,
        )


def _packages(model_dir, seeds, folds):
    from dietnet import package

    packages = [
        p
        for p in package.find(model_dir)
        if (not seeds or p.metadata["seed"] in seeds) and (not folds or p.metadata["fold"] in folds)
    ]
    if not packages:
        raise click.UsageError("no package matches --seeds/--folds")
    return packages


@main.command()
@_model
@_plink
@click.option(
    "--work-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Where the matched cohort HDF5 goes.",
)
@_seeds
@_folds
def match(model_dir, plink_prefix, work_dir, seeds, folds):
    """Match a cohort to the models' SNPs, ahead of predict.

    CPU and I/O only. Run it as its own job, then `dietnet predict` with the
    same --work-dir on a GPU: predict finds the matched cohort and goes
    straight to inference.
    """
    import json

    import h5py

    from dietnet import predict as prediction

    started = time.perf_counter()
    matched = prediction.matched_dataset(_packages(model_dir, seeds, folds), plink_prefix, work_dir)
    with h5py.File(matched) as f:
        counts = json.loads(f.attrs.get("alignment", "{}"))
    click.echo(f"matched in {time.perf_counter() - started:.0f}s: {json.dumps(counts)}")
    _warn_if_sparse(counts)
    click.echo(f"wrote {matched}")


@main.command()
@_model
@_plink
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Predictions TSV.")
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    help="Where the matched cohort HDF5 goes, or is already (default: next to --out).",
)
@_seeds
@_folds
@click.option("--batch-size", default=1024, show_default=True)
@click.option("--device", default=None, help="cpu, cuda, ... (default: cuda if available).")
@click.option(
    "--scores", type=click.Path(path_type=Path), help="Also save per-model scores (.npz)."
)
def predict(model_dir, plink_prefix, out, work_dir, seeds, folds, batch_size, device, scores):
    """Predict a cohort with every selected model and combine them."""
    import json

    import numpy as np

    from dietnet import predict as prediction

    packages = _packages(model_dir, seeds, folds)
    work_dir = work_dir or out.parent
    started = time.perf_counter()
    matched = prediction.matched_dataset(packages, plink_prefix, work_dir)
    matched_at = time.perf_counter()
    result = prediction.predict(packages, matched, batch_size=batch_size, device=device)
    click.echo(
        f"matching {matched_at - started:.0f}s, "
        f"inference {time.perf_counter() - matched_at:.0f}s ({len(packages)} model(s))"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    result.table().to_csv(out, sep="\t", index=False)
    if scores:
        np.savez(scores, samples=result.samples, models=result.model_names,
                 classes=result.classes, scores=result.scores)  # fmt: skip
    counts = result.alignment
    _warn_if_sparse(counts)
    click.echo(
        f"{len(result.samples)} samples, {len(packages)} model(s); "
        f"{counts.get('model_snps', 0) - counts.get('missing', 0)}/{counts.get('model_snps', '?')} "
        f"model SNPs found (scale {result.scale:.4f}): {json.dumps(counts)}"
    )
    click.echo(f"wrote {out}")


@main.command()
@click.option(
    "--predictions", type=click.Path(exists=True, path_type=Path), help="From dietnet predict."
)
@click.option(
    "--packages",
    "packages_dir",
    type=click.Path(exists=True, path_type=Path),
    help="Report each package's held-out test accuracy.",
)
@click.option(
    "--labels", type=click.Path(exists=True, path_type=Path), help="TSV: sample id, label."
)
@click.option("--min-accuracy", type=float, help="Exit with status 1 below this.")
def check(predictions, packages_dir, labels, min_accuracy):
    """Accuracy of predictions against labels, overall and per class."""
    import pandas as pd

    from dietnet import package

    if packages_dir:
        rows = []
        for pkg in package.find(packages_dir):
            held_out = pd.read_csv(pkg.path / "test_predictions.tsv", sep="\t", dtype=str)
            rows.append({**{k: pkg.metadata[k] for k in ("seed", "fold")},
                         "n": len(held_out),
                         "accuracy": (held_out["prediction"] == held_out["label"]).mean()})  # fmt: skip
        table = pd.DataFrame(rows)
        click.echo(table.to_string(index=False))
        accuracy = table["accuracy"].mean()
        click.echo(f"mean held-out accuracy: {accuracy:.4f} (sd {table['accuracy'].std():.4f})")
    elif predictions and labels:
        preds = pd.read_csv(predictions, sep="\t", dtype=str)
        truth = pd.read_csv(labels, sep="\t", dtype=str)
        truth.columns = ["sample", "label", *truth.columns[2:]]
        merged = preds.merge(truth[["sample", "label"]], on="sample")
        if merged.empty:
            raise click.ClickException("no sample in common between predictions and labels")
        merged["correct"] = merged["prediction"] == merged["label"]
        per_class = merged.groupby("label")["correct"].agg(["size", "mean"])
        click.echo(per_class.rename(columns={"size": "n", "mean": "accuracy"}).to_string())
        accuracy = merged["correct"].mean()
        click.echo(f"accuracy: {accuracy:.4f} over {len(merged)} samples")
    else:
        raise click.UsageError("give --packages, or --predictions with --labels")

    if min_accuracy is not None and accuracy < min_accuracy:
        click.echo(f"FAIL: accuracy {accuracy:.4f} < {min_accuracy}", err=True)
        sys.exit(1)


@main.command()
@click.option("--dest", required=True, type=click.Path(path_type=Path), help="Where packages go.")
@click.option(
    "--release",
    type=click.Path(exists=True, path_type=Path),
    help="An already unpacked Dietnet_training/ (skips the download).",
)
def published(dest, release):
    """Fetch the 15 published models (Zenodo) as model packages."""
    from dietnet import published as pub

    release = release or pub.fetch(dest / "download")
    packages = pub.import_training_release(release, dest / "packages")
    click.echo(f"{len(packages)} published models in {dest / 'packages'}")


@main.command()
def info():
    """Versions, and whether the upstream code is exactly the pinned commit."""
    import torch

    from dietnet import upstream

    click.echo(f"dietnet {__version__}")
    click.echo(f"upstream HussinLab/DIETNETWORK {upstream.commit()} at {upstream.root()}")
    problems = upstream.verify()
    click.echo(
        "upstream files: "
        + ("identical to the pinned commit" if not problems else "; ".join(problems))
    )
    click.echo(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
