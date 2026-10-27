"""Train with upstream's train.py and package what it produces.

Training is upstream's script, run unmodified: the published model, loop,
early stopping and checkpoint selection. This module supplies its inputs (one
config per seed, since upstream reads the seed from the config), finds its
results directory by upstream's own naming, and turns each (seed, fold) into a
model package with the fold's held-out predictions next to it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from dietnet import _launch, genotypes, package, upstream


def default_workers() -> int:
    """Loader processes: the CPUs this process may use (a SLURM allocation's, not
    the node's), less one for training itself, at most 8. Past ~8 the disk, not
    the CPUs, sets the pace (docs/scaling.md)."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        cpus = os.cpu_count() or 1
    return max(0, min(8, cpus - 1))


def fold_slice(array: np.ndarray, fold: int) -> np.ndarray:
    """The fold's embedding from a per-fold (folds x SNPs x features) array, or a
    fold-independent (SNPs x features) one as it is -- the two shapes upstream's
    DietNetwork accepts."""
    array = np.asarray(array)
    return array[fold] if array.ndim == 3 else array


class InputError(ValueError):
    """Training inputs upstream's model cannot learn from."""


def check_inputs(embedding, input_stats, dataset, partition, folds=None) -> None:
    """Refuse per-fold inputs upstream's model cannot learn from, saying why.

    ``input_stats`` may be None: generate-embedding checks before they exist;
    ``embedding`` may be None: the MLP baseline has none.

    Upstream's generate_embedding.py divides each class's genotype counts by
    the number of its samples in the fold's training split, and sizes the
    embedding by the largest class index present. A class with no training
    samples in a fold -- a population of one or two people, split five ways --
    therefore either gives 0/0 (NaN, which spreads to every weight: the model
    trains to NaN and predicts the first class for everyone) or, if it is the
    last class, silently drops its columns. Either way the class cannot be
    learned. Caught here, before any training time is spent.
    """
    emb = np.load(embedding, allow_pickle=True)["emb"] if embedding else None
    stats = np.load(input_stats, allow_pickle=True) if input_stats else None
    parts = np.load(partition, allow_pickle=True)["folds_indexes"]
    with h5py.File(dataset, "r") as f:
        labels = np.array(f["class_labels"]) if "class_labels" in f else None
        names = np.array(f["class_label_names"]).astype(str) if "class_label_names" in f else None

    problems = []
    for fold in range(len(parts)) if folds is None else folds:
        if labels is not None:
            train = np.asarray(parts[fold][0], dtype=int)
            absent = names[np.bincount(labels[train], minlength=len(names)) == 0]
            if len(absent):
                problems.append(f"fold {fold}: no training samples for {', '.join(absent)}")
        nan = ["embedding"] if emb is not None and np.isnan(fold_slice(emb, fold)).any() else []
        for key in ("means_by_fold", "sd_by_fold"):
            if stats is not None and key in stats.files and np.isnan(stats[key][fold]).any():
                nan.append(key)
        if nan:
            problems.append(f"fold {fold}: NaN in {', '.join(nan)}")

    if problems:
        raise InputError(
            "the model cannot learn from these training inputs:\n  "
            + "\n  ".join(problems)
            + "\nMerge or drop classes too small to appear in every fold's training split "
            "(at least nb-folds samples each), then rebuild the dataset and rerun."
        )


def _held_out_predictions(results_dir: Path, train_dataset: Path, task: str) -> pd.DataFrame:
    """Upstream's test_results_epoch*.npz as a table of sample ids."""
    (path,) = sorted(results_dir.glob("test_results_epoch*.npz"))
    results = np.load(path)
    with h5py.File(train_dataset, "r") as f:
        samples = np.array(f["samples"]).astype(str)
        names = np.array(f["class_label_names"]).astype(str) if "class_label_names" in f else None
    rows = results["samples"].astype(int)
    table = pd.DataFrame({"sample": samples[rows]})
    if task == "classification":
        table["prediction"] = names[results["preds"].astype(int)]
        table["label"] = names[results["labels"].astype(int)]
    else:
        table["prediction"] = results["preds"]
        table["label"] = results["labels"]
    return table


def train(
    *,
    dataset,
    partition,
    embedding,
    input_stats,
    config,
    exp_dir,
    seeds,
    folds,
    task="classification",
    normalize=True,
    model="Dietnet",
    inputs="auto",
    loader="fast",
    workers: int | None = None,
) -> list[package.Package]:
    """Train every (seed, fold) with upstream's train.py; return their packages.

    ``model`` is upstream's: "Dietnet", or "Mlp", the baseline without an
    auxiliary network, which takes no embedding. How train.py is fed, which
    leaves the trained model bit-identical:

    - ``inputs``: the genotype store it reads (:mod:`dietnet.genotypes`): "ram"
      (upstream's copy of the whole dataset), "memmap", "packed", or "auto".
    - ``loader``: "fast" reads each batch in one call with ``workers`` loader
      processes (default :func:`default_workers`) and fills missing values in
      one operation; "upstream" leaves train.py's own loader alone.
    """
    if loader not in ("fast", "upstream"):
        raise ValueError(f"loader must be 'fast' or 'upstream', not {loader!r}")
    if model not in upstream.MODELS:
        raise ValueError(f"model must be one of {sorted(upstream.MODELS)}")
    if (embedding is None) != (model == "Mlp"):
        raise ValueError("an embedding is needed for the Diet Network, and unused by the MLP")
    dataset, partition = Path(dataset).resolve(), Path(partition).resolve()
    embedding = Path(embedding).resolve() if embedding else None
    input_stats = Path(input_stats).resolve()
    exp_dir = Path(exp_dir).resolve()
    base = yaml.safe_load(Path(config).read_text())
    check_inputs(embedding, input_stats, dataset, partition, folds)
    try:
        env = {_launch.INPUTS_ENV: genotypes.choose(dataset, inputs)}
    except ValueError as error:
        raise InputError(str(error)) from None
    if loader == "fast":
        workers = default_workers() if workers is None else workers
        env.update({_launch.LOADER_ENV: "fast", _launch.WORKERS_ENV: str(workers)})

    emb_by_fold = np.load(embedding, allow_pickle=True)["emb"] if embedding else None
    described = Path(f"{embedding}.json") if embedding else None
    feature_meta = json.loads(described.read_text()) if described and described.exists() else None
    stats = np.load(input_stats, allow_pickle=True)

    packages = []
    for seed in seeds:
        cfg = {**base, "seed": int(seed)}
        cfg_path = exp_dir / "configs" / f"config_seed{seed}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        for fold in folds:
            args = [
                "--exp-path", exp_dir, "--exp-name", "runs", "--config", cfg_path,
                "--model", model, "--dataset", dataset, "--partition", partition,
                # Required by train.py, read only for the Diet Network.
                "--embedding", embedding or "none", "--input-features-stats", input_stats,
                "--task", task, "--which-fold", fold,
            ]  # fmt: skip
            if normalize:
                args.append("--normalize")
            upstream.run(
                "train.py",
                args,
                log=exp_dir / "logs" / f"train_seed{seed}_fold{fold}.log",
                env=env,
            )

            results = (
                exp_dir / "runs" / f"RESULTS_{upstream.exp_identifier(cfg, fold, task, model)}"
            )
            pkg = package.build(
                exp_dir / "packages" / f"seed_{seed}" / f"fold_{fold}",
                checkpoint=results / "best_model.pt",
                config=cfg,
                embedding=None if emb_by_fold is None else fold_slice(emb_by_fold, fold),
                means=stats["means_by_fold"][fold],
                sds=stats["sd_by_fold"][fold] if "sd_by_fold" in stats.files else None,
                train_dataset=dataset,
                task=task,
                normalize=normalize,
                seed=seed,
                fold=fold,
                model=model,
                extra={"results_dir": str(results), "embedding": feature_meta},
            )
            _held_out_predictions(results, dataset, task).to_csv(
                pkg.path / "test_predictions.tsv", sep="\t", index=False
            )
            packages.append(pkg)
    return packages
