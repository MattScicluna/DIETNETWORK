"""Train with upstream's train.py and package what it produces.

Training is upstream's script, run unmodified: the published model, loop,
early stopping and checkpoint selection. This module supplies its inputs (one
config per seed, since upstream reads the seed from the config), finds its
results directory by upstream's own naming, and turns each (seed, fold) into a
model package with the fold's held-out predictions next to it.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from dietnet import package, upstream


def fold_slice(array: np.ndarray, fold: int) -> np.ndarray:
    """The fold's embedding from a per-fold (folds x SNPs x features) array, or a
    fold-independent (SNPs x features) one as it is -- the two shapes upstream's
    DietNetwork accepts."""
    array = np.asarray(array)
    return array[fold] if array.ndim == 3 else array


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
) -> list[package.Package]:
    """Train every (seed, fold) with upstream's train.py; return their packages."""
    dataset, partition = Path(dataset).resolve(), Path(partition).resolve()
    embedding, input_stats = Path(embedding).resolve(), Path(input_stats).resolve()
    exp_dir = Path(exp_dir).resolve()
    base = yaml.safe_load(Path(config).read_text())

    emb_by_fold = np.load(embedding, allow_pickle=True)["emb"]
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
                "--model", "Dietnet", "--dataset", dataset, "--partition", partition,
                "--embedding", embedding, "--input-features-stats", input_stats,
                "--task", task, "--which-fold", fold,
            ]  # fmt: skip
            if normalize:
                args.append("--normalize")
            upstream.run(
                "train.py", args, log=exp_dir / "logs" / f"train_seed{seed}_fold{fold}.log"
            )

            results = exp_dir / "runs" / f"RESULTS_{upstream.exp_identifier(cfg, fold, task)}"
            pkg = package.build(
                exp_dir / "packages" / f"seed_{seed}" / f"fold_{fold}",
                checkpoint=results / "best_model.pt",
                config=cfg,
                embedding=fold_slice(emb_by_fold, fold),
                means=stats["means_by_fold"][fold],
                sds=stats["sd_by_fold"][fold] if "sd_by_fold" in stats.files else None,
                train_dataset=dataset,
                task=task,
                normalize=normalize,
                seed=seed,
                fold=fold,
                extra={"results_dir": str(results)},
            )
            _held_out_predictions(results, dataset, task).to_csv(
                pkg.path / "test_predictions.tsv", sep="\t", index=False
            )
            packages.append(pkg)
    return packages
