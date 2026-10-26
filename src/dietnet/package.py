"""Model packages: one trained (seed, fold) model, self-contained and portable.

A package is a directory holding everything upstream's model needs, so that
prediction never needs the training data or the experiment directory::

    model.pt                  upstream's best_model.pt, as train.py saved it
    config.yaml               the training config (seed included)
    embedding.npz             ``emb``: this fold's SNP embedding (SNPs x features);
                              absent for the MLP baseline, which has no auxiliary net
    input_features_stats.npz  ``means_by_fold`` / ``sd_by_fold`` for this fold only
    train_header.h5           ``snp_names``, ``class_label_names``, ``snps/``: the
                              parts of the training dataset upstream's classes read
    metadata.json             task, normalisation, seed, fold, upstream commit, ...

The model is rebuilt with upstream's own classes (``DietNetworkHandler``) and its
weights loaded strictly: a checkpoint that does not fit fails instead of leaving
layers at their random initialisation.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from dietnet import __version__, dataset, upstream

FORMAT_VERSION = 2


def _write_header(train_dataset, out: Path) -> None:
    with h5py.File(train_dataset, "r") as src, h5py.File(out, "w") as dst:
        for key in ("snp_names", "class_label_names"):
            if key in src:
                dst.create_dataset(key, data=np.array(src[key]))
        if "snps" in src:
            src.copy(src["snps"], dst, "snps")


def build(
    out,
    *,
    checkpoint,
    config: dict,
    embedding: np.ndarray | None,
    means: np.ndarray,
    sds: np.ndarray | None,
    train_dataset,
    task: str,
    normalize: bool,
    seed: int,
    fold: int,
    model: str = "Dietnet",
    extra: dict | None = None,
) -> "Package":
    """Assemble a package from a finished upstream training run.

    ``embedding`` is the fold's auxiliary-network input; None for the MLP,
    which has no auxiliary network.
    """
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copy(checkpoint, out / "model.pt")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if embedding is not None:
        np.savez(out / "embedding.npz", emb=np.asarray(embedding))
    stats = {"means_by_fold": np.asarray(means)[None]}
    if sds is not None:
        stats["sd_by_fold"] = np.asarray(sds)[None]
    np.savez(out / "input_features_stats.npz", **stats)
    _write_header(train_dataset, out / "train_header.h5")

    with h5py.File(out / "train_header.h5", "r") as f:
        n_snps = len(f["snp_names"])
        classes = (
            list(np.array(f["class_label_names"]).astype(str)) if task == "classification" else []
        )
    if embedding is not None and np.asarray(embedding).shape[0] != n_snps:
        raise ValueError(f"embedding has {np.asarray(embedding).shape[0]} rows for {n_snps} SNPs")

    metadata = {
        "format_version": FORMAT_VERSION,
        "model": model,
        "task": task,
        "classes": classes,
        "preprocessing": {"normalize": bool(normalize), "missing_snps": "scale D/(D-|C|)"},
        "seed": int(seed),
        "fold": int(fold),
        "n_snps": n_snps,
        "upstream_commit": upstream.commit(),
        "dietnet_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **(extra or {}),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return Package(out)


@dataclass(frozen=True)
class Package:
    path: Path

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
        if not (self.path / "metadata.json").is_file():
            raise FileNotFoundError(f"{self.path} is not a model package (no metadata.json)")

    @property
    def metadata(self) -> dict:
        return json.loads((self.path / "metadata.json").read_text())

    @property
    def config(self) -> dict:
        return yaml.safe_load((self.path / "config.yaml").read_text())

    @property
    def task(self) -> str:
        return self.metadata["task"]

    @property
    def model(self) -> str:
        return self.metadata.get("model", "Dietnet")

    @property
    def normalize(self) -> bool:
        return self.metadata["preprocessing"]["normalize"]

    @property
    def classes(self) -> list[str]:
        return self.metadata["classes"]

    @property
    def name(self) -> str:
        m = self.metadata
        return f"seed{m['seed']}_fold{m['fold']}"

    def snps(self) -> pd.DataFrame:
        return dataset.snp_table(self.path / "train_header.h5")

    def stats(self, device) -> tuple[torch.Tensor, torch.Tensor | None]:
        data = np.load(self.path / "input_features_stats.npz")
        mus = torch.from_numpy(data["means_by_fold"][0]).float().to(device)
        sds = None
        if self.normalize:
            sds = torch.from_numpy(data["sd_by_fold"][0]).float().to(device)
        return mus, sds

    def handler(self, device):
        """Upstream's model handler with this package's weights, in eval mode."""
        m = upstream.modules()
        header = str(self.path / "train_header.h5")
        if self.task == "classification":
            task = m.task_handlers.ClassificationHandler(header, nn.CrossEntropyLoss())
        else:
            task = m.task_handlers.RegressionHandler(header, nn.MSELoss())
        if self.model == "Mlp":
            handler = m.model_handlers.MlpHandler(task, header, self.config)
        else:
            handler = m.model_handlers.DietNetworkHandler(
                task, 0, str(self.path / "embedding.npz"), device, header, self.config, None
            )
        checkpoint = torch.load(self.path / "model.pt", map_location=device, weights_only=False)
        handler.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        handler.model.to(device)
        handler.model.eval()
        return handler


def find(root) -> list[Package]:
    """Every package at or below ``root``, in (seed, fold) order."""
    root = Path(root)
    found = [Package(p.parent) for p in root.rglob("metadata.json")]
    if not found:
        raise FileNotFoundError(f"no model packages under {root}")
    return sorted(found, key=lambda p: (p.metadata["seed"], p.metadata["fold"]))
