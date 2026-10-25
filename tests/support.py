"""Test datasets and helpers shared across the suite.

Imported by name (``from support import ...``); pytest puts tests/ on the path.
The datasets are described in tests/fixtures/build_fixtures.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from click.testing import CliRunner

from dietnet import upstream
from dietnet.cli import main

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Small enough to train in seconds, large enough to learn both fixtures.
CONFIG = {
    "batch_size": 32,
    "epochs": 60,
    "input_dropout": 0.5,
    "dropout_main": 0.5,
    "lr_aux": 0.001,
    "lr_main": 0.001,
    "learning_rate_annealing": 0.99,
    "nb_hidden_u_aux": [64, 64],
    "nb_hidden_u_main": [64],
    "patience": 60,
    "seed": 1,
    "uniform_init_limit": 0.02,
}
SEED = 3


@dataclass(frozen=True)
class Dataset:
    """One committed fixture: PLINK prefix, labels, and upstream's HDF5 of it."""

    name: str
    plink_prefix: Path
    labels: Path
    upstream_hdf5: Path


DATASETS = {
    name: Dataset(
        name=name,
        plink_prefix=FIXTURES / name,
        labels=FIXTURES / f"{name}_labels.tsv",
        upstream_hdf5=FIXTURES / f"{name}_upstream.hdf5",
    )
    for name in ("kgp_mini", "synthetic")
}


def dietnet(command, *flags, **options):
    """Run ``dietnet COMMAND --opt value ... FLAGS`` and insist it succeeds.

    Options are keyword arguments with underscores for dashes, so a pipeline
    step reads like the command line it stands for.
    """
    args = [command]
    for name, value in options.items():
        args += [f"--{name.replace('_', '-')}", str(value)]
    args += [str(f) for f in flags]
    result = CliRunner().invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


def train_pipeline(root: Path, data: Dataset, folds="0,1", task="classification", **extra):
    """create-dataset -> partition -> embedding -> stats -> train, under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    labels = {"labels": data.labels}
    if task == "regression":
        labels["regression_labels"] = data.labels.with_name("synthetic_trait.tsv")
    dietnet("create-dataset", plink=data.plink_prefix, out=root / "dataset.h5", **labels)
    common = dict(dataset=root / "dataset.h5", partition=root / "partition.npz")
    dietnet("partition", dataset=root / "dataset.h5", out=root / "partition.npz")
    dietnet("generate-embedding", **common, out=root / "embedding.npz")
    dietnet("compute-input-stats", **common, out=root / "stats.npz")
    dietnet(
        "train",
        *extra.pop("flags", ()),
        **common,
        embedding=root / "embedding.npz",
        input_stats=root / "stats.npz",
        config=root / "config.yaml",
        exp_dir=root / "exp",
        seeds=SEED,
        folds=folds,
        task=task,
    )
    return root / "exp" / "packages"


def ci_pinned_commit() -> str:
    """The upstream commit CI clones for the parity tests."""
    ci = yaml.safe_load((FIXTURES.parents[1] / ".github" / "workflows" / "ci.yml").read_text())
    return ci["env"]["UPSTREAM_SHA"]


def manifest_commit() -> str:
    return upstream.manifest()["commit"]


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())
