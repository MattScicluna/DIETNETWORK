"""The documented pipeline, run through the CLI on the committed fixtures.

partition -> generate-embedding -> compute-input-stats -> train -> predict -> check,
with the commands and flags README.md gives. Small enough to run on every pull
request; it is what would notice a broken hand-off between two commands that
are each fine on their own, or a change that quietly costs accuracy.
"""

import json
import shutil

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from Dietnet.cli import main

pytestmark = pytest.mark.integration

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
    "seed": 3,
    "uniform_init_limit": 0.02,
}


def dietnet(command, *flags, **options):
    """Run ``dietnet COMMAND --opt value ... FLAGS`` and insist it succeeds.

    Options are keyword arguments, underscores for dashes, so a pipeline step
    reads like the command line it stands for.
    """
    args = [command]
    for name, value in options.items():
        args += [f"--{name.replace('_', '-')}", str(value)]
    args += list(flags)

    result = CliRunner().invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


class Experiment:
    """A fresh experiment directory holding one fixture, taken through the pipeline."""

    def __init__(self, root, dataset, task, label_file=None):
        self.dir = root / "exp"
        self.task = task
        self.prefix = self.dir / "data"
        self.labels = label_file or dataset.labels

        self.dir.mkdir()
        for ext in (".bed", ".bim", ".fam"):
            shutil.copy(f"{dataset.plink_prefix}{ext}", f"{self.prefix}{ext}")
        shutil.copy(self.labels, self.dir / "labels.tsv")
        (self.dir / "run").mkdir()
        (self.dir / "run" / "config.yaml").write_text(yaml.safe_dump(CONFIG))

        common = dict(exp_path=self.dir, dataset="data.bed")
        stratify = "--stratify" if task == "classification" else "--no-stratify"
        dietnet("partition", stratify, **common, label_file="labels.tsv", nb_folds=5)
        dietnet(
            "generate-embedding",
            **common,
            partition="partitioned_idx.npz",
            label_file="labels.tsv",
            task=task,
        )
        dietnet("compute-input-stats", **common, partition="partitioned_idx.npz")

    @property
    def package(self):
        return self.dir / "packages" / f"seed_{CONFIG['seed']}" / "fold_0"

    def train(self, *flags):
        dietnet(
            "train",
            *flags,
            exp_path=self.dir,
            exp_name="run",
            config="config.yaml",
            plink_prefix=self.prefix,
            label_file="labels.tsv",
            partition="partitioned_idx.npz",
            embedding="embedding.npz",
            input_features_means="input_features_means.npz",
            seeds=CONFIG["seed"],
            folds=0,
            output_dir=self.dir / "packages",
            task=self.task,
        )

    def predict(self, output, *flags):
        dietnet(
            "predict",
            "--skip-preprocess",
            *flags,
            model=self.dir / "packages",
            plink_prefix=self.prefix,
            output=output,
            seeds=CONFIG["seed"],
            folds=0,
        )

    def check(self, predictions, min_accuracy):
        """Accuracy on fold 0's test samples, which training never saw."""
        return dietnet(
            "check",
            predictions=predictions,
            labels=self.labels,
            partition_file=self.dir / "partitioned_idx.npz",
            fold=0,
            min_accuracy=min_accuracy,
        )


# Chance is 1/4 on synthetic and 1/24 on kgp_mini. kgp_mini scored 40-45% over
# seeds 1-3 when this was calibrated (2,000 SNPs, 336 training samples), so its
# bar sits at six times chance with room for seed noise.
MIN_ACCURACY = {"synthetic": 0.6, "kgp_mini": 0.25}


def test_classification_pipeline_beats_chance(tmp_path, dataset):
    exp = Experiment(tmp_path, dataset, "classification")
    exp.train()

    predictions = tmp_path / "predictions.tsv"
    exp.predict(predictions)
    exp.check(predictions, MIN_ACCURACY[dataset.name])

    # One headerless "sample<TAB>label" row per sample in the PLINK file.
    n_samples = sum(1 for _ in open(f"{dataset.plink_prefix}.fam"))
    assert len(pd.read_csv(predictions, sep="\t", header=None)) == n_samples


@pytest.mark.xfail(reason="B4: the package does not record whether inputs were normalised")
def test_package_records_input_normalisation(tmp_path, synthetic):
    exp = Experiment(tmp_path, synthetic, "classification")
    exp.train("--no-normalize")

    metadata = json.loads((exp.package / "metadata.json").read_text())
    assert metadata.get("preprocessing", {}).get("normalize") is False


@pytest.mark.xfail(
    reason="B16: regression is broken end to end -- generate-embedding computes per-class "
    "frequencies over the float trait and crashes; predict then builds a "
    "len(label_mapping) == 0 output layer"
)
def test_regression_pipeline_predicts(tmp_path, synthetic):
    trait = synthetic.labels.with_name("synthetic_trait.tsv")
    exp = Experiment(tmp_path, synthetic, "regression", label_file=trait)
    exp.train()

    exp.predict(tmp_path / "predictions.tsv", "--task", "regression")
