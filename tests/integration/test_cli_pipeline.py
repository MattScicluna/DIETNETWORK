"""The pipeline through the command line, on the committed fixtures.

create-dataset -> partition -> generate-embedding -> compute-input-stats ->
train -> predict -> check, as README.md gives it. Small enough for every pull
request; what would notice a broken hand-off between two commands that are
each fine alone, or a change that quietly costs accuracy.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner
from support import SEED, dietnet, train_pipeline

from dietnet import package
from dietnet.cli import main

pytestmark = pytest.mark.integration

# Chance is 1/4 on synthetic and 1/24 on kgp_mini. kgp_mini's held-out accuracy
# was 40-45% over seeds when this was calibrated (2,000 SNPs, ~290 training
# samples), so its floor is six times chance with room for seed noise.
MIN_ACCURACY = {"synthetic": 0.9, "kgp_mini": 0.25}


@pytest.fixture(scope="module")
def trained(tmp_path_factory, request):
    """Each fixture trained once for this module, keyed by name."""
    cache = {}

    def get(data):
        if data.name not in cache:
            cache[data.name] = train_pipeline(tmp_path_factory.mktemp(data.name), data)
        return cache[data.name]

    return get


def test_held_out_accuracy_beats_chance(trained, dataset):
    """Upstream's own test split, as packaged after training."""
    out = dietnet("check", packages=trained(dataset), min_accuracy=MIN_ACCURACY[dataset.name])
    assert "mean held-out accuracy" in out


def test_predict_writes_one_row_per_sample(tmp_path, trained, dataset):
    predictions = tmp_path / "predictions.tsv"
    dietnet("predict", model=trained(dataset), plink=dataset.plink_prefix, out=predictions)

    table = pd.read_csv(predictions, sep="\t")
    n_samples = sum(1 for _ in open(f"{dataset.plink_prefix}.fam"))
    assert len(table) == n_samples
    assert list(table.columns) == ["sample", "prediction", "probability", "model_agreement"]
    assert table["probability"].between(0, 1).all()
    # Two models (folds 0, 1). Agreement can be 0: with many classes the mean
    # probability can favour a class neither model ranked first.
    assert set(table["model_agreement"]) <= {0.0, 0.5, 1.0}


def test_check_scores_predictions_against_labels(tmp_path, trained, synthetic):
    predictions = tmp_path / "predictions.tsv"
    dietnet("predict", model=trained(synthetic), plink=synthetic.plink_prefix, out=predictions)
    out = dietnet("check", predictions=predictions, labels=synthetic.labels, min_accuracy=0.9)
    assert "accuracy" in out


def test_check_fails_below_the_minimum(trained, synthetic):
    result = CliRunner().invoke(
        main, ["check", "--packages", str(trained(synthetic)), "--min-accuracy", "1.01"]
    )
    assert result.exit_code == 1


def test_seed_and_fold_selection(tmp_path, trained, synthetic):
    scores = tmp_path / "scores.npz"
    dietnet(
        "predict",
        model=trained(synthetic),
        plink=synthetic.plink_prefix,
        out=tmp_path / "p.tsv",
        folds=1,
        scores=scores,
    )
    assert list(np.load(scores)["models"]) == [f"seed{SEED}_fold1"]


def test_matched_cohort_goes_to_the_work_dir_not_beside_the_cohort(tmp_path, trained, synthetic):
    """The matched copy is as large as the cohort; it must go where the user says."""
    before = set(synthetic.plink_prefix.parent.iterdir())
    dietnet(
        "predict",
        model=trained(synthetic),
        plink=synthetic.plink_prefix,
        out=tmp_path / "p.tsv",
        work_dir=tmp_path / "work",
    )
    assert set(synthetic.plink_prefix.parent.iterdir()) == before
    assert list((tmp_path / "work").glob("*.matched.h5"))


def test_packages_record_preprocessing(tmp_path, synthetic):
    packages = train_pipeline(tmp_path, synthetic, folds="0", flags=["--no-normalize"])
    (pkg,) = package.find(packages)
    assert pkg.metadata["preprocessing"]["normalize"] is False
    assert pkg.metadata["upstream_commit"]
    mus, sds = pkg.stats("cpu")
    assert sds is None


def test_a_checkpoint_that_does_not_fit_refuses_to_load(tmp_path, trained, synthetic):
    """Strict loading: a missing layer is an error, not random weights."""
    pkg = package.find(trained(synthetic))[0]
    broken = tmp_path / "broken"
    import shutil

    shutil.copytree(pkg.path, broken)
    checkpoint = torch.load(broken / "model.pt", weights_only=False)
    del checkpoint["model_state_dict"]["main_net.out.weight"]
    torch.save(checkpoint, broken / "model.pt")

    with pytest.raises(RuntimeError, match="main_net.out.weight"):
        package.Package(broken).handler("cpu")


def test_regression_end_to_end(tmp_path, synthetic):
    """Upstream builds the regression embedding from class labels stored beside the trait."""
    packages = train_pipeline(tmp_path / "reg", synthetic, folds="0", task="regression")
    predictions = tmp_path / "predictions.tsv"
    dietnet("predict", model=packages, plink=synthetic.plink_prefix, out=predictions)

    table = pd.read_csv(predictions, sep="\t")
    assert list(table.columns) == ["sample", "prediction", "sd_across_models"]
    assert np.isfinite(table["prediction"]).all()


def test_info_reports_unmodified_upstream():
    out = dietnet("info")
    assert "identical to the pinned commit" in out


def test_match_then_predict_reuses_the_matched_cohort(tmp_path, trained, synthetic):
    """match (CPU job) then predict (GPU job): predict goes straight to inference."""
    work = tmp_path / "work"
    out = dietnet("match", model=trained(synthetic), plink=synthetic.plink_prefix, work_dir=work)
    (matched,) = work.glob("*.matched.h5")
    assert str(matched) in out
    stamp = matched.stat().st_mtime_ns

    dietnet(
        "predict",
        model=trained(synthetic),
        plink=synthetic.plink_prefix,
        out=tmp_path / "p.tsv",
        work_dir=work,
    )
    assert matched.stat().st_mtime_ns == stamp
