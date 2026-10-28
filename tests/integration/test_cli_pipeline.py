"""The pipeline through the command line, on the committed fixtures.

create-dataset -> partition -> generate-embedding -> compute-input-stats ->
train -> predict -> check, as README.md gives it. Small enough for every pull
request; what would notice a broken hand-off between two commands that are
each fine alone, or a change that quietly costs accuracy.
"""

import re

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from click.testing import CliRunner
from support import CONFIG, SEED, dietnet, train_pipeline

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

    report = dietnet("check", packages=packages)
    assert "mean held-out r:" in report and "R^2" in report


def test_regression_on_pooled_frequencies_with_the_dietnet_engine(tmp_path, synthetic):
    """The height setup: a trait with no classes, the auxiliary network fed genotype
    frequencies pooled over classes, trained by the dietnet engine."""
    root = tmp_path / "reg"
    train_pipeline(root, synthetic, folds="0", task="regression")  # dataset, partition, stats
    (root / "spec.yaml").write_text(yaml.safe_dump({"blocks": [{"provider": "genotype_freq"}]}))
    dietnet("build-embedding", dataset=root / "dataset.h5", partition=root / "partition.npz",
            spec=root / "spec.yaml", out=root / "pooled.npz")  # fmt: skip
    dietnet(
        "train",
        "--engine=dietnet",
        dataset=root / "dataset.h5",
        partition=root / "partition.npz",
        embedding=root / "pooled.npz",
        input_stats=root / "stats.npz",
        config=root / "config.yaml",
        exp_dir=root / "pooled",
        seeds=SEED,
        folds="0",
        task="regression",
        workers=0,
    )

    report = dietnet("check", packages=root / "pooled" / "packages")
    r = float(re.search(r"mean held-out r: (-?\d\.\d+)", report).group(1))
    assert r > 0.3, report


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


@pytest.mark.parametrize("name", ["AAA_Singleton", "ZZZ_Singleton"])
def test_a_class_too_small_for_every_fold_is_refused(tmp_path, synthetic, name):
    """A class missing from a fold's training split cannot be learned.

    Upstream's embedding gives it NaN columns (and the model trains to NaN), or,
    when it sorts last, silently no columns at all. One sample relabelled into a
    class of its own triggers both; dietnet stops at generate-embedding and
    names the class.
    """
    labels = pd.read_csv(synthetic.labels, sep="\t")
    labels.iloc[0, 1] = name
    labels.to_csv(tmp_path / "labels.tsv", sep="\t", index=False)
    dietnet("create-dataset", plink=synthetic.plink_prefix, labels=tmp_path / "labels.tsv",
            out=tmp_path / "d.h5")  # fmt: skip
    dietnet("partition", dataset=tmp_path / "d.h5", out=tmp_path / "p.npz")

    result = CliRunner().invoke(main, [
        "generate-embedding", "--dataset", str(tmp_path / "d.h5"),
        "--partition", str(tmp_path / "p.npz"), "--out", str(tmp_path / "e.npz"),
    ])  # fmt: skip
    assert result.exit_code == 1
    assert f"no training samples for {name}" in result.output


def test_train_and_predict_with_a_built_embedding(tmp_path, synthetic):
    """build-embedding instead of generate-embedding: frequencies plus a random block.

    Upstream's model takes the wider matrix as it is; the package records which
    blocks it was trained on.
    """
    root = tmp_path / "run"
    root.mkdir()
    (root / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    (root / "features.yaml").write_text(
        yaml.safe_dump(
            {"blocks": [{"provider": "class_genotype_freq"}, {"provider": "random", "columns": 4}]}
        )
    )
    dietnet(
        "create-dataset", plink=synthetic.plink_prefix, labels=synthetic.labels, out=root / "d.h5"
    )
    common = dict(dataset=root / "d.h5", partition=root / "p.npz")
    dietnet("partition", dataset=root / "d.h5", out=root / "p.npz")
    dietnet("build-embedding", **common, spec=root / "features.yaml", out=root / "e.npz")
    dietnet("compute-input-stats", **common, out=root / "s.npz")
    dietnet(
        "train",
        **common,
        embedding=root / "e.npz",
        input_stats=root / "s.npz",
        config=root / "config.yaml",
        exp_dir=root / "exp",
        seeds=SEED,
        folds=0,
    )

    (pkg,) = package.find(root / "exp" / "packages")
    blocks = pkg.metadata["embedding"]["blocks"]
    assert [b["provider"] for b in blocks] == ["class_genotype_freq", "random"]
    assert np.load(pkg.path / "embedding.npz")["emb"].shape == (500, 4 * 3 + 4)

    dietnet("predict", model=root / "exp" / "packages", plink=synthetic.plink_prefix,
            out=tmp_path / "p.tsv")  # fmt: skip
    dietnet("check", predictions=tmp_path / "p.tsv", labels=synthetic.labels, min_accuracy=0.9)


def test_the_mlp_baseline_trains_and_predicts_without_an_embedding(tmp_path, synthetic):
    """Upstream's MLP: the experiments' baseline without an auxiliary network."""
    root = tmp_path / "mlp"
    root.mkdir()
    (root / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    dietnet(
        "create-dataset", plink=synthetic.plink_prefix, labels=synthetic.labels, out=root / "d.h5"
    )
    common = dict(dataset=root / "d.h5", partition=root / "p.npz")
    dietnet("partition", dataset=root / "d.h5", out=root / "p.npz")
    dietnet("compute-input-stats", **common, out=root / "s.npz")
    dietnet(
        "train",
        **common,
        input_stats=root / "s.npz",
        config=root / "config.yaml",
        exp_dir=root / "exp",
        seeds=SEED,
        folds=0,
        model="mlp",
    )

    (pkg,) = package.find(root / "exp" / "packages")
    assert pkg.metadata["model"] == "Mlp" and not (pkg.path / "embedding.npz").exists()
    dietnet("predict", model=root / "exp" / "packages", plink=synthetic.plink_prefix,
            out=tmp_path / "p.tsv")  # fmt: skip
    dietnet("check", predictions=tmp_path / "p.tsv", labels=synthetic.labels, min_accuracy=0.9)


@pytest.mark.parametrize("model, embedding", [("Dietnet", None), ("Mlp", "e.npz")])
def test_an_embedding_goes_with_the_diet_network_only(tmp_path, model, embedding):
    from dietnet import training

    with pytest.raises(ValueError, match="embedding"):
        training.train(
            dataset="d.h5", partition="p.npz", embedding=embedding, input_stats="s.npz",
            config="c.yaml", exp_dir=tmp_path, seeds=[1], folds=[0], model=model,
        )  # fmt: skip
