"""The command line's error paths and less common options.

Usage errors exit with status 2, errors reported as ``click.ClickException``
with status 1. Everything here reuses the session's trained synthetic packages
or stubs out the slow or networked part.
"""

import json
import runpy
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from support import DATASETS

from dietnet import cli, package


def run(*args):
    return CliRunner().invoke(cli.main, [str(a) for a in args])


@pytest.fixture(scope="module")
def trained_root(trained_synthetic):
    """The directory train_pipeline worked in: dataset.h5, partition.npz, stats.npz, ..."""
    return trained_synthetic.parents[1]


# create-dataset, partition, build-embedding, adjust-phenotype, train


def test_create_dataset_needs_labels(tmp_path):
    result = run("create-dataset", "--plink", DATASETS["synthetic"].plink_prefix,
                 "--out", tmp_path / "d.h5")  # fmt: skip
    assert result.exit_code == 2
    assert "give --labels, --regression-labels or both" in result.output
    assert not (tmp_path / "d.h5").exists()


def test_partition_unbalanced_adds_the_npz_suffix(trained_root, tmp_path):
    """--unbalanced leaves out --balance-class; the output name gains .npz."""
    result = run("partition", "--dataset", trained_root / "dataset.h5",
                 "--out", tmp_path / "split", "--unbalanced", "--nb-folds", 3)  # fmt: skip
    assert result.exit_code == 0, result.output
    folds = np.load(tmp_path / "split.npz", allow_pickle=True)["folds_indexes"]
    assert len(folds) == 3


def test_build_embedding_reports_a_bad_spec(trained_root, tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({"blocks": [{"provider": "nonsense"}]}))
    result = run("build-embedding", "--dataset", trained_root / "dataset.h5",
                 "--partition", trained_root / "partition.npz", "--spec", spec,
                 "--out", tmp_path / "emb")  # fmt: skip
    assert result.exit_code == 1
    assert "block 0: unknown provider 'nonsense'" in result.output
    assert "Traceback" not in result.output


def test_adjust_phenotype_reports_an_unknown_column(tmp_path):
    table = tmp_path / "pheno.tsv"
    table.write_text("id\theight\tage\ns1\t1.7\t30\ns2\t1.8\t40\n")
    result = run("adjust-phenotype", "--phenotypes", table, "--trait", "weight",
                 "--covariates", "age,sex", "--out", tmp_path / "out.tsv")  # fmt: skip
    assert result.exit_code == 1
    assert "columns not in the phenotype table: ['weight', 'sex']" in result.output
    assert not (tmp_path / "out.tsv").exists()


def test_train_reports_inconsistent_options(trained_root, tmp_path):
    """Option checks in training.train surface as a one-line error, before any work."""
    result = run("train", "--dataset", trained_root / "dataset.h5",
                 "--partition", trained_root / "partition.npz",
                 "--embedding", trained_root / "embedding.npz",
                 "--input-stats", trained_root / "stats.npz",
                 "--config", trained_root / "config.yaml", "--exp-dir", tmp_path / "exp",
                 "--seeds", 1, "--folds", 0, "--engine", "upstream", "--gpus", 2)  # fmt: skip
    assert result.exit_code == 1
    assert "engine 'upstream' is train.py's own loop, on one GPU" in result.output
    assert not (tmp_path / "exp").exists()


# alphagenome


class FakeBackend:
    """Scores every variant with one track whose effect is its position."""

    def score(self, variants):
        return pd.DataFrame(
            {
                "chrom": variants["chrom"],
                "pos": variants["pos"],
                "ref": variants["ref"],
                "alt": variants["alt"],
                "output_type": "DNASE",
                "track_name": "t0",
                "raw_score": variants["pos"].astype(float),
            }
        )


@pytest.mark.parametrize("given", ["neither", "both"])
def test_alphagenome_needs_exactly_one_source(given, trained_root, tmp_path):
    sources = []
    if given == "both":
        sources = ["--dataset", trained_root / "dataset.h5",
                   "--bim", f"{DATASETS['synthetic'].plink_prefix}.bim"]  # fmt: skip
    result = run("alphagenome", *sources, "--out", tmp_path / "ag.tsv")
    assert result.exit_code == 2
    assert "give exactly one of --dataset or --bim" in result.output


def test_alphagenome_without_a_key_is_a_clean_error(monkeypatch, tmp_path):
    """No API key (or no client installed) is reported, not raised as a traceback."""
    monkeypatch.delenv("ALPHAGENOME_API_KEY", raising=False)
    result = run("alphagenome", "--bim", f"{DATASETS['synthetic'].plink_prefix}.bim",
                 "--out", tmp_path / "ag.tsv")  # fmt: skip
    assert result.exit_code == 1
    assert "ALPHAGENOME_API_KEY" in result.output or "pip install" in result.output


def test_alphagenome_scores_a_bim_with_progress(monkeypatch, tmp_path):
    """The CLI passes its options to the backend and reports progress per chunk."""
    from dietnet import alphagenome

    seen = {}

    def backend(**options):
        seen.update(options)
        return FakeBackend()

    monkeypatch.setattr(alphagenome, "AlphaGenomeAPI", backend)
    out = tmp_path / "ag.tsv"
    result = run("alphagenome", "--bim", f"{DATASETS['synthetic'].plink_prefix}.bim",
                 "--out", out, "--scorers", " DNASE, ,ATAC", "--sequence-length", "16KB",
                 "--chunk", 200, "--max-workers", 2)  # fmt: skip

    assert result.exit_code == 0, result.output
    assert seen == {"scorers": ["DNASE", "ATAC"], "sequence_length": "16KB", "max_workers": 2}
    assert "scored 200/500" in result.stderr and "scored 500/500" in result.stderr
    assert f"wrote {out}" in result.stdout
    assert len(pd.read_csv(out, sep="\t")) == 500


def test_alphagenome_reports_a_backend_value_error(monkeypatch, tmp_path):
    from dietnet import alphagenome

    def backend(**options):
        raise ValueError("unknown scorers ['NOPE']")

    monkeypatch.setattr(alphagenome, "AlphaGenomeAPI", backend)
    result = run("alphagenome", "--bim", f"{DATASETS['synthetic'].plink_prefix}.bim",
                 "--out", tmp_path / "ag.tsv", "--scorers", "NOPE")  # fmt: skip
    assert result.exit_code == 1
    assert "unknown scorers ['NOPE']" in result.output


# match and predict: package selection


@pytest.mark.parametrize("command", ["match", "predict"])
def test_no_package_matches_the_selection(command, trained_synthetic, tmp_path):
    extra = ["--out", tmp_path / "p.tsv"] if command == "predict" else ["--work-dir", tmp_path]
    result = run(command, "--model", trained_synthetic,
                 "--plink", DATASETS["synthetic"].plink_prefix, "--seeds", 999, *extra)  # fmt: skip
    assert result.exit_code == 2
    assert "no package matches --seeds/--folds" in result.output


def test_match_selects_by_fold(trained_synthetic, tmp_path):
    """--folds keeps only those folds; the matched file is written either way."""
    result = run("match", "--model", trained_synthetic,
                 "--plink", DATASETS["synthetic"].plink_prefix, "--work-dir", tmp_path,
                 "--folds", 1)  # fmt: skip
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("*.matched.h5"))
    # 500 SNPs is below the published advice, so the warning shows.
    assert "WARNING: only 500 of the model's 500 SNPs" in result.stderr


# check


@pytest.fixture(scope="module")
def regression_packages(trained_synthetic, tmp_path_factory):
    """The trained packages relabelled as regression, with numeric held-out predictions."""
    root = tmp_path_factory.mktemp("regression_packages")
    rng = np.random.default_rng(0)
    for i, pkg in enumerate(package.find(trained_synthetic)):
        dest = root / pkg.path.name / str(i)
        shutil.copytree(pkg.path, dest)
        meta = json.loads((dest / "metadata.json").read_text())
        meta.update(task="regression", classes=[])
        (dest / "metadata.json").write_text(json.dumps(meta))
        label = rng.normal(size=50)
        pd.DataFrame({"sample": [f"s{j}" for j in range(50)], "prediction": label + 0.1,
                      "label": label}).to_csv(dest / "test_predictions.tsv", sep="\t",
                                              index=False)  # fmt: skip
    return root


def test_check_regression_packages(regression_packages):
    result = run("check", "--packages", regression_packages)
    assert result.exit_code == 0, result.output
    assert "mean held-out r: 1.0000" in result.output
    assert "RMSE: 0.1000" in result.output


def test_check_regression_packages_refuse_min_accuracy(regression_packages):
    result = run("check", "--packages", regression_packages, "--min-accuracy", 0.5)
    assert result.exit_code == 2
    assert "--min-accuracy applies to classification packages" in result.output


@pytest.fixture
def labels(tmp_path):
    path = tmp_path / "labels.tsv"
    path.write_text("id\tpop\nA\tX\nB\tX\nC\tY\nD\tY\n")
    return path


def _predictions(tmp_path, rows):
    path = tmp_path / "predictions.tsv"
    pd.DataFrame(rows, columns=["sample", "prediction"]).to_csv(path, sep="\t", index=False)
    return path


def test_check_class_predictions_per_class(tmp_path, labels):
    preds = _predictions(tmp_path, [("A", "X"), ("B", "Y"), ("C", "Y"), ("D", "Y"), ("E", "X")])
    result = run("check", "--predictions", preds, "--labels", labels, "--min-accuracy", 0.75)
    assert result.exit_code == 0, result.output
    assert "accuracy: 0.7500 over 4 samples" in result.output


def test_check_class_predictions_below_min_accuracy_fail(tmp_path, labels):
    preds = _predictions(tmp_path, [("A", "Y"), ("B", "Y"), ("C", "Y"), ("D", "Y")])
    result = run("check", "--predictions", preds, "--labels", labels, "--min-accuracy", 0.6)
    assert result.exit_code == 1
    assert "FAIL: accuracy 0.5000 < 0.6" in result.stderr


def test_check_numeric_predictions_are_regression(tmp_path):
    truth = tmp_path / "trait.tsv"
    truth.write_text("id\ttrait\nA\t1\nB\t2\nC\t3\nD\t4\n")
    preds = _predictions(tmp_path, [("A", 1.5), ("B", 1.5), ("C", 3.5), ("D", 3.5)])
    result = run("check", "--predictions", preds, "--labels", truth)
    assert result.exit_code == 0, result.output
    assert "R^2: 0.8000, RMSE: 0.5000 over 4 samples" in result.output

    result = run("check", "--predictions", preds, "--labels", truth, "--min-accuracy", 0.5)
    assert result.exit_code == 2
    assert "--min-accuracy applies to class predictions" in result.output


def test_check_needs_samples_in_common(tmp_path, labels):
    preds = _predictions(tmp_path, [("Q", "X"), ("R", "Y")])
    result = run("check", "--predictions", preds, "--labels", labels)
    assert result.exit_code == 1
    assert "no sample in common between predictions and labels" in result.output


@pytest.mark.parametrize("given", ["nothing", "predictions only", "labels only"])
def test_check_needs_packages_or_predictions_with_labels(given, tmp_path, labels):
    args = {
        "nothing": [],
        "predictions only": ["--predictions", _predictions(tmp_path, [("A", "X")])],
        "labels only": ["--labels", labels],
    }[given]
    result = run("check", *args)
    assert result.exit_code == 2
    assert "give --packages, or --predictions with --labels" in result.output


# published, info, python -m


def test_published_from_an_unpacked_release_skips_the_download(monkeypatch, tmp_path):
    from dietnet import published

    def no_download(dest):
        raise AssertionError("fetch must not run when --release is given")

    calls = []
    monkeypatch.setattr(published, "fetch", no_download)
    monkeypatch.setattr(published, "import_training_release",
                        lambda release, out: calls.append((release, out)) or ["a", "b"])  # fmt: skip
    release = tmp_path / "Dietnet_training"
    release.mkdir()

    result = run("published", "--dest", tmp_path / "dest", "--release", release)

    assert result.exit_code == 0, result.output
    assert calls == [(release, tmp_path / "dest" / "packages")]
    assert f"2 published models in {tmp_path / 'dest' / 'packages'}" in result.output


def test_published_downloads_without_a_release(monkeypatch, tmp_path):
    from dietnet import published

    fetched = []
    monkeypatch.setattr(published, "fetch", lambda dest: fetched.append(dest) or dest / "rel")
    monkeypatch.setattr(published, "import_training_release", lambda release, out: [release])

    result = run("published", "--dest", tmp_path)

    assert result.exit_code == 0, result.output
    assert fetched == [tmp_path / "download"]
    assert "1 published models" in result.output


def test_info_fails_when_upstream_differs(monkeypatch):
    from dietnet import upstream

    monkeypatch.setattr(upstream, "verify", lambda: ["modified: train.py"])
    result = run("info")
    assert result.exit_code == 1
    assert "upstream files: modified: train.py" in result.output


def test_info_passes_on_the_pinned_upstream():
    result = run("info")
    assert result.exit_code == 0, result.output
    assert "identical to the pinned commit" in result.output


def test_module_runs_as_a_script(monkeypatch, capsys):
    """``python -m dietnet.cli`` runs the same group."""
    monkeypatch.setattr(sys, "argv", ["dietnet", "--help"])
    monkeypatch.delitem(sys.modules, "dietnet.cli")
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("dietnet.cli", run_name="__main__")
    assert exit_info.value.code == 0
    assert "create-dataset" in capsys.readouterr().out
