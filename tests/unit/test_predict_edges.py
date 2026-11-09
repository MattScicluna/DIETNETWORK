"""Edge cases of prediction: packages that do not belong together, and upstream's order."""

import json
import shutil

import h5py
import numpy as np
import pytest
import torch
from support import DATASETS

from dietnet import package
from dietnet import predict as prediction
from dietnet import upstream


@pytest.fixture(scope="module")
def packages(trained_synthetic):
    return package.find(trained_synthetic)


@pytest.fixture(scope="module")
def matched(packages, tmp_path_factory):
    work = tmp_path_factory.mktemp("matched")
    return prediction.matched_dataset(packages, DATASETS["synthetic"].plink_prefix, work)


def _copy(pkg, dest, **metadata):
    """A copy of ``pkg`` at ``dest`` with some metadata changed."""
    shutil.copytree(pkg.path, dest)
    meta = json.loads((dest / "metadata.json").read_text())
    meta.update(metadata)
    (dest / "metadata.json").write_text(json.dumps(meta))
    return package.Package(dest)


def test_device_for_names_and_defaults():
    assert prediction.device_for("cpu") == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert prediction.device_for(None).type == expected


def test_packages_on_different_snps_are_refused(packages, tmp_path):
    other = _copy(packages[0], tmp_path / "other")
    with h5py.File(other.path / "train_header.h5", "r+") as f:
        names = np.array(f["snp_names"])
        names[0] = b"renamed_A"
        del f["snp_names"]
        f.create_dataset("snp_names", data=names)

    with pytest.raises(ValueError, match="trained on different SNPs than"):
        prediction.matched_dataset(
            [packages[0], other], DATASETS["synthetic"].plink_prefix, tmp_path / "work"
        )
    assert not (tmp_path / "work").exists()


def test_the_matched_cohort_is_reused(packages, matched):
    """A second call with the same inputs finds the file and does not rewrite it."""
    before = matched.stat().st_mtime_ns
    again = prediction.matched_dataset(packages, DATASETS["synthetic"].plink_prefix, matched.parent)
    assert again == matched and matched.stat().st_mtime_ns == before


def test_packages_that_mix_tasks_are_refused(packages, matched, tmp_path):
    regression = _copy(packages[0], tmp_path / "regression", task="regression")
    with pytest.raises(ValueError, match=r"packages mix tasks: \['classification', 'regression'\]"):
        prediction.predict([packages[0], regression], matched, device="cpu")


def test_packages_with_different_classes_are_refused(packages, matched, tmp_path):
    classes = list(reversed(packages[0].classes))
    other = _copy(packages[0], tmp_path / "other", classes=classes)
    with pytest.raises(ValueError, match="predicts different classes than"):
        prediction.predict([packages[0], other], matched, device="cpu")


def test_samples_out_of_order_from_upstream_are_an_error(packages, matched, monkeypatch):
    """predict relies on upstream returning each batch's samples in input order."""
    utils = upstream.modules().mainloop_utils
    real = utils.indep_test_step

    def reversed_step(*args):
        results = real(*args)
        results["samples"] = results["samples"][::-1].copy()
        return results

    monkeypatch.setattr(utils, "indep_test_step", reversed_step)
    with pytest.raises(RuntimeError, match="returned samples out of order"):
        prediction.predict(packages[:1], matched, batch_size=64, device="cpu")


def test_batch_size_does_not_change_scores(packages, matched):
    """Scores are the same whether the cohort is read in one batch or many uneven ones."""
    whole = prediction.predict(packages, matched, batch_size=10_000, device="cpu")
    pieces = prediction.predict(packages, matched, batch_size=7, device="cpu")

    np.testing.assert_allclose(pieces.scores, whole.scores, rtol=1e-5, atol=1e-6)
    assert whole.scores.shape == (len(packages), len(whole.samples), len(whole.classes))
    assert whole.model_names == [p.name for p in packages]


def test_regression_table_averages_models():
    result = prediction.Result(
        samples=np.array(["a", "b"]),
        model_names=["m1", "m2"],
        classes=[],
        scores=np.array([[1.0, 2.0], [3.0, 2.0]]),
        alignment={},
        scale=1.0,
    )
    table = result.table()
    assert list(table.columns) == ["sample", "prediction", "sd_across_models"]
    assert table["prediction"].tolist() == [2.0, 2.0]
    assert table["sd_across_models"].tolist() == [1.0, 0.0]


def test_classification_table_reports_agreement():
    """The call is the mean's argmax; agreement is the share of models that voted for it."""
    scores = np.array([[[0.9, 0.1], [0.2, 0.8]], [[0.4, 0.6], [0.3, 0.7]]])
    result = prediction.Result(
        samples=np.array(["a", "b"]),
        model_names=["m1", "m2"],
        classes=["X", "Y"],
        scores=scores,
        alignment={},
        scale=1.0,
    )
    table = result.table()
    assert table["prediction"].tolist() == ["X", "Y"]
    np.testing.assert_allclose(table["probability"], [0.65, 0.75])
    assert table["model_agreement"].tolist() == [0.5, 1.0]
