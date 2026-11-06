"""rescale_eval.py: validation rescaling of Diet Network packages, on a known answer."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[2] / "experiments" / "ukbb_traits" / "benchmark" / "rescale_eval.py"


@pytest.fixture
def run(tmp_path):
    """Two folds whose predictions have the wrong slope and offset, with their inputs."""
    rng = np.random.default_rng(0)
    n = 3000
    ids = np.array([str(1000 + i) for i in range(n)])
    y = rng.normal(size=n)
    pd.DataFrame({"FID": ids, "IID": ids, "height": y}).to_csv(
        tmp_path / "pheno.tsv", sep="\t", index=False
    )
    scores = []
    for fold in (0, 1):
        perm = rng.permutation(n)
        train, valid, test = perm[:1000], perm[1000:2000], perm[2000:]
        score = 0.5 * (0.6 * y + 0.8 * rng.normal(size=n)) + 0.3
        scores.append(score)
        samples = np.empty((1, 3), dtype=object)
        samples[0, 0], samples[0, 1], samples[0, 2] = ids[train], ids[valid], ids[test]
        np.savez(tmp_path / f"part{fold}.npz", folds_samples=samples)
        package = tmp_path / "packages" / "seed_78" / f"fold_{fold}"
        package.mkdir(parents=True)
        pd.DataFrame({"sample": ids[test], "prediction": score[test], "label": y[test]}).to_csv(
            package / "test_predictions.tsv", sep="\t", index=False
        )
    # The cohort's order is not the partition's.
    order = rng.permutation(n)
    np.savez(
        tmp_path / "scores.npz",
        samples=ids[order],
        models=np.array(["seed78_fold0", "seed78_fold1"]),
        scores=np.array(scores)[:, order],
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--scores",
        str(tmp_path / "scores.npz"),
        "--packages",
        str(tmp_path / "packages"),
        "--partition",
        str(tmp_path / "part{fold}.npz"),
        "--pheno",
        str(tmp_path / "pheno.tsv"),
        "--label",
        "height",
        "--out",
        str(tmp_path / "out.json"),
    ]
    env = {"PYTHONPATH": str(SCRIPT.parent)}
    return tmp_path, command, env


def test_rescaling_recovers_r_squared_and_matches_training(run):
    tmp_path, command, env = run
    subprocess.run(command, check=True, env=env, capture_output=True)
    out = json.loads((tmp_path / "out.json").read_text())
    assert [row["fold"] for row in out["folds"]] == [0, 1]
    for row in out["folds"]:
        assert row["test_r"] == pytest.approx(0.6, abs=0.05)
        # Rescaled on validation, R^2 is close to r^2; raw, the bad scale costs a lot.
        assert row["test_r2"] == pytest.approx(row["test_r"] ** 2, abs=0.01)
        assert row["raw_test_r2"] < row["test_r2"] - 0.05
        assert row["max_abs_diff_saved_prediction"] < 1e-9
        assert row["valid_n"] == 1000


def test_fails_when_predictions_differ_from_training(run):
    tmp_path, command, env = run
    path = tmp_path / "packages" / "seed_78" / "fold_1" / "test_predictions.tsv"
    saved = pd.read_csv(path, sep="\t")
    saved["prediction"] += 0.1
    saved.to_csv(path, sep="\t", index=False)
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "differ from training" in result.stderr
