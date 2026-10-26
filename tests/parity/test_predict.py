"""dietnet predict gives the scores upstream's own scripts give.

``dietnet predict`` replaces upstream's inference scripts for scale (one matched
copy of the cohort, memory-mapped, shared by every model) but runs upstream's
model class and inference step. These tests run upstream's scripts from a full
checkout of the pinned commit and require the same scores:

* test_independent_dataset.py, upstream's generic "apply a trained model to
  another dataset", on the cohort dietnet matched;
* the same script doing its *own* SNP matching on a cohort missing 30% of the
  model's SNPs -- so the missing-SNP handling and D/(D-|C|) scale are
  upstream's too, not only the arithmetic after them;
* Dietnet_inference/inference.py, the published inference path, with the
  published models (slow; downloads them from Zenodo).
"""

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from pyplink import PyPlink

from dietnet import dataset as ds
from dietnet import package
from dietnet import predict as prediction


def run_upstream(checkout: Path, relative: str, args: list):
    script = checkout / relative
    result = subprocess.run(
        [sys.executable, str(script), *[str(a) for a in args]],
        cwd=script.parent,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def upstream_independent_test(checkout, pkg, test_h5, out_dir, matched=True):
    """Scores from upstream's test_independent_dataset.py for one package."""
    args = [
        "--model-params", pkg.path / "model.pt",
        "--train-dataset", pkg.path / "train_header.h5",
        "--test-path", out_dir, "--test-name", pkg.name,
        "--test-dataset", test_h5,
        "--config", pkg.path / "config.yaml",
        "--task", "classification", "--model", "Dietnet",
        "--input-features-stats", pkg.path / "input_features_stats.npz",
        "--embedding", pkg.path / "embedding.npz",
        "--which-fold", 0,
    ]  # fmt: skip
    if matched:
        args += ["--matched-test-dataset", test_h5]
    if pkg.normalize:
        args.append("--normalize")
    run_upstream(checkout, "Dietnet_train_and_evaluate/test_independent_dataset.py", args)
    return np.load(Path(out_dir) / f"{pkg.name}_results.npz")["scores"]


def test_scores_match_upstream_test_independent_dataset(
    tmp_path, trained_synthetic, synthetic, upstream_checkout
):
    packages = package.find(trained_synthetic)
    matched = prediction.matched_dataset(packages, synthetic.plink_prefix, tmp_path)
    ours = prediction.predict(packages, matched, device="cpu")

    for i, pkg in enumerate(packages):
        theirs = upstream_independent_test(upstream_checkout, pkg, matched, tmp_path)
        np.testing.assert_allclose(ours.scores[i], theirs, rtol=1e-5, atol=1e-6)


def test_missing_snps_are_handled_as_upstream_handles_them(
    tmp_path, trained_synthetic, synthetic, upstream_checkout
):
    """30% of the model's SNPs absent: upstream matches by name and scales; so do we."""
    packages = package.find(trained_synthetic)[:1]
    with h5py.File(synthetic.upstream_hdf5) as f:
        genotypes = np.array(f["inputs"])
    keep = np.sort(np.random.default_rng(0).choice(genotypes.shape[1], 350, replace=False))

    cohort = tmp_path / "partial"
    with PyPlink(str(cohort), "w") as bed:
        for j in keep:
            bed.write_genotypes(genotypes[:, j])
    bim = pd.read_csv(f"{synthetic.plink_prefix}.bim", sep="\t", header=None).iloc[keep]
    bim.to_csv(f"{cohort}.bim", sep="\t", header=False, index=False)
    Path(f"{cohort}.fam").write_text(Path(f"{synthetic.plink_prefix}.fam").read_text())

    matched = prediction.matched_dataset(packages, cohort, tmp_path / "work")
    ours = prediction.predict(packages, matched, device="cpu")
    assert ours.scale == pytest.approx(500 / 350)

    # Upstream's route: the cohort as its own dataset, matched by upstream's code.
    cohort_h5 = ds.build(cohort, tmp_path / "partial.hdf5", class_labels=synthetic.labels)
    theirs = upstream_independent_test(
        upstream_checkout, packages[0], cohort_h5, tmp_path, matched=False
    )
    np.testing.assert_allclose(ours.scores[0], theirs, rtol=1e-5, atol=1e-6)


@pytest.mark.slow
@pytest.mark.network
def test_published_models_match_upstream_inference(
    tmp_path, published_packages, kgp_mini, upstream_checkout
):
    """The 15 published models on kgp_mini, against Dietnet_inference/inference.py."""
    matched = prediction.matched_dataset(published_packages, kgp_mini.plink_prefix, tmp_path)
    ours = prediction.predict(published_packages, matched, device="cpu")

    for i, pkg in enumerate(published_packages):
        stats = np.load(pkg.path / "input_features_stats.npz")
        norm = tmp_path / f"{pkg.name}_norm.npz"
        np.savez(norm, mean=stats["means_by_fold"][0], std=stats["sd_by_fold"][0])
        out = tmp_path / f"{pkg.name}_inferred.npz"
        run_upstream(upstream_checkout, "Dietnet_inference/inference.py", [
            "--test-h5", matched, "--model-param", pkg.path / "model.pt",
            "--snps-emb", pkg.path / "embedding.npz", "--norm-stats", norm, "--out", out,
        ])  # fmt: skip
        np.testing.assert_allclose(ours.scores[i], np.load(out)["scores"], rtol=1e-5, atol=1e-6)
