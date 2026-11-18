"""Data-parallel training (dietnet.parallel), on CPU processes.

``dietnet train --gpus N`` trains each model under torchrun, one process per
GPU, with the config's batch split across them. Its batches differ from
train.py's, so its weights do too; what it must give is a model of the same
quality, packaged like any other. Two gloo processes on the synthetic fixture
stand in for two GPUs.
"""

import re

import pytest
from support import DATASETS, dietnet, train_pipeline

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "inputs, processes", [("memmap", 2), ("packed", 2), ("packed", 1)]
)  # fmt: skip
def test_the_dietnet_engine_trains_a_model_as_good_as_train_py(tmp_path, inputs, processes):
    data = DATASETS["synthetic"]
    root = tmp_path / "ddp"
    flags = [f"--gpus={processes}", "--engine=dietnet", "--workers=1", f"--inputs={inputs}"]
    if inputs == "packed":  # train_pipeline builds the dataset; pack it, then train
        train_pipeline(root, data, folds="0", flags=["--inputs=memmap", "--loader=upstream"])
        dietnet("pack-genotypes", dataset=root / "dataset.h5")
        packages = root / "exp_packed"
        dietnet(
            "train",
            *flags,
            dataset=root / "dataset.h5",
            partition=root / "partition.npz",
            embedding=root / "embedding.npz",
            input_stats=root / "stats.npz",
            config=root / "config.yaml",
            exp_dir=packages,
            seeds=3,
            folds="0",
        )
        log_dir, packages = packages / "logs", packages / "packages"
    else:
        packages = train_pipeline(root, data, folds="0", flags=flags)
        log_dir = root / "exp" / "logs"

    (log,) = log_dir.glob("train_*.log")
    text = log.read_text()
    per = -(-32 // processes)
    assert (
        f"dietnet: data-parallel training on {processes} CPU processes; global batch 32 "
        f"({per} per process)" in text
    )
    assert f"{inputs} store" in text

    report = dietnet("check", packages=packages)
    accuracy = float(re.search(r"(\d\.\d+)", report.splitlines()[-1]).group(1))
    assert accuracy >= 0.9, report


def test_snp_sampling_trains_a_model_as_good(tmp_path):
    """--snp-sample 0.5 in place of per-sample input dropout: the same quality, and the
    package records how it was trained."""
    import json

    data = DATASETS["synthetic"]
    flags = ["--engine=dietnet", "--workers=1", "--inputs=memmap", "--snp-sample=0.5"]
    packages = train_pipeline(tmp_path / "ss", data, folds="0", flags=flags)

    report = dietnet("check", packages=packages)
    accuracy = float(re.search(r"(\d\.\d+)", report.splitlines()[-1]).group(1))
    assert accuracy >= 0.9, report
    (meta,) = packages.glob("seed_*/fold_*/metadata.json")
    assert json.loads(meta.read_text())["snp_sample"] == 0.5


@pytest.mark.parametrize("processes", [1, 2])
def test_chunked_reads_train_a_model_as_good(tmp_path, processes):
    """--chunk-rows and --snp-blocks: the same quality, on one and on several processes
    (which must see equal batch counts, or DDP hangs)."""
    data = DATASETS["synthetic"]
    flags = [
        f"--gpus={processes}", "--engine=dietnet", "--workers=1", "--inputs=memmap",
        "--snp-sample=0.5", "--chunk-rows=64", "--chunks-per-buffer=2", "--snp-blocks=8",
    ]  # fmt: skip
    packages = train_pipeline(tmp_path / "ch", data, folds="0", flags=flags)

    report = dietnet("check", packages=packages)
    accuracy = float(re.search(r"(\d\.\d+)", report.splitlines()[-1]).group(1))
    assert accuracy >= 0.9, report


def test_chunked_reads_from_a_packed_copy_cache_the_evaluation_rows(tmp_path):
    """The packed store with --chunk-rows: training buffers read through gaps, and the
    validation, recalibration and test rows are read once into memory."""
    data = DATASETS["synthetic"]
    root = tmp_path / "pk"
    train_pipeline(root, data, folds="0", flags=["--inputs=memmap", "--loader=upstream"])
    dietnet("pack-genotypes", dataset=root / "dataset.h5")
    dietnet(
        "train", "--engine=dietnet", "--workers=1", "--inputs=packed", "--snp-sample=0.5",
        "--chunk-rows=64", "--chunks-per-buffer=2", "--snp-blocks=8", "--bn-recalibrate=50",
        dataset=root / "dataset.h5", partition=root / "partition.npz",
        embedding=root / "embedding.npz", input_stats=root / "stats.npz",
        config=root / "config.yaml", exp_dir=root / "exp_packed", seeds=3, folds="0",
    )  # fmt: skip
    (log,) = (root / "exp_packed" / "logs").glob("train_*.log")
    assert log.read_text().count("evaluation rows cached in memory") == 2  # valid + test
    report = dietnet("check", packages=root / "exp_packed" / "packages")
    accuracy = float(re.search(r"(\d\.\d+)", report.splitlines()[-1]).group(1))
    assert accuracy >= 0.9, report


def test_free_snp_weights_kept_at_zero_train_the_plain_model(tmp_path):
    """--free-snp-weights with a penalty so strong that every free weight stays 0:
    bit for bit the model trained without them."""
    import torch

    data = DATASETS["synthetic"]
    flags = ["--engine=dietnet", "--workers=0", "--inputs=memmap"]
    plain = train_pipeline(tmp_path / "plain", data, folds="0", flags=flags)
    held = train_pipeline(
        tmp_path / "held", data, folds="0", flags=[*flags, "--free-snp-weights=1e12"]
    )

    def weights(packages):
        (model,) = packages.glob("seed_*/fold_*/model.pt")
        return torch.load(model, weights_only=False)["model_state_dict"]

    a, b = weights(plain), weights(held)
    assert not b.pop("free_snp_weights").any()
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_penalty_weights_of_zero_free_only_those_snps(tmp_path):
    """--snp-penalty-weights under a penalty that holds every other SNP at 0: only the SNPs
    weighted 0 get free weights, and the package records the weights file."""
    import json

    import numpy as np
    import torch

    from dietnet import package

    data = DATASETS["synthetic"]
    n_snps = sum(1 for _ in open(f"{data.plink_prefix}.bim"))
    weights = np.ones(n_snps)
    weights[: n_snps // 4] = 0
    np.save(path := tmp_path / "weights.npy", weights)
    flags = ["--engine=dietnet", "--workers=0", "--inputs=memmap", "--free-snp-weights=1e12",
             f"--snp-penalty-weights={path}"]  # fmt: skip
    (pkg,) = package.find(train_pipeline(tmp_path / "w", data, folds="0", flags=flags))
    delta = torch.load(pkg.path / "model.pt", weights_only=False)["model_state_dict"][
        "free_snp_weights"
    ]
    moved = delta.abs().sum(dim=0) > 0
    assert moved.any() and not moved[n_snps // 4 :].any()
    meta = json.loads((pkg.path / "metadata.json").read_text())["free_snp_weights"]
    assert meta["penalty_weights"]["path"] == str(path.resolve())
    assert meta["penalty_weights"]["mean"] == weights.mean()


def test_a_hybrid_package_predicts_what_training_reported(tmp_path):
    """Free weights that do change the model: the package carries them, loads them
    strictly, and predict gives the held-out predictions training wrote."""
    import json

    import pandas as pd

    from dietnet import package

    data = DATASETS["synthetic"]
    root = tmp_path / "hy"
    flags = ["--engine=dietnet", "--workers=0", "--inputs=memmap", "--free-snp-weights=1e-4"]
    packages = train_pipeline(root, data, folds="0", flags=flags)
    (pkg,) = package.find(packages)
    meta = json.loads((pkg.path / "metadata.json").read_text())
    assert meta["format_version"] == 3 and meta["free_snp_weights"]["lambda"] == 1e-4
    import torch

    state = torch.load(pkg.path / "model.pt", weights_only=False)["model_state_dict"]
    assert state["free_snp_weights"].abs().sum() > 0  # the free weights do change the model

    (log,) = (root / "exp" / "logs").glob("train_*.log")
    assert "free weights on" in log.read_text()
    report = dietnet("check", packages=packages)
    accuracy = float(re.search(r"(\d\.\d+)", report.splitlines()[-1]).group(1))
    assert accuracy >= 0.9, report

    out = tmp_path / "predictions.tsv"
    dietnet("predict", model=packages, plink=data.plink_prefix, out=out)
    predicted = pd.read_csv(out, sep="\t", dtype={"sample": str})
    held = pd.read_csv(pkg.path / "test_predictions.tsv", sep="\t", dtype={"sample": str})
    merged = held.merge(predicted, on="sample", suffixes=("_train", "_predict"))
    assert len(merged) == len(held) > 0
    assert (
        merged["prediction_train"].astype(str) == merged["prediction_predict"].astype(str)
    ).all()


def test_recalibrated_batchnorm_trains_a_model_as_good(tmp_path):
    """--bn-recalibrate: the same quality, with statistics recomputed before validation."""
    data = DATASETS["synthetic"]
    flags = ["--engine=dietnet", "--workers=1", "--inputs=memmap", "--bn-recalibrate=100"]
    packages = train_pipeline(tmp_path / "bn", data, folds="0", flags=flags)

    (log,) = (tmp_path / "bn" / "exp" / "logs").glob("train_*.log")
    assert "batch-norm statistics recomputed on 100 training samples" in log.read_text()
    report = dietnet("check", packages=packages)
    accuracy = float(re.search(r"(\d\.\d+)", report.splitlines()[-1]).group(1))
    assert accuracy >= 0.9, report
