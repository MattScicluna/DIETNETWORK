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
