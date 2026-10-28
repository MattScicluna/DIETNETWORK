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
