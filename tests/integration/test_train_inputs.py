"""Training from a memory map or a packed copy is training: the same model, bit for bit.

``dietnet train --inputs memmap|packed`` replaces upstream's copy of every
genotype in RAM with a genotype store (dietnet.genotypes, via the launcher).
The rows, their values and the batch order are unchanged, so the weights must
be identical, not merely close.
"""

import pytest
import torch
from support import DATASETS, SEED, dietnet, train_pipeline

pytestmark = pytest.mark.integration


def _weights(packages):
    (model,) = sorted(packages.rglob("model.pt"))
    return torch.load(model, map_location="cpu", weights_only=False)["model_state_dict"]


def _store_line(root):
    (log,) = (root / "exp" / "logs").glob("train_*.log")
    return [line for line in log.read_text().splitlines() if line.startswith("dietnet: train.py")]


def test_a_memory_mapped_run_trains_the_same_model(tmp_path):
    data = DATASETS["synthetic"]
    ram = _weights(train_pipeline(tmp_path / "ram", data, folds="0", flags=["--inputs=ram"]))
    mapped = _weights(
        train_pipeline(tmp_path / "memmap", data, folds="0", flags=["--inputs=memmap"])
    )

    # The setting reached train.py's process, and only in the memmap run.
    assert _store_line(tmp_path / "ram") == []
    assert _store_line(tmp_path / "memmap") == [
        "dietnet: train.py reads genotypes from the memmap store"
    ]
    assert ram.keys() == mapped.keys()
    differ = [name for name in ram if not torch.equal(ram[name], mapped[name])]
    assert differ == []

    # The same inputs again, from the 2-bit copy.
    root = tmp_path / "ram"
    dietnet("pack-genotypes", dataset=root / "dataset.h5")
    dietnet(
        "train",
        dataset=root / "dataset.h5",
        partition=root / "partition.npz",
        embedding=root / "embedding.npz",
        input_stats=root / "stats.npz",
        config=root / "config.yaml",
        exp_dir=tmp_path / "packed" / "exp",
        seeds=SEED,
        folds="0",
        inputs="packed",
    )
    packed = _weights(tmp_path / "packed" / "exp" / "packages")
    assert _store_line(tmp_path / "packed") == [
        "dietnet: train.py reads genotypes from the packed store"
    ]
    differ = [name for name in ram if not torch.equal(ram[name], packed[name])]
    assert differ == []


def test_the_launcher_swaps_the_copy_for_the_store(tmp_path, monkeypatch):
    """Guards the test above against passing because nothing was swapped.

    Replays the lines of train.py that load the genotypes, against the class the
    launcher installs: the np.array copy must see an empty placeholder, and
    data_x must end up the memory map.
    """
    import h5py
    import numpy as np

    from dietnet import _launch
    from dietnet import dataset as ds
    from dietnet import upstream

    data = DATASETS["synthetic"]
    dataset = ds.build(data.plink_prefix, tmp_path / "d.h5", class_labels=data.labels)
    monkeypatch.syspath_prepend(str(upstream.script("train.py").parent))
    import helpers.dataset_utils as du

    monkeypatch.setattr(du, "FoldDataset", du.FoldDataset)  # restored after the test
    _launch._store_train_inputs("memmap")

    # train.py lines 189-198, verbatim but for the module alias.
    du.FoldDataset.dataset_file = str(dataset)
    du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, "r")
    copied = np.array(du.FoldDataset.f["inputs"], dtype=np.int8)
    du.FoldDataset.data_x = copied

    assert copied.size == 0
    assert isinstance(du.FoldDataset.data_x, np.memmap)
    assert du.FoldDataset.data_x.shape == ds.memmap_inputs(dataset).shape
    assert du.FoldDataset.f["samples"].shape[0] == du.FoldDataset.data_x.shape[0]
