"""However train.py is fed, it trains the same model, bit for bit.

``dietnet train`` changes how upstream's train.py gets its genotypes, never
what it computes (dietnet._launch):

- ``--inputs memmap|packed``: from a genotype store (dietnet.genotypes) instead
  of train.py's copy of every genotype in RAM;
- ``--loader fast``: whole batches per read, from loader workers, with missing
  values filled in one operation instead of a loop over the batch.

The batches, their order and their values are upstream's, so the weights must
equal those of train.py left entirely alone (``--inputs ram --loader
upstream``): identical, not close.
"""

import pytest
import torch
from support import DATASETS, SEED, dietnet, train_pipeline

pytestmark = pytest.mark.integration

UPSTREAM = ["--inputs=ram", "--loader=upstream"]


def _weights(packages):
    (model,) = sorted(packages.rglob("model.pt"))
    return torch.load(model, map_location="cpu", weights_only=False)["model_state_dict"]


def _launcher_lines(root):
    (log,) = (root / "exp" / "logs").glob("train_*.log")
    return [line for line in log.read_text().splitlines() if line.startswith("dietnet: ")]


def _retrain(root, out, **options):
    """Train fold 0 again on the inputs train_pipeline built under ``root``."""
    dietnet(
        "train",
        dataset=root / "dataset.h5",
        partition=root / "partition.npz",
        embedding=root / "embedding.npz",
        input_stats=root / "stats.npz",
        config=root / "config.yaml",
        exp_dir=out / "exp",
        seeds=SEED,
        folds="0",
        **options,
    )
    return _weights(out / "exp" / "packages")


def _differ(a, b):
    assert a.keys() == b.keys()
    return [name for name in a if not torch.equal(a[name], b[name])]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Fold 0 of the synthetic fixture, trained by train.py left entirely alone."""
    root = tmp_path_factory.mktemp("upstream")
    weights = _weights(train_pipeline(root, DATASETS["synthetic"], folds="0", flags=UPSTREAM))
    assert _launcher_lines(root) == []  # nothing was swapped
    dietnet("pack-genotypes", dataset=root / "dataset.h5")
    return root, weights


@pytest.mark.parametrize(
    "inputs, loader, workers, lines",
    [
        ("memmap", "upstream", None, ["dietnet: train.py reads genotypes from the memmap store"]),
        ("packed", "upstream", None, ["dietnet: train.py reads genotypes from the packed store"]),
        ("ram", "fast", 0, ["dietnet: loader reads whole batches with 0 workers"]),
        ("memmap", "fast", 2, ["dietnet: train.py reads genotypes from the memmap store",
                               "dietnet: loader reads whole batches with 2 workers"]),
        ("packed", "fast", 2, ["dietnet: train.py reads genotypes from the packed store",
                               "dietnet: loader reads whole batches with 2 workers"]),
    ],
)  # fmt: skip
def test_every_way_of_feeding_train_py_trains_the_same_model(
    reference, tmp_path, inputs, loader, workers, lines
):
    root, expected = reference
    options = {"inputs": inputs, "loader": loader}
    if workers is not None:
        options["workers"] = workers

    weights = _retrain(root, tmp_path, **options)

    # The settings reached train.py's process...
    got = [line.split(";")[0] for line in _launcher_lines(tmp_path)]
    assert got == lines
    # ...and changed nothing it computed.
    assert _differ(expected, weights) == []


def test_the_launcher_swaps_the_copy_for_the_store(tmp_path, monkeypatch):
    """Guards the tests above against passing because nothing was swapped.

    Replays the lines of train.py that load the genotypes, against the class the
    launcher installs: the np.array copy must see an empty placeholder, and
    data_x must end up the store.
    """
    import h5py
    import numpy as np

    from dietnet import _launch
    from dietnet import dataset as ds
    from dietnet import genotypes, upstream

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
    assert isinstance(du.FoldDataset.data_x, genotypes.MemmapStore)
    assert du.FoldDataset.data_x.shape == ds.memmap_inputs(dataset).shape
    assert du.FoldDataset.f["samples"].shape[0] == du.FoldDataset.data_x.shape[0]
