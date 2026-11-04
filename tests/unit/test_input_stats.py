"""compute-input-stats --engine dietnet: upstream's statistics, streamed."""

import h5py
import numpy as np
import pytest
from support import dietnet

from dietnet import dataset as ds
from dietnet import genotypes, input_stats


@pytest.fixture
def prepared(tmp_path, kgp_mini):
    """kgp_mini as a dataset with some genotypes set missing, and a partition."""
    h5 = ds.build(kgp_mini.plink_prefix, tmp_path / "d.h5", class_labels=kgp_mini.labels)
    with h5py.File(h5, "r+") as f:
        x = np.array(f["inputs"])
        rng = np.random.default_rng(0)
        x[rng.random(x.shape) < 0.05] = -1  # 5% missing, everywhere
        x[:, 3] = -1  # and one SNP missing in all but two samples
        x[:2, 3] = [0, 2]
        f["inputs"][...] = x
    dietnet("partition", dataset=h5, out=tmp_path / "p.npz")
    return h5, tmp_path / "p.npz"


@pytest.mark.parametrize("inputs", ["ram", "memmap", "packed"])
def test_streamed_statistics_are_upstreams(prepared, tmp_path, inputs):
    """Means exactly upstream's; SDs to floating-point rounding, in every store and
    with blocks that split the folds' rows unevenly."""
    h5, partition = prepared
    if inputs == "packed":
        genotypes.pack(h5)
    dietnet("compute-input-stats", dataset=h5, partition=partition, out=tmp_path / "up.npz")
    upstream = np.load(tmp_path / "up.npz")

    ours = input_stats.compute(h5, partition, inputs=inputs, block=37)

    assert ours["means_by_fold"].shape == upstream["means_by_fold"].shape
    np.testing.assert_array_equal(ours["means_by_fold"], upstream["means_by_fold"])
    np.testing.assert_allclose(ours["sd_by_fold"], upstream["sd_by_fold"], rtol=1e-12, atol=0)


def test_the_cli_writes_what_train_reads(prepared, tmp_path):
    h5, partition = prepared
    dietnet("compute-input-stats", dataset=h5, partition=partition, out=tmp_path / "up.npz")
    dietnet(
        "compute-input-stats", dataset=h5, partition=partition, out=tmp_path / "ours.npz",
        engine="dietnet",
    )  # fmt: skip
    up, ours = np.load(tmp_path / "up.npz"), np.load(tmp_path / "ours.npz")
    assert sorted(ours.files) == sorted(up.files)
    for key in up.files:
        assert ours[key].dtype == up[key].dtype and ours[key].shape == up[key].shape
