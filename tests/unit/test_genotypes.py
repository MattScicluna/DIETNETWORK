"""Genotype stores: where training reads its rows from (dietnet.genotypes)."""

import h5py
import numpy as np
import pytest

from dietnet import dataset as ds
from dietnet import genotypes


@pytest.fixture
def dataset(tmp_path, kgp_mini):
    return ds.build(kgp_mini.plink_prefix, tmp_path / "d.h5", class_labels=kgp_mini.labels)


@pytest.fixture
def chunked(tmp_path, dataset):
    """The same dataset with ``inputs`` chunked, as h5py writes it by default
    with compression; such a file cannot be memory-mapped."""
    out = tmp_path / "chunked.h5"
    with h5py.File(dataset, "r") as src, h5py.File(out, "w") as dst:
        for key in src:
            if key == "inputs":
                dst.create_dataset(key, data=src[key][:], chunks=True, compression="gzip")
            else:
                src.copy(key, dst)
    return out


def test_a_memory_map_gives_exactly_the_rows_upstream_copies(dataset):
    ram = genotypes.open_store(dataset, "ram")
    mapped = genotypes.open_store(dataset, "memmap")

    rows = np.array([5, 0, 479, 5, 200])
    assert mapped.shape == ram.shape and mapped.dtype == np.int8
    np.testing.assert_array_equal(mapped[rows], ram[rows])
    np.testing.assert_array_equal(mapped[17], ram[17])


def test_auto_keeps_small_datasets_in_ram_and_maps_large_ones(dataset, monkeypatch):
    assert genotypes.choose(dataset, "auto") == "ram"

    monkeypatch.setattr(genotypes, "AUTO_MEMMAP_ABOVE_BYTES", 0)
    assert genotypes.choose(dataset, "auto") == "memmap"


def test_a_dataset_that_cannot_be_mapped(chunked, monkeypatch):
    monkeypatch.setattr(genotypes, "AUTO_MEMMAP_ABOVE_BYTES", 0)

    assert genotypes.choose(chunked, "auto") == "ram"
    with pytest.raises(ValueError, match="cannot memory-map"):
        genotypes.choose(chunked, "memmap")


def test_an_explicit_mode_is_kept_and_an_unknown_one_refused(dataset):
    assert genotypes.choose(dataset, "ram") == "ram"
    assert genotypes.choose(dataset, "memmap") == "memmap"
    with pytest.raises(ValueError, match="inputs mode"):
        genotypes.choose(dataset, "disk")
