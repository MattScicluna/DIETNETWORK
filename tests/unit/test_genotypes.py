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


def test_the_map_is_read_only_and_batches_are_writable_copies(dataset):
    # Copy-on-write counts against the node's memory: the kernel refuses a mapping larger
    # than its RAM, which a biobank's dataset is.
    x = ds.memmap_inputs(dataset)
    assert x.mode == "r" and not x.flags.writeable
    assert genotypes.open_store(dataset, "memmap")[np.array([0, 3])].flags.writeable


def test_auto_keeps_small_datasets_in_ram_and_maps_large_ones(dataset, monkeypatch):
    assert genotypes.choose(dataset, "auto") == "ram"

    monkeypatch.setattr(genotypes, "AUTO_MEMMAP_ABOVE_BYTES", 0)
    assert genotypes.choose(dataset, "auto") == "memmap"


def test_a_dataset_that_cannot_be_mapped(chunked, monkeypatch):
    monkeypatch.setattr(genotypes, "AUTO_MEMMAP_ABOVE_BYTES", 0)

    assert genotypes.choose(chunked, "auto") == "ram"
    with pytest.raises(ValueError, match="cannot memory-map"):
        genotypes.choose(chunked, "memmap")


@pytest.mark.parametrize("n_snps", [1, 4, 7, 2001])
def test_two_bits_hold_every_genotype_exactly(n_snps):
    rng = np.random.default_rng(n_snps)
    rows = rng.integers(-1, 3, size=(13, n_snps)).astype(np.int8)  # -1 is missing

    packed = genotypes.encode(rows)

    assert packed.dtype == np.uint8 and packed.shape == (13, -(-n_snps // 4))
    np.testing.assert_array_equal(genotypes.decode(packed, n_snps), rows)


def test_a_packed_store_gives_exactly_the_rows_upstream_copies(dataset):
    genotypes.pack(dataset)
    ram = genotypes.open_store(dataset, "ram")
    packed = genotypes.open_store(dataset, "packed")

    assert packed.shape == ram.shape and len(packed) == len(ram)
    for rows in (np.array([5, 0, 479, 5, 200]), np.arange(len(ram)), [3]):
        np.testing.assert_array_equal(packed[rows], ram[rows])
    for row in (0, np.int64(17), len(ram) - 1):
        np.testing.assert_array_equal(packed[row], ram[row])
    assert genotypes.packed_path(dataset).stat().st_size < ram.nbytes / 3


def test_file_reads_give_the_memory_maps_bytes(dataset):
    """PackedReader (one read per run of consecutive rows) == indexing the memory map,
    for runs, gaps, repeats, any order, single rows and the last row; and it pickles
    without its open file, for loader workers."""
    import pickle

    genotypes.pack(dataset)
    mapped = np.load(genotypes.packed_path(dataset), mmap_mode="r")
    reader = genotypes.PackedReader.of(genotypes.packed_path(dataset))
    n = len(mapped)
    for rows in (
        np.arange(n), np.array([0, 1, 2, 7, 8, 30]), np.array([9, 3, 3, 4, n - 1]),
        np.array([], dtype=np.int64), np.sort(np.random.default_rng(0).choice(n, 50, False)),
    ):  # fmt: skip
        np.testing.assert_array_equal(reader[rows], mapped[rows])
    np.testing.assert_array_equal(reader[n - 1], mapped[n - 1])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(reader))[[4, 5]], mapped[[4, 5]])
    with pytest.raises(IndexError):
        reader[[n]]


def test_reads_through_gaps_and_cached_rows_give_the_same_bytes(dataset):
    genotypes.pack(dataset)
    mapped = np.load(genotypes.packed_path(dataset), mmap_mode="r")
    reader = genotypes.PackedReader.of(genotypes.packed_path(dataset))
    n = len(mapped)
    rng = np.random.default_rng(1)
    for density in (0.2, 0.6, 1.0):
        rows = np.flatnonzero(rng.random(n) < density)
        for gap in (0, 1, 8, 1000):
            np.testing.assert_array_equal(reader.read(rows, max_gap=gap), mapped[rows])
    np.testing.assert_array_equal(reader.read([5, 3, 3, 9], max_gap=8), mapped[[5, 3, 3, 9]])

    rows = np.flatnonzero(rng.random(n) < 0.3)
    cache = genotypes.CachedRows(lambda r: reader.read(r, max_gap=4), rows[::-1])
    some = rng.permutation(rows)[:20]
    np.testing.assert_array_equal(cache[some], mapped[some])
    absent = np.setdiff1d(np.arange(n), rows)[:1]
    with pytest.raises(KeyError):
        cache[absent]


def test_auto_prefers_an_up_to_date_pack(dataset):
    assert genotypes.choose(dataset, "auto") == "ram"
    with pytest.raises(ValueError, match="pack-genotypes"):
        genotypes.choose(dataset, "packed")

    genotypes.pack(dataset)

    assert genotypes.choose(dataset, "auto") == "packed"
    assert genotypes.choose(dataset, "memmap") == "memmap"


def test_a_pack_of_a_dataset_that_changed_since_is_refused(dataset):
    genotypes.pack(dataset)
    with h5py.File(dataset, "a") as f:  # e.g. rebuilt with other samples
        f.attrs["rebuilt"] = 1

    assert not genotypes.is_packed(dataset)
    assert genotypes.choose(dataset, "auto") == "ram"
    with pytest.raises(ValueError, match="up-to-date"):
        genotypes.open_store(dataset, "packed")


def test_an_explicit_mode_is_kept_and_an_unknown_one_refused(dataset):
    assert genotypes.choose(dataset, "ram") == "ram"
    assert genotypes.choose(dataset, "memmap") == "memmap"
    with pytest.raises(ValueError, match="inputs mode"):
        genotypes.choose(dataset, "disk")


@pytest.fixture
def packed_only(tmp_path, kgp_mini):
    return ds.build(
        kgp_mini.plink_prefix, tmp_path / "p.h5", class_labels=kgp_mini.labels, packed_only=True
    )


def test_a_packed_only_dataset_is_the_pack_of_the_full_one(dataset, packed_only):
    """create-dataset --packed-only: the same file minus the int8 values, and an up-to-date
    2-bit copy with exactly the full dataset's pack."""
    genotypes.pack(dataset)
    assert genotypes.is_packed(packed_only)
    assert np.array_equal(
        np.load(genotypes.packed_path(packed_only)), np.load(genotypes.packed_path(dataset))
    )
    with h5py.File(dataset, "r") as full, h5py.File(packed_only, "r") as f:
        assert f["inputs"].shape == full["inputs"].shape
        assert f["inputs"].id.get_storage_size() == 0  # nothing stored
        assert not f.attrs["inputs_stored"] and full.attrs["inputs_stored"]
        for key in ("snp_names", "samples", "class_labels", "snps/pos"):
            assert np.array_equal(f[key][()], full[key][()]), key
    assert genotypes.inputs_stored(dataset) and not genotypes.inputs_stored(packed_only)
    rows = np.array([3, 0, 7])
    assert np.array_equal(
        genotypes.open_store(packed_only, "packed")[rows],
        genotypes.open_store(dataset, "ram")[rows],
    )


@pytest.mark.parametrize("load_bytes", [1, 7 * 1000, 10**9])
def test_snp_stripes_give_the_whole_matrix_and_the_same_pack(
    tmp_path, kgp_mini, dataset, load_bytes
):
    """Stripes of SNPs (one per 4 SNPs at the smallest, several of uneven width, or the
    whole file) cover the matrix exactly, and a packed-only build from them is the full
    dataset's pack."""
    from dietnet.plink import Bed

    bed = Bed.open(kgp_mini.plink_prefix)
    whole = np.full((bed.n_samples, bed.n_snps), 9, dtype=np.int8)
    for start, snp, block in bed.snp_stripes(load_bytes, block_bytes=4 * 1000):
        assert snp % 4 == 0 and start % 4 == 0
        assert (whole[start : start + len(block), snp : snp + block.shape[1]] == 9).all()
        whole[start : start + len(block), snp : snp + block.shape[1]] = block
    with h5py.File(dataset, "r") as f:
        assert np.array_equal(whole, f["inputs"][()])

    out = ds.build(kgp_mini.plink_prefix, tmp_path / "s.h5", class_labels=kgp_mini.labels,
                   packed_only=True, load_bytes=load_bytes, block_bytes=4 * 1000)  # fmt: skip
    genotypes.pack(dataset)
    assert np.array_equal(
        np.load(genotypes.packed_path(out)), np.load(genotypes.packed_path(dataset))
    )


def test_a_packed_only_dataset_is_read_only_through_its_pack(packed_only):
    """auto picks the pack; anything that would read the absent int8 values is refused,
    including auto once the pack is stale, and re-packing."""
    assert genotypes.choose(packed_only, "auto") == "packed"
    assert genotypes.choose(packed_only, "packed") == "packed"
    for mode in ("ram", "memmap"):
        with pytest.raises(ValueError, match=f"--inputs {mode} reads the int8 genotypes"):
            genotypes.choose(packed_only, mode)
    with pytest.raises(ValueError, match="pack-genotypes reads the int8 genotypes"):
        genotypes.pack(packed_only)
    with h5py.File(packed_only, "a") as f:
        f.attrs["rebuilt"] = 1
    with pytest.raises(ValueError, match="only in its packed copy"):
        genotypes.choose(packed_only, "auto")
