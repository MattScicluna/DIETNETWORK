"""Genotype stores: where training reads its rows from (dietnet.genotypes)."""

import h5py
import numpy as np
import pytest

from dietnet import dataset as ds
from dietnet import genotypes


@pytest.fixture
def dataset(tmp_path, kgp_mini):
    return ds.build(
        kgp_mini.plink_prefix, tmp_path / "d.h5", int8=True, class_labels=kgp_mini.labels
    )


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
    genotypes.packed_path(dataset).unlink()  # without its 2-bit copy, an int8 dataset is read
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


def rowwise(dataset) -> np.ndarray:
    """The reference: every row's 2-bit bytes, encoded from the int8 inputs."""
    with h5py.File(dataset, "r") as f:
        return genotypes.encode(f["inputs"][()])


def test_a_packed_store_gives_exactly_the_rows_upstream_copies(dataset):
    """The build writes the 2-bit copy along with int8 inputs; the store decodes it."""
    ram = genotypes.open_store(dataset, "ram")
    packed = genotypes.open_store(dataset, "packed")

    assert packed.shape == ram.shape and len(packed) == len(ram)
    for rows in (np.array([5, 0, 479, 5, 200]), np.arange(len(ram)), [3]):
        np.testing.assert_array_equal(packed[rows], ram[rows])
    for row in (0, np.int64(17), len(ram) - 1):
        np.testing.assert_array_equal(packed[row], ram[row])
    assert genotypes.packed_path(dataset).stat().st_size < ram.nbytes / 3


def test_whole_rows_are_the_rows_bytes(dataset):
    """PackedFile.read (one read per run of rows and block, through gaps) is the rows'
    2-bit bytes: runs, gaps, repeats, any order, none, single rows, the last row; and it
    pickles without its open file, for loader workers."""
    import pickle

    reference = rowwise(dataset)
    reader = genotypes.PackedFile(dataset)
    n = len(reference)
    rng = np.random.default_rng(1)
    for rows in (
        np.arange(n), np.array([0, 1, 2, 7, 8, 30]), np.array([9, 3, 3, 4, n - 1]),
        np.array([], dtype=np.int64), np.flatnonzero(rng.random(n) < 0.3),
    ):  # fmt: skip
        for gap in (0, 1, 8, 1000):
            np.testing.assert_array_equal(reader.read(rows, max_gap=gap), reference[rows])
    np.testing.assert_array_equal(reader[n - 1], reference[n - 1])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(reader))[[4, 5]], reference[[4, 5]])
    with pytest.raises(IndexError):
        reader[[n]]


@pytest.mark.parametrize("n_blocks", [1, 3, 8, genotypes.PACKED_BLOCKS])
def test_ranges_are_read_from_the_blocks_that_cover_them(dataset, n_blocks):
    """read(rows, ranges=...) is the rows' bytes of those SNP ranges side by side: whole
    blocks of the copy, other blocks' ranges (cut from the blocks that cover them), and
    ranges across blocks. A range must start on a byte."""
    reference = rowwise(dataset)
    reader = genotypes.PackedFile(dataset)
    edges = genotypes.block_edges(reader.n_snps, n_blocks)
    blocks = np.stack([edges[:-1], edges[1:]], axis=1)
    rows = np.array([0, 2, 3, 9, len(reference) - 1])
    across = [[4, 40], [100, reader.n_snps]]
    for ranges in (blocks[:1], blocks, blocks[::2], across):
        expected = np.concatenate([reference[rows][:, a // 4 : -(-b // 4)] for a, b in ranges], 1)
        np.testing.assert_array_equal(reader.read(rows, max_gap=4, ranges=ranges), expected)
    with pytest.raises(ValueError, match="multiple of 4"):
        reader.read(rows, ranges=[[2, 40]])


@pytest.mark.parametrize("load_bytes", [1, 7 * 1000, 10**9])
def test_every_way_of_writing_the_copy_writes_the_same_file(
    tmp_path, kgp_mini, dataset, load_bytes
):
    """From the .bed in stripes of whole blocks (a block per stripe at the smallest, several,
    or the whole file), while writing int8 inputs, or from them (pack): the same bytes."""
    from dietnet.plink import Bed

    bed = Bed.open(kgp_mini.plink_prefix)
    stripes = genotypes.packed_stripes(bed.n_snps, max(4, load_bytes // bed.bytes_per_snp))
    assert stripes[0][0] == 0 and stripes[-1][1] == bed.n_snps
    assert all(a[1] == b[0] for a, b in zip(stripes, stripes[1:]))
    whole = np.full((bed.n_samples, bed.n_snps), 9, dtype=np.int8)
    for start, snp, block in bed.snp_stripes(stripes, block_bytes=4 * 1000):
        whole[start : start + len(block), snp : snp + block.shape[1]] = block
    with h5py.File(dataset, "r") as f:
        np.testing.assert_array_equal(whole, f["inputs"][()])

    out = ds.build(kgp_mini.plink_prefix, tmp_path / "s.h5", class_labels=kgp_mini.labels,
                   load_bytes=load_bytes, block_bytes=4 * 1000)  # fmt: skip
    built = genotypes.packed_path(dataset).read_bytes()
    assert genotypes.packed_path(out).read_bytes() == built
    genotypes.pack(dataset)
    assert genotypes.packed_path(dataset).read_bytes() == built
    assert not list(tmp_path.glob("*.partial"))


def test_auto_prefers_an_up_to_date_pack(dataset):
    assert genotypes.choose(dataset, "auto") == "packed"
    genotypes.packed_path(dataset).unlink()
    assert genotypes.choose(dataset, "auto") == "ram"
    with pytest.raises(ValueError, match="pack-genotypes"):
        genotypes.choose(dataset, "packed")

    genotypes.pack(dataset)

    assert genotypes.choose(dataset, "auto") == "packed"
    assert genotypes.choose(dataset, "memmap") == "memmap"


def test_a_pack_of_a_dataset_that_changed_since_is_refused(dataset):
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
    return ds.build(kgp_mini.plink_prefix, tmp_path / "p.h5", class_labels=kgp_mini.labels)


def test_by_default_a_dataset_keeps_its_genotypes_only_in_the_copy(dataset, packed_only):
    """create-dataset without --int8: the same file minus the int8 values, and the same
    2-bit copy."""
    assert genotypes.is_packed(packed_only)
    assert (
        genotypes.packed_path(packed_only).read_bytes()
        == genotypes.packed_path(dataset).read_bytes()
    )
    with h5py.File(dataset, "r") as full, h5py.File(packed_only, "r") as f:
        assert f["inputs"].shape == full["inputs"].shape
        assert f["inputs"].id.get_storage_size() == 0  # nothing stored
        assert not f.attrs["inputs_stored"] and full.attrs["inputs_stored"]
        for key in ("snp_names", "samples", "class_labels", "snps/pos"):
            assert np.array_equal(f[key][()], full[key][()]), key
    assert genotypes.inputs_stored(dataset) and not genotypes.inputs_stored(packed_only)
    rows = np.array([3, 0, 7])
    np.testing.assert_array_equal(
        genotypes.open_store(packed_only, "packed")[rows],
        genotypes.open_store(dataset, "ram")[rows],
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


def test_cached_rows_are_reused_for_the_same_rows_of_the_same_data(tmp_path):
    """CachedRows: written a batch at a time; a later run asking for the same rows of the
    same data maps that file without reading; other data or rows get their own."""
    source = np.arange(40, dtype=np.uint8).reshape(10, 4)
    first = genotypes.CachedRows(lambda r: source[r], [7, 2, 5], tmp_path / "c", "data", batch=2)
    assert not first.reused and isinstance(first.data, np.memmap)
    np.testing.assert_array_equal(first[[5, 2]], source[[5, 2]])
    with pytest.raises(KeyError):
        first[[3]]

    def unreadable(rows):
        raise AssertionError("read again")

    again = genotypes.CachedRows(unreadable, [2, 5, 7], tmp_path / "c", "data")
    assert again.reused and again.path == first.path
    other = genotypes.CachedRows(lambda r: source[r], [2, 5, 7], tmp_path / "c", "other data")
    assert not other.reused and other.path != first.path
    assert not list((tmp_path / "c").glob("*.partial"))
