"""Corner cases of reading data: labels, SNP tables, PLINK files, genotype stores
and covariate adjustment."""

import os
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest

from dietnet import dataset as ds
from dietnet import genotypes, phenotype, plink

# --- dataset -------------------------------------------------------------------


def test_a_labels_file_with_one_column_is_refused(tmp_path):
    """Labels need a sample id and a label; one column alone is an error naming the file."""
    path = tmp_path / "labels.tsv"
    path.write_text("Sample\nS1\nS2\n")
    with pytest.raises(ValueError, match="labels.tsv: expected two tab-separated columns"):
        ds.read_labels(path)


def test_duplicated_sample_ids_in_labels_are_refused(tmp_path):
    """A sample labelled twice is ambiguous, so it is an error that shows the id."""
    path = tmp_path / "labels.tsv"
    path.write_text("Sample\tPop\nS1\tA\nS2\tB\nS1\tC\n")
    with pytest.raises(ValueError, match=r"duplicated sample ids, e.g. \['S1'\]"):
        ds.read_labels(path)


def test_labels_keep_ids_and_extra_columns_are_ignored(tmp_path):
    """Ids and labels are read as text; only the first two columns count."""
    path = tmp_path / "labels.tsv"
    path.write_text("Sample\tPop\tNote\n001\t7\tx\n002\t8\ty\n")
    labels = ds.read_labels(path)
    assert list(labels.index) == ["001", "002"] and list(labels) == ["7", "8"]


def test_a_regression_only_dataset_has_no_class_labels(tmp_path, synthetic):
    """With only regression labels the file holds them as float64 and no classes."""
    trait = synthetic.labels.with_name("synthetic_trait.tsv")
    out = ds.build(synthetic.plink_prefix, tmp_path / "r.h5", regression_labels=trait)

    with h5py.File(out) as f:
        assert "class_labels" not in f and "class_label_names" not in f
        values = np.array(f["regression_labels"])
        samples = np.array(f["samples"]).astype(str)
    table = pd.read_csv(trait, sep="\t", dtype=str)
    truth = table.set_index(table.columns[0])[table.columns[1]].astype(float)
    assert values.dtype == np.float64
    np.testing.assert_allclose(values, truth.loc[samples].to_numpy())


def test_a_sample_without_a_label_is_refused(tmp_path, synthetic):
    """Every PLINK sample needs a label; the error counts and names the unlabeled ones."""
    table = pd.read_csv(synthetic.labels, sep="\t", dtype=str)
    path = tmp_path / "labels.tsv"
    table.iloc[2:].to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="2 samples in the PLINK files have no label"):
        ds.build(synthetic.plink_prefix, tmp_path / "x.h5", class_labels=path)


def write_names(path, names):
    """An upstream-style dataset file: only ``snp_names``, no ``snps/`` group."""
    with h5py.File(path, "w") as f:
        f.create_dataset("snp_names", data=np.array(names, dtype="S"))
    return path


def test_snp_names_without_positions_give_ids_and_alleles_only(tmp_path):
    """``rs`` ids carry no position: the table has the counted allele but no chrom or pos."""
    table = ds.snp_table(write_names(tmp_path / "d.h5", ["rs1_A", "rs2_GT"]))
    assert list(table["snp"]) == ["rs1", "rs2"] and list(table["a1"]) == ["A", "GT"]
    assert "chrom" not in table.columns and "pos" not in table.columns


def test_snp_names_with_a_non_numeric_position_give_no_positions(tmp_path):
    """Four ``:`` fields whose second is not a number are not ``chr:pos:ref:alt``."""
    table = ds.snp_table(write_names(tmp_path / "d.h5", ["chr1:abc:G:A_A"]))
    assert "pos" not in table.columns and list(table["a1"]) == ["A"]


def test_positional_names_without_an_allele_suffix_give_no_alleles(tmp_path):
    """``chr:pos:ref:alt`` without ``_A1`` gives position but neither allele."""
    table = ds.snp_table(write_names(tmp_path / "d.h5", ["chr1:100:G:A", "chr2:200:C:T"]))
    assert list(table["snp"]) == ["chr1:100:G:A", "chr2:200:C:T"]
    assert list(table["chrom"]) == ["1", "2"] and list(table["pos"]) == [100, 200]
    assert table["a1"].isna().all() and "a2" not in table.columns


def test_the_other_allele_is_unknown_when_a1_is_neither_ref_nor_alt(tmp_path):
    """``a2`` is REF when A1 is ALT, ALT when A1 is REF, and unknown otherwise."""
    names = ["chr1:100:G:A_A", "chr1:200:G:A_G", "chr1:300:G:A_T"]
    table = ds.snp_table(write_names(tmp_path / "d.h5", names))
    assert list(table["a2"][:2]) == ["G", "A"] and table["a2"][2] is None


# --- plink ---------------------------------------------------------------------


def write_fileset(prefix, n_samples, n_snps, bed: bytes):
    prefix.with_suffix(".fam").write_text(
        "".join(f"F{i} S{i} 0 0 0 -9\n" for i in range(n_samples))
    )
    prefix.with_suffix(".bim").write_text(
        "".join(f"1 rs{j} 0 {j + 1} A G\n" for j in range(n_snps))
    )
    prefix.with_suffix(".bed").write_bytes(bed)
    return prefix


def test_a_bed_without_plinks_magic_bytes_is_refused(tmp_path):
    """An individual-major or non-PLINK .bed is refused up front."""
    prefix = write_fileset(tmp_path / "x", 4, 2, bytes([0x6C, 0x1B, 0x00, 0, 0]))
    with pytest.raises(ValueError, match="is not a SNP-major PLINK 1 .bed file"):
        plink.Bed.open(prefix)


def test_a_bed_of_the_wrong_size_is_refused(tmp_path):
    """5 samples need two bytes per SNP: a .bed with one byte per SNP is refused."""
    prefix = write_fileset(tmp_path / "x", 5, 3, bytes([0x6C, 0x1B, 0x01]) + bytes(3))
    with pytest.raises(ValueError, match="has 6 bytes; 5 samples x 3 SNPs need 9"):
        plink.Bed.open(f"{prefix}.bed")


def test_a_bed_of_the_right_size_decodes(tmp_path):
    """Two samples per byte pair: hom A1, het, missing, hom A2 decode to 2, 1, -1, 0."""
    byte = 0b11_01_10_00  # samples 0..3: 00, 10, 01, 11
    prefix = write_fileset(tmp_path / "x", 4, 1, bytes([0x6C, 0x1B, 0x01, byte]))
    bed = plink.Bed.open(prefix)
    [(start, block)] = list(bed.sample_blocks())
    assert start == 0 and block[:, 0].tolist() == [2, 1, -1, 0]


def test_a_short_file_raises_eof(tmp_path):
    """Asking for more bytes than the file holds raises EOFError, saying how many came."""
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(10)))
    with pytest.raises(EOFError, match="expected 20 bytes after offset 3, got 7"):
        plink._read_uncached(path, offset=3, size=20, chunk=4)


def test_pages_are_dropped_where_posix_fadvise_exists(tmp_path, monkeypatch):
    """Each chunk read is followed by a DONTNEED advice for exactly its bytes."""
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(10)))
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda fd, off, n, how: calls.append((off, n, how)), raising=False
    )
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)

    out = plink._read_uncached(path, offset=2, size=7, chunk=3)

    assert out.tolist() == list(range(2, 9))
    assert calls == [(2, 3, 4), (5, 3, 4), (8, 1, 4)]


# --- genotypes -----------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path, synthetic):
    return ds.build(synthetic.plink_prefix, tmp_path / "d.h5")


def test_an_unsupported_npy_version_is_refused(tmp_path):
    """A version 3.0 .npy header is not read, rather than misread."""
    path = tmp_path / "x.npy"
    with open(path, "wb") as f:
        np.lib.format.write_array(f, np.zeros((2, 3), dtype=np.uint8), version=(3, 0))
    with pytest.raises(ValueError, match=r"unsupported .npy format version \(3, 0\)"):
        genotypes.PackedReader.of(path)


def test_a_truncated_pack_raises_eof(tmp_path):
    """A row the header promises but the file lacks raises EOFError naming the row."""
    path = tmp_path / "x.npy"
    np.save(path, np.arange(6, dtype=np.uint8).reshape(2, 3))
    reader = genotypes.PackedReader.of(path)
    assert reader.read([1]).tolist() == [[3, 4, 5]]

    longer = genotypes.PackedReader(path, n_rows=4, row_bytes=3, offset=reader.offset)
    with pytest.raises(EOFError, match="short read at row 3"):
        longer.read([3])


def test_a_packed_store_reuses_its_map_and_pickles_without_it(dataset):
    """The memory map is opened once, and left out when the store is pickled."""
    genotypes.pack(dataset)
    store = genotypes.PackedStore(dataset)
    first = store.packed
    assert store.packed is first

    copy = pickle.loads(pickle.dumps(store))
    assert copy._packed is None
    np.testing.assert_array_equal(copy[[0, 5, 2]], store[[0, 5, 2]])


def test_a_memmap_store_has_a_length_and_pickles_as_its_path(dataset):
    """A pickled MemmapStore maps the file again and gives the same rows."""
    store = genotypes.MemmapStore(dataset)
    with h5py.File(dataset) as f:
        inputs = np.array(f["inputs"])
    assert len(store) == inputs.shape[0]

    copy = pickle.loads(pickle.dumps(store))
    assert copy._x is None
    np.testing.assert_array_equal(copy[[3, 1]], inputs[[3, 1]])


def test_read_rows_returns_rows_in_the_order_asked(dataset):
    """Rows read sorted come back in the caller's order, repeats included."""
    with h5py.File(dataset) as f:
        inputs = np.array(f["inputs"])
    rows = [7, 2, 7, 0]
    np.testing.assert_array_equal(genotypes.read_rows(inputs, rows), inputs[rows])


def test_open_store_refuses_an_unknown_mode(dataset):
    with pytest.raises(ValueError, match="inputs mode must be one of"):
        genotypes.open_store(dataset, "auto")


# --- phenotype -----------------------------------------------------------------


def test_a_trait_the_covariates_explain_fully_is_refused():
    """Nothing is left to learn when the residual is zero, so adjusting refuses."""
    table = pd.DataFrame(
        {"height": ["1", "3", "5", "7"], "age": ["0", "1", "2", "3"]}, index=list("abcd")
    )
    with pytest.raises(ValueError, match="height has no variance left after the covariates"):
        phenotype.adjust(table, "height", ["age"])


def test_a_constant_trait_is_refused():
    table = pd.DataFrame({"height": ["2", "2", "2"]}, index=list("abc"))
    with pytest.raises(ValueError, match="no variance left"):
        phenotype.adjust(table, "height", [])
