"""The HDF5 this package builds is the one upstream's create_dataset.py builds.

Upstream goes PLINK -> ``plink --recode A`` text -> create_dataset.py; this
package reads the .bed directly. Everything downstream -- partition, embedding,
statistics, training -- reads the file, so it must be the same file: same
datasets, same dtypes, same values, including which allele a dosage counts and
where calls are missing. The references are committed (tests/fixtures), built
by upstream's own script with plink 1.9.
"""

import h5py
import numpy as np
import pandas as pd
import pytest

from dietnet import dataset as ds


@pytest.fixture(params=[2**32, 37_000], ids=["one-block", "many-blocks"])
def built(tmp_path, dataset, request):
    """Built in one sample block, and in many with a partial last one."""
    extra = {}
    if dataset.name == "synthetic":
        extra["regression_labels"] = dataset.labels.with_name("synthetic_trait.tsv")
    return ds.build(
        dataset.plink_prefix,
        tmp_path / "d.h5",
        class_labels=dataset.labels,
        block_bytes=request.param,
        **extra,
    )


def test_every_upstream_dataset_is_identical(built, dataset):
    with h5py.File(built) as ours, h5py.File(dataset.upstream_hdf5) as theirs:
        assert set(theirs) <= set(ours)
        for key in theirs:
            assert ours[key].dtype == theirs[key].dtype, key
            np.testing.assert_array_equal(np.array(ours[key]), np.array(theirs[key]), err_msg=key)


def test_fixtures_cover_missing_calls_and_regression(synthetic):
    with h5py.File(synthetic.upstream_hdf5) as f:
        assert (np.array(f["inputs"]) < 0).mean() > 0.01
        assert "regression_labels" in f


def test_inputs_are_contiguous_for_memory_mapping(built):
    with h5py.File(built) as f:
        expected = np.array(f["inputs"])
        assert f["inputs"].chunks is None and f["inputs"].compression is None
    np.testing.assert_array_equal(np.asarray(ds.memmap_inputs(built)), expected)


def test_snp_table_of_an_upstream_made_file_matches_the_bim(built, dataset):
    """Files without our snps/ group (upstream's, the published one) are parsed from names."""
    parsed = ds.snp_table(dataset.upstream_hdf5)
    recorded = ds.snp_table(built)
    if dataset.name == "kgp_mini":
        # chr:pos:REF:ALT ids: position and both alleles recoverable from the name.
        cols = ["snp", "chrom", "pos", "a1", "a2"]
        pd.testing.assert_frame_equal(parsed[cols], recorded[cols], check_dtype=False)
    else:
        assert (parsed["a1"] == recorded["a1"]).all()


def test_samples_without_labels_are_an_error(tmp_path, synthetic):
    labels = pd.read_csv(synthetic.labels, sep="\t").iloc[:-1]
    labels.to_csv(tmp_path / "partial.tsv", sep="\t", index=False)

    with pytest.raises(ValueError, match="no label"):
        ds.build(synthetic.plink_prefix, tmp_path / "d.h5", class_labels=tmp_path / "partial.tsv")
