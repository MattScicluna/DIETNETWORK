"""The data this repository feeds the model is the data upstream fed it.

Upstream reads an HDF5 file built from ``plink --recode A``; this repository
reads PLINK directly. Everything downstream -- input statistics, the
genotype-frequency embedding, the model -- assumes the two agree, including on
which allele a dosage counts.
"""

import numpy as np
import pytest
import torch
from support import stratified_folds

from Dietnet.generate_embedding import compute_fold_embedding
from Dietnet.helpers import dataset_utils as du


def plink_in_upstream_order(dataset):
    """This repository's dosage matrix, rows reordered to match upstream's HDF5."""
    genotypes, fam, _ = du.load_plink_genotypes(
        str(dataset.plink_prefix), cache_file=None, verbose=False
    )
    _, _, samples = dataset.upstream_arrays()
    row_of = {iid: i for i, iid in enumerate(fam["iid"].values)}
    return np.asarray(genotypes)[[row_of[s] for s in samples]]


def test_plink_dosages_match_upstream_hdf5(dataset):
    """Same matrix, same allele counted, missing calls in the same places.

    If the reader counted the other allele every value would be 2 - x, and the
    published upstream models and embeddings would be mirror-coded against it.
    """
    upstream_genotypes, _, _ = dataset.upstream_arrays()

    np.testing.assert_array_equal(plink_in_upstream_order(dataset), upstream_genotypes)


def test_fixture_has_missing_calls(synthetic):
    """The comparisons below only cover imputation if there is something to impute."""
    genotypes, _, _ = synthetic.upstream_arrays()
    assert (genotypes < 0).mean() > 0.01


@pytest.mark.upstream
def test_input_statistics_match_upstream(dataset, upstream):
    stats = upstream["compute_input_features_statistics"]
    genotypes, labels, _ = dataset.upstream_arrays()
    train = stratified_folds(labels)[0][0]
    x = genotypes[train]

    expected = np.array(
        [stats.compute_feature_mean_and_sd(i, x[:, i])[1:] for i in range(x.shape[1])]
    )
    mean, sd = du.compute_norm_values(torch.from_numpy(x.astype(np.float32)))

    np.testing.assert_allclose(mean.numpy(), expected[:, 0], rtol=1e-5)
    np.testing.assert_allclose(sd.numpy(), expected[:, 1], rtol=1e-4)


@pytest.mark.upstream
def test_genotype_frequency_embedding_matches_upstream(dataset, upstream):
    genotypes, labels, _ = dataset.upstream_arrays()
    train = stratified_folds(labels)[0][0]

    _, _, expected = upstream["generate_embedding"].compute_fold_embedding(
        genotypes[train], labels[train], 0
    )
    actual = compute_fold_embedding(genotypes[train], labels[train])

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.upstream
def test_missing_value_imputation_matches_upstream(synthetic, upstream):
    genotypes, _, _ = synthetic.upstream_arrays()
    x = torch.from_numpy(genotypes[:32].astype(np.float32))
    means = torch.rand(x.shape[1]) * 2

    ours, theirs = x.clone(), x.clone()
    du.replace_missing_values(ours, means)
    upstream["helpers.dataset_utils"].replace_missing_values(theirs, means)

    torch.testing.assert_close(ours, theirs, rtol=0, atol=0)
    assert (ours >= 0).all()
