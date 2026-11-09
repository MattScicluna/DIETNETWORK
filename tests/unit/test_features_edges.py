"""Corner cases of building auxiliary-network features: bad tables, bad specs.

Providers are called directly with a hand-made :class:`dietnet.features.Context`
where they only need the SNP table, so most tests need no dataset file.
"""

import numpy as np
import pandas as pd
import pytest

from dietnet import dataset as ds
from dietnet import features

SNPS = pd.DataFrame(
    {
        "chrom": ["1", "1", "2", "2"],
        "pos": [100, 200, 300, 400],
        "a1": ["A", "C", "A", "T"],
        "a2": ["G", "T", "T", "C"],
    }
)


def context(snps=SNPS, n_folds=1, built=None):
    return features.Context("unused.h5", "unused.npz", snps, n_folds, built or [])


def write_table(path, rows, columns=("chrom", "pos", "ref", "alt", "score")):
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, sep="\t", index=False)
    return path


def test_a_parquet_table_matches_like_a_tsv(tmp_path):
    """A ``.parquet`` table is read with pandas' parquet reader and matched the same way."""
    pytest.importorskip("pyarrow")
    rows = [("chr1", 100, "G", "A", 1.5), ("1", 200, "C", "T", -2.0)]
    tsv = write_table(tmp_path / "t.tsv", rows)
    parquet = tmp_path / "t.parquet"
    pd.read_csv(tsv, sep="\t").to_parquet(parquet)

    from_tsv = features._precomputed(context(), {"path": str(tsv)})
    from_parquet = features._precomputed(context(), {"path": str(parquet)})

    np.testing.assert_array_equal(from_parquet.values, from_tsv.values)
    assert from_parquet.columns == ["t:score", "t:missing"]


def test_a_table_missing_key_columns_is_refused(tmp_path):
    path = write_table(tmp_path / "t.tsv", [("1", 100, "G", 1.0)], ("chrom", "pos", "ref", "s"))
    with pytest.raises(ValueError, match=r"missing columns \['alt'\]"):
        features._match_table(context(), {}, path)


def test_an_unknown_orientation_is_refused(tmp_path):
    path = write_table(tmp_path / "t.tsv", [("1", 100, "G", "A", 1.0)])
    with pytest.raises(ValueError, match="orient must be 'effect' or 'none', not 'flip'"):
        features._match_table(context(), {"orient": "flip"}, path)


@pytest.mark.parametrize(
    "snps",
    [SNPS.drop(columns="pos"), SNPS.drop(columns="a2"), SNPS.assign(a2=[None, "T", "T", "C"])],
    ids=["no-positions", "no-a2", "one-a2-unknown"],
)
def test_a_dataset_without_positions_or_alleles_cannot_match_a_table(tmp_path, snps):
    path = write_table(tmp_path / "t.tsv", [("1", 100, "G", "A", 1.0)])
    with pytest.raises(ValueError, match="lack positions or alleles"):
        features._match_table(context(snps), {}, path)


def test_nan_values_in_matched_rows_are_refused(tmp_path):
    """A NaN in a matched SNP's row is an error naming the table."""
    path = write_table(tmp_path / "t.tsv", [("1", 100, "G", "A", float("nan"))])
    with pytest.raises(ValueError, match="t.tsv: NaN feature values"):
        features._match_table(context(), {}, path)


def test_nan_values_in_unmatched_rows_are_ignored(tmp_path):
    """Only rows matched to the dataset are read, so a NaN elsewhere does no harm."""
    path = write_table(
        tmp_path / "t.tsv", [("1", 100, "G", "A", 2.0), ("9", 999, "G", "A", float("nan"))]
    )
    values, covered, names, info = features._match_table(context(), {}, path)
    assert values[:, 0].tolist() == [2.0, 0, 0, 0] and covered.tolist() == [1, 0, 0, 0]
    assert names == ["score"] and info["coverage"] == 0.25


def test_effects_are_signed_by_the_counted_allele_on_either_strand(tmp_path):
    """ALT counted keeps the sign, REF counted negates it, also on the other strand;
    an ambiguous A/T SNP has no knowable sign and is left out."""
    rows = [
        ("1", 100, "G", "A", 1.0),  # A1=A is ALT: kept
        ("1", 200, "C", "T", 2.0),  # A1=C is REF: negated
        ("2", 300, "T", "A", 3.0),  # A/T: ambiguous
        ("2", 400, "G", "A", 4.0),  # other strand: A1=T is A complemented, ALT: kept
    ]
    path = write_table(tmp_path / "t.tsv", rows)
    values, covered, _, info = features._match_table(context(), {"orient": "effect"}, path)

    assert values[:, 0].tolist() == [1.0, -2.0, 0.0, 4.0]
    assert covered.tolist() == [True, True, False, True] and info["negated"] == 1

    unsigned, covered, _, _ = features._match_table(context(), {}, path)
    assert unsigned[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0] and covered.all()


def test_per_fold_tables_must_have_the_same_feature_columns(tmp_path):
    """Folds whose tables name different features cannot be stacked."""
    write_table(tmp_path / "g0.tsv", [("1", 100, "G", "A", 1.0)])
    write_table(
        tmp_path / "g1.tsv", [("1", 100, "G", "A", 1.0)], ("chrom", "pos", "ref", "alt", "z")
    )
    spec = {"path": str(tmp_path / "g{fold}.tsv")}
    with pytest.raises(ValueError, match=r"g1.tsv: feature columns \['z'\] differ from"):
        features._precomputed(context(n_folds=2), spec)


@pytest.mark.parametrize("of", [-1, 1, 5])
def test_shuffled_must_name_an_earlier_block(of):
    built = [features.Block(np.ones((4, 2)), ["a:0", "a:1"])]
    with pytest.raises(ValueError, match=r"'of' must name an earlier block \(0..0\)"):
        features._shuffled(context(built=built), {"of": of})


def test_shuffled_with_no_earlier_block_is_refused():
    with pytest.raises(ValueError, match="earlier block"):
        features._shuffled(context(), {"of": 0})


def test_shuffled_permutes_rows_of_a_per_fold_block():
    """A per-fold block's SNP rows are permuted the same way in every fold."""
    values = np.arange(2 * 4 * 1, dtype=float).reshape(2, 4, 1)
    built = [features.Block(values, ["a:0"])]
    block = features._shuffled(context(built=built), {"of": 0, "seed": 3})

    perm = np.random.default_rng(3).permutation(4)
    np.testing.assert_array_equal(block.values, values[:, perm, :])
    assert block.columns == ["shuffled3:a:0"] and block.info == {"of": 0, "seed": 3}


def test_pca_cannot_keep_more_components_than_columns():
    block = features.Block(np.random.default_rng(0).standard_normal((5, 2)), ["x:0", "x:1"])
    with pytest.raises(ValueError, match="pca: 3 components from 2 columns"):
        features._transform(block, {"pca": 3})


def test_standardize_without_pca_is_refused():
    block = features.Block(np.ones((3, 1)), ["x:0"])
    with pytest.raises(ValueError, match="standardize applies to pca"):
        features._transform(block, {"standardize": True})


def test_standardize_refuses_a_constant_column():
    values = np.column_stack([np.arange(4.0), np.ones(4)])
    with pytest.raises(ValueError, match="a column does not vary"):
        features._pca(values, 1, standardize=True)


@pytest.fixture
def small(tmp_path, synthetic):
    """A dataset and a two-fold partition file, without running upstream."""
    dataset = ds.build(synthetic.plink_prefix, tmp_path / "d.h5")
    partition = tmp_path / "p.npz"
    np.savez(partition, folds_indexes=np.zeros((2, 1)))
    return dataset, partition


def test_an_unknown_provider_is_refused_naming_the_known_ones(tmp_path, small):
    spec = {"blocks": [{"provider": "random", "columns": 2}, {"provider": "magic"}]}
    with pytest.raises(ValueError, match=r"block 1: unknown provider 'magic'; one of \["):
        features.build(spec, *small, tmp_path / "e.npz")


def test_a_block_without_a_provider_is_refused(tmp_path, small):
    with pytest.raises(ValueError, match="block 0: unknown provider None"):
        features.build({"blocks": [{}]}, *small, tmp_path / "e.npz")


def test_a_spec_with_every_block_excluded_is_refused(tmp_path, small):
    spec = {"blocks": [{"provider": "random", "columns": 2, "include": False}]}
    with pytest.raises(ValueError, match="nothing to write"):
        features.build(spec, *small, tmp_path / "e.npz")
