"""SNP features from several sources: what build-embedding writes, block by block."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest
from support import dietnet

from dietnet import dataset as ds
from dietnet import features

COMPLEMENT = str.maketrans("ACGT", "TGCA")


@pytest.fixture
def prepared(tmp_path, kgp_mini):
    """kgp_mini as a dataset, partitioned the way the pipeline does it."""
    h5 = ds.build(kgp_mini.plink_prefix, tmp_path / "d.h5", class_labels=kgp_mini.labels)
    dietnet("partition", dataset=h5, out=tmp_path / "p.npz")
    return h5, tmp_path / "p.npz"


def test_genotype_freq_pools_upstreams_per_class_frequencies(prepared, tmp_path):
    """The regression baseline: count(genotype == g) / n over each fold's training
    samples, which is upstream's per-class embedding averaged with class sizes as
    weights."""
    h5, partition = prepared
    pooled, meta = features.build(
        {"blocks": [{"provider": "genotype_freq", "block": 37}]}, h5, partition, tmp_path / "e.npz"
    )
    per_class, _ = features.build(
        {"blocks": [{"provider": "class_genotype_freq"}]}, h5, partition, tmp_path / "c.npz"
    )

    x = ds.memmap_inputs(h5)
    with h5py.File(h5, "r") as f:
        labels = np.array(f["class_labels"])
    folds = np.load(partition, allow_pickle=True)["folds_indexes"]
    assert pooled.shape == (len(folds), x.shape[1], 3)
    assert meta["column_names"] == ["all:genotype0", "all:genotype1", "all:genotype2"]
    for f, fold in enumerate(folds):
        train = np.asarray(fold[0])
        g = np.asarray(x[np.sort(train)])
        direct = np.stack([(g == value).mean(axis=0) for value in range(3)], axis=1)
        np.testing.assert_allclose(pooled[f], direct)

        sizes = np.bincount(labels[train], minlength=per_class.shape[-1] // 3)
        weighted = per_class[f].reshape(x.shape[1], -1, 3) * (sizes / sizes.sum())[None, :, None]
        np.testing.assert_allclose(pooled[f], weighted.sum(axis=1))


def table_from_ids(h5):
    """A table keyed by the chr:pos:REF:ALT ids, one effect column of +1 (ALT raises it)."""
    snps = ds.snp_table(h5)
    parts = snps["snp"].str.split(":", expand=True)
    return snps, pd.DataFrame(
        {
            "chrom": parts[0],
            "pos": parts[1].astype(int),
            "ref": parts[2],
            "alt": parts[3],
            "effect": 1.0,
        }
    )


def test_class_genotype_freq_alone_is_upstreams_embedding(tmp_path, prepared):
    h5, partition = prepared
    dietnet("generate-embedding", dataset=h5, partition=partition, out=tmp_path / "upstream.npz")

    values, meta = features.build(
        {"blocks": [{"provider": "class_genotype_freq"}]}, h5, partition, tmp_path / "e.npz"
    )

    np.testing.assert_array_equal(values, np.load(tmp_path / "upstream.npz")["emb"])
    assert meta["fold_dependent"] and meta["column_names"][0].endswith(":genotype0")


def test_relative_paths_reach_upstreams_script(tmp_path, prepared, monkeypatch):
    """Upstream's scripts run from their own directory; paths given relative to ours
    must still find the dataset, as they do for generate-embedding."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spec.yaml").write_text("blocks:\n  - provider: class_genotype_freq\n")
    dietnet("build-embedding", dataset="d.h5", partition="p.npz", spec="spec.yaml", out="e.npz")
    assert np.load(tmp_path / "e.npz")["emb"].shape[1] == len(ds.snp_table(prepared[0]))


def test_effects_are_oriented_to_the_counted_allele(tmp_path, prepared):
    """+1 where the dataset counts ALT, -1 where it counts REF, missing where unknowable."""
    h5, partition = prepared
    snps, table = table_from_ids(h5)
    dropped = np.arange(0, len(table), 50)
    table.drop(index=dropped).to_csv(tmp_path / "t.tsv", sep="\t", index=False)

    spec = {
        "blocks": [{"provider": "precomputed", "path": str(tmp_path / "t.tsv"), "orient": "effect"}]
    }
    values, meta = features.build(spec, h5, partition, tmp_path / "e.npz")

    alt, ref = table["alt"], table["ref"]
    counts_alt = (snps["a1"] == alt).to_numpy()
    ambiguous = (ref == alt.str.translate(COMPLEMENT)).to_numpy()
    missing = ambiguous.copy()
    missing[dropped] = True

    effect, flag = values[:, 0], values[:, 1]
    np.testing.assert_array_equal(flag, missing.astype(float))
    assert (effect[missing] == 0).all()
    np.testing.assert_array_equal(effect[~missing], np.where(counts_alt, 1.0, -1.0)[~missing])
    # Both orientations occur in the 1KGP data: the sign really is per SNP.
    assert 0 < meta["blocks"][0]["negated"] < (~missing).sum()


def test_a_table_on_the_other_strand_gives_the_same_features(tmp_path, prepared):
    h5, partition = prepared
    _, table = table_from_ids(h5)
    table.to_csv(tmp_path / "plus.tsv", sep="\t", index=False)
    flipped = table.assign(
        ref=table["ref"].str.translate(COMPLEMENT), alt=table["alt"].str.translate(COMPLEMENT)
    )
    flipped.to_csv(tmp_path / "minus.tsv", sep="\t", index=False)

    def build(path):
        spec = {"blocks": [{"provider": "precomputed", "path": str(path), "orient": "effect"}]}
        return features.build(spec, h5, partition, tmp_path / "e.npz")[0]

    np.testing.assert_array_equal(build(tmp_path / "plus.tsv"), build(tmp_path / "minus.tsv"))


def test_controls_transforms_and_mixing_shared_with_per_fold_blocks(tmp_path, prepared):
    h5, partition = prepared
    spec = {
        "blocks": [
            {"provider": "class_genotype_freq"},
            {"provider": "random", "columns": 20, "seed": 1, "pca": 5},
            {"provider": "shuffled", "of": 1, "seed": 2},
            {"provider": "random", "columns": 3, "seed": 3, "center": True},
        ]
    }
    values, meta = features.build(spec, h5, partition, tmp_path / "e.npz")

    n_folds, n_snps = 5, 2000
    assert values.shape == (n_folds, n_snps, 72 + 5 + 5 + 3)
    shared = values[:, :, 72:]
    assert (shared == shared[0]).all()  # broadcast to every fold
    pcs, shuffled, centred = shared[0, :, :5], shared[0, :, 5:10], shared[0, :, 10:]
    assert sorted(map(tuple, pcs)) == sorted(map(tuple, shuffled))  # same rows, permuted
    assert not np.array_equal(pcs, shuffled)
    np.testing.assert_allclose(centred.mean(axis=0), 0, atol=1e-12)
    assert [b["columns"] for b in meta["blocks"]] == [[0, 72], [72, 77], [77, 82], [82, 85]]
    saved = json.loads((tmp_path / "e.npz.json").read_text())
    assert saved["shape"] == list(values.shape)


def test_all_zero_columns_are_refused(tmp_path, prepared):
    h5, partition = prepared
    _, table = table_from_ids(h5)
    table.assign(effect=0.0).to_csv(tmp_path / "t.tsv", sep="\t", index=False)
    spec = {"blocks": [{"provider": "precomputed", "path": str(tmp_path / "t.tsv")}]}

    with pytest.raises(ValueError, match="all-zero"):
        features.build(spec, h5, partition, tmp_path / "e.npz")
