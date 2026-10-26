"""AlphaGenome scores to a feature table, with a stand-in for the API.

The stand-in returns rows shaped like ``variant_scorers.tidy_scores`` output.
Each track's effect is a known function of the variant, so the table and the
features built from it can be checked exactly, without network or key.
"""

import numpy as np
import pandas as pd
import pytest
from support import dietnet

from dietnet import alphagenome
from dietnet import dataset as ds
from dietnet import features


class FakeAPI:
    """Two tracks of one output type, plus a gene-level track with two genes per variant."""

    def __init__(self, fail_after=None, extra_track_from_call=None):
        self.calls, self.scored = 0, 0
        self.fail_after, self.extra_from = fail_after, extra_track_from_call

    @staticmethod
    def effect(pos, track):
        return ((pos % 97) - 48) / 10.0 * (1 + track)

    def score(self, variants):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise ConnectionError("API went away")
        self.calls += 1
        self.scored += len(variants)
        rows = []
        for v in variants.itertuples():
            base = dict(chrom=v.chrom, pos=v.pos, ref=v.ref, alt=v.alt)
            for track in (0, 1):
                rows.append({**base, "output_type": "DNASE", "track_name": f"t{track}",
                             "raw_score": self.effect(v.pos, track)})  # fmt: skip
            rows.append({**base, "output_type": "RNA_SEQ", "track_name": "liver", "raw_score": 0.1})
            rows.append(
                {**base, "output_type": "RNA_SEQ", "track_name": "liver", "raw_score": -2.0}
            )
            if self.extra_from is not None and self.calls > self.extra_from:
                rows.append({**base, "output_type": "CAGE", "track_name": "late", "raw_score": 1.0})
        return pd.DataFrame(rows)


@pytest.fixture
def dataset(tmp_path, kgp_mini):
    return ds.build(kgp_mini.plink_prefix, tmp_path / "d.h5", class_labels=kgp_mini.labels)


def test_variants_come_from_chr_pos_ref_alt_ids(dataset):
    variants = alphagenome.variants_of(dataset)
    first = variants.iloc[0]
    assert first["chrom"].startswith("chr") and first["pos"] > 0
    assert len(variants) == 2000


def test_score_writes_one_row_per_snp_with_the_strongest_gene_effect(tmp_path, dataset):
    out = alphagenome.score(dataset, tmp_path / "ag.tsv", FakeAPI(), chunk=300)
    table = pd.read_csv(out, sep="\t")

    assert list(table.columns) == [
        "chrom",
        "pos",
        "ref",
        "alt",
        "DNASE|t0",
        "DNASE|t1",
        "RNA_SEQ|liver",
    ]
    assert len(table) == 2000
    np.testing.assert_allclose(table["DNASE|t1"], FakeAPI.effect(table["pos"], 1))
    assert (table["RNA_SEQ|liver"] == -2.0).all()  # larger magnitude, sign kept


def test_an_interrupted_run_resumes_without_rescoring(tmp_path, dataset):
    flaky = FakeAPI(fail_after=3)
    with pytest.raises(ConnectionError):
        alphagenome.score(dataset, tmp_path / "ag.tsv", flaky, chunk=300)

    again = FakeAPI()
    alphagenome.score(dataset, tmp_path / "ag.tsv", again, chunk=300)

    assert flaky.scored == 900 and again.scored == 1100
    assert len(pd.read_csv(tmp_path / "ag.tsv", sep="\t")) == 2000


def test_tracks_appearing_later_are_zero_where_unscored(tmp_path, dataset):
    alphagenome.score(dataset, tmp_path / "ag.tsv", FakeAPI(extra_track_from_call=2), chunk=500)
    table = pd.read_csv(tmp_path / "ag.tsv", sep="\t")

    assert (table["CAGE|late"].iloc[:1000] == 0).all()
    assert (table["CAGE|late"].iloc[1000:] == 1.0).all()


def test_the_table_feeds_build_embedding_oriented_to_the_counted_allele(tmp_path, dataset):
    alphagenome.score(dataset, tmp_path / "ag.tsv", FakeAPI(), chunk=1000)
    dietnet("partition", dataset=dataset, out=tmp_path / "p.npz")
    spec = {"blocks": [{"provider": "precomputed", "path": str(tmp_path / "ag.tsv"),
                        "orient": "effect", "columns": ["DNASE|t0"]}]}  # fmt: skip

    values, meta = features.build(spec, dataset, tmp_path / "p.npz", tmp_path / "e.npz")

    snps = ds.snp_table(dataset)
    variants = alphagenome.variants_of(dataset)
    effect = FakeAPI.effect(variants["pos"], 0).to_numpy()
    sign = np.where(snps["a1"] == variants["alt"], 1.0, -1.0)
    covered = values[:, -1] == 0  # the missing flag: ambiguous SNPs have no sign
    np.testing.assert_allclose(values[covered, 0], (effect * sign)[covered])
    assert meta["blocks"][0]["negated"] > 0


def test_the_api_backend_needs_a_key(monkeypatch):
    pytest.importorskip("alphagenome")
    monkeypatch.delenv(alphagenome.API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=alphagenome.API_KEY_ENV):
        alphagenome.AlphaGenomeAPI()


def test_the_api_backend_maps_scores_back_to_snps(monkeypatch):
    """AlphaGenomeAPI.score with the client's network calls replaced.

    Real Variant objects, a fake score_variants and tidy_scores: checks the
    glue -- requests in 1-based chr coordinates at the chosen length, and each
    tidy row mapped back to the SNP it scores.
    """
    pytest.importorskip("alphagenome")
    from alphagenome.models import dna_client, variant_scorers

    seen = {}

    class Client:
        def score_variants(self, intervals, variants, scorers, **kwargs):
            seen.update(intervals=intervals, variants=variants, scorers=scorers, **kwargs)
            return [[v] for v in variants]

    def tidy(results):
        return pd.DataFrame(
            [{"variant_id": v, "output_type": "DNASE", "track_name": "t", "raw_score": v.position / 1e6}
             for (v,) in results]
        )  # fmt: skip

    monkeypatch.setattr(dna_client, "create", lambda key: Client())
    monkeypatch.setattr(variant_scorers, "tidy_scores", tidy)
    api = alphagenome.AlphaGenomeAPI(scorers=["DNASE"], sequence_length="16KB", api_key="k")

    variants = pd.DataFrame(
        {"chrom": ["chr1", "chr2"], "pos": [858952, 1000], "ref": ["G", "A"], "alt": ["A", "T"]}
    )
    scored = api.score(variants)

    assert [str(v) for v in seen["variants"]] == ["chr1:858952:G>A", "chr2:1000:A>T"]
    assert all(i.width == 2**14 for i in seen["intervals"])
    assert list(scored["pos"]) == [858952, 1000] and list(scored["alt"]) == ["A", "T"]
    np.testing.assert_allclose(scored["raw_score"], [0.858952, 0.001])
