"""Corner cases of AlphaGenome scoring, without the network: ids that are not
``chr:pos:REF:ALT``, bad backend options, and progress reports."""

import pandas as pd
import pytest

from dietnet import alphagenome


def write_bim(path, ids):
    path.write_text("".join(f"1\t{snp}\t0\t{i + 1}\tA\tG\n" for i, snp in enumerate(ids)))
    return path


class Constant:
    """A stand-in backend: one track, score 1.0 for every variant."""

    def __init__(self):
        self.calls = []

    def score(self, variants):
        self.calls.append(len(variants))
        return variants.assign(output_type="DNASE", track_name="t", raw_score=1.0)


@pytest.mark.parametrize(
    "ids",
    [["rs1", "rs2"], ["chr1:100:G", "chr1:200:C"], ["chr1:x:G:A", "chr1:200:C:T"]],
    ids=["rs-ids", "three-fields", "non-numeric-position"],
)
def test_ids_that_are_not_chr_pos_ref_alt_are_refused(tmp_path, ids):
    """Without REF and ALT in the id there is nothing to score, so it is an error."""
    bim = write_bim(tmp_path / "x.bim", ids)
    with pytest.raises(ValueError, match="SNP ids are not chr:pos:REF:ALT"):
        alphagenome.variants_of(bim)


def test_variants_get_a_chr_prefix_whether_or_not_the_id_has_one(tmp_path):
    bim = write_bim(tmp_path / "x.bim", ["1:100:G:A", "chr2:200:C:T"])
    variants = alphagenome.variants_of(bim)
    assert variants.to_dict("list") == {
        "chrom": ["chr1", "chr2"],
        "pos": [100, 200],
        "ref": ["G", "C"],
        "alt": ["A", "T"],
    }


def test_progress_is_reported_after_each_chunk(tmp_path):
    """``progress(done, total)`` after every chunk, the last one partial."""
    bim = write_bim(tmp_path / "x.bim", [f"chr1:{p}:G:A" for p in range(100, 600, 100)])
    seen = []
    backend = Constant()

    out = alphagenome.score(
        bim, tmp_path / "ag.tsv", backend, chunk=2, progress=lambda *a: seen.append(a)
    )

    assert seen == [(2, 5), (4, 5), (5, 5)] and backend.calls == [2, 2, 1]
    table = pd.read_csv(out, sep="\t")
    assert list(table["pos"]) == [100, 200, 300, 400, 500] and (table["DNASE|t"] == 1.0).all()


def test_a_finished_run_reports_no_progress(tmp_path):
    """A rerun with every SNP already scored calls neither the backend nor ``progress``."""
    bim = write_bim(tmp_path / "x.bim", ["chr1:100:G:A", "chr1:200:G:A"])
    alphagenome.score(bim, tmp_path / "ag.tsv", Constant())

    seen, backend = [], Constant()
    alphagenome.score(bim, tmp_path / "ag.tsv", backend, progress=lambda *a: seen.append(a))
    assert seen == [] and backend.calls == []


@pytest.fixture
def offline(monkeypatch):
    """The real client library with ``dna_client.create`` replaced: no network."""
    pytest.importorskip("alphagenome")
    from alphagenome.models import dna_client

    monkeypatch.setattr(dna_client, "create", lambda key: object())


def test_the_api_backend_refuses_an_unknown_sequence_length(offline):
    with pytest.raises(ValueError, match="sequence_length must be one of"):
        alphagenome.AlphaGenomeAPI(sequence_length="2MB", api_key="k")


def test_the_api_backend_refuses_unknown_scorers(offline):
    with pytest.raises(ValueError, match=r"unknown scorers \['NOPE'\]; one of \["):
        alphagenome.AlphaGenomeAPI(scorers=["DNASE", "NOPE"], api_key="k")


def test_the_api_key_can_come_from_the_environment(offline, monkeypatch):
    monkeypatch.setenv(alphagenome.API_KEY_ENV, "from-env")
    api = alphagenome.AlphaGenomeAPI(scorers=["DNASE"], sequence_length="16KB")
    assert len(api._scorers) == 1 and api._max_workers == 5


def test_an_error_without_a_grpc_code_is_raised_at_once(monkeypatch):
    """An ordinary exception has no status, so it is not retried."""
    monkeypatch.setattr(alphagenome, "RETRY_WAIT", 0)
    calls = []

    def broken():
        calls.append(1)
        raise KeyError("bad")

    with pytest.raises(KeyError):
        alphagenome._retrying(broken)
    assert len(calls) == 1
