"""Corner cases of matching a model's SNPs to a cohort's .bim.

Only the .bim is read by :func:`dietnet.align.align`, so each test writes a
handful of .bim rows by hand and a model SNP table to match against them.
"""

import numpy as np
import pandas as pd
import pytest

from dietnet import align


def write_bim(prefix, rows):
    """A .bim from (chrom, id, pos, a1, a2) tuples."""
    lines = [f"{c}\t{snp}\t0\t{pos}\t{a1}\t{a2}" for c, snp, pos, a1, a2 in rows]
    (prefix.parent / f"{prefix.name}.bim").write_text("\n".join(lines) + "\n")
    return prefix


@pytest.fixture
def cohort(tmp_path):
    return write_bim(
        tmp_path / "cohort",
        [
            ("1", "rs1", 100, "A", "G"),
            ("1", "rs2", 200, "C", "T"),
            ("chr2", "rs3", 300, "A", "T"),  # ambiguous A/T
            ("2", "rs4", 400, "G", "C"),  # ambiguous C/G
            ("3", "rs5", 500, "A", "C"),
            ("3", "rs5b", 500, "A", "G"),  # a second variant at the same position
        ],
    )


def test_scale_refuses_an_alignment_with_no_snp_found():
    """D / (D - |C|) is undefined when no SNP was found, so ``scale`` says so."""
    nothing = align.Alignment(source=np.full(4, -1), flip=np.zeros(4, dtype=bool))
    assert nothing.n_model == 4 and nothing.n_matched == 0
    with pytest.raises(ValueError, match="no SNP of the model was found"):
        nothing.scale


def test_scale_is_model_snps_over_matched_ones():
    """Three of four SNPs found gives a scale of 4 / 3."""
    some = align.Alignment(source=np.array([0, -1, 2, 5]), flip=np.zeros(4, dtype=bool))
    assert some.scale == pytest.approx(4 / 3)


def test_snps_are_matched_by_id_when_the_model_has_no_positions(cohort):
    """Without chrom and pos the key is the SNP id, and alleles are still checked."""
    model = pd.DataFrame(
        {"snp": ["rs2", "rs1", "rs9"], "a1": ["T", "A", "A"], "a2": ["C", "G", "G"]}
    )
    a = align.align(model, cohort)

    assert a.counts["matched_by"] == "id"
    assert list(a.source) == [1, 0, -1]
    assert list(a.flip) == [True, False, False]
    assert a.counts["same"] == 1 and a.counts["swapped"] == 1 and a.counts["missing"] == 1


def test_a_missing_position_falls_back_to_matching_by_id(cohort):
    """A single unknown position is enough to match every SNP by id instead."""
    model = pd.DataFrame(
        {
            "snp": ["rs1", "rs2"],
            "chrom": ["1", "1"],
            "pos": [100.0, np.nan],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
        }
    )
    a = align.align(model, cohort)

    assert a.counts["matched_by"] == "id"
    assert list(a.source) == [0, 1] and not a.flip.any()


def test_without_a_counted_allele_snps_match_on_position_alone(cohort):
    """A model that records no A1 is matched on the key alone, unflipped, and the
    counts say how many SNPs were taken without checking their alleles."""
    model = pd.DataFrame(
        {
            "snp": ["x1", "x2", "x3", "x9"],
            "chrom": ["1", "2", "3", "9"],
            "pos": [200, 300, 500, 900],
        }
    )
    a = align.align(model, cohort)

    # The first of the two cohort rows at 3:500 is taken.
    assert list(a.source) == [1, 2, 4, -1]
    assert not a.flip.any()
    assert a.counts == {
        "alleles_unchecked": 3,
        "matched_by": "position",
        "model_snps": 4,
        "missing": 1,
    }


def test_an_all_missing_a1_column_counts_as_no_counted_allele(cohort):
    """An ``a1`` column with nothing in it is the same as none, matched by id here."""
    model = pd.DataFrame({"snp": ["rs4", "rs1"], "a1": [None, None]})
    a = align.align(model, cohort)

    assert list(a.source) == [3, 0]
    assert a.counts["alleles_unchecked"] == 2 and a.counts["matched_by"] == "id"


def test_without_a_counted_allele_nothing_found_gives_no_scale(cohort):
    """No SNP in common: every model SNP is missing and the scale is refused."""
    model = pd.DataFrame({"snp": ["nope1", "nope2"]})
    a = align.align(model, cohort)

    assert a.counts["alleles_unchecked"] == 0 and a.counts["missing"] == 2
    with pytest.raises(ValueError, match="no SNP"):
        a.scale


def test_ambiguous_snps_are_kept_only_on_an_outright_match(cohort):
    """An A/T or C/G SNP is copied when its alleles match as they are, and missing
    when they are swapped, since a swap cannot be told from the other strand."""
    model = pd.DataFrame(
        {
            "snp": ["at_same", "cg_swapped"],
            "chrom": ["2", "2"],
            "pos": [300, 400],
            "a1": ["A", "C"],
            "a2": ["T", "G"],
        }
    )
    a = align.align(model, cohort)

    assert list(a.source) == [2, -1] and not a.flip.any()
    assert a.counts["ambiguous_kept"] == 1
    assert a.counts["found_with_other_alleles"] == 1 and a.counts["missing"] == 1


def test_strand_flips_are_copied_or_flipped(cohort):
    """Complemented alleles copy as they are; complemented and swapped ones flip."""
    model = pd.DataFrame(
        {
            "snp": ["strand", "strand_swapped"],
            "chrom": ["1", "1"],
            "pos": [100, 200],
            "a1": ["T", "A"],  # cohort A/G -> T/C; cohort C/T -> G/A, swapped
            "a2": ["C", "G"],
        }
    )
    a = align.align(model, cohort)

    assert list(a.source) == [0, 1] and list(a.flip) == [False, True]
    assert a.counts["strand"] == 1 and a.counts["strand_swapped"] == 1


def test_a_multiallelic_position_takes_the_variant_whose_alleles_fit(cohort):
    """Of two cohort rows at 3:500, the one with the model's alleles is taken."""
    model = pd.DataFrame({"snp": ["m"], "chrom": ["3"], "pos": [500], "a1": ["G"], "a2": ["A"]})
    a = align.align(model, cohort)

    assert list(a.source) == [5] and list(a.flip) == [True]


def test_a_model_without_a2_checks_only_the_counted_allele(cohort):
    """With no ``a2`` column, A1 alone decides: equal copies, equal to the cohort's A2 flips."""
    model = pd.DataFrame(
        {"snp": ["a", "b"], "chrom": ["1", "1"], "pos": [100, 200], "a1": ["A", "C"]}
    )
    a = align.align(model, cohort)
    assert list(a.source) == [0, 1] and not a.flip.any()

    swapped = model.assign(a1=["G", "T"])
    b = align.align(swapped, cohort)
    assert list(b.source) == [0, 1] and b.flip.all()


def test_the_fingerprint_changes_with_the_snps_or_the_cohort(tmp_path, cohort):
    """The same SNP names and cohort files give the same id; either changing changes it."""
    for ext in (".fam", ".bed"):
        (tmp_path / f"cohort{ext}").write_bytes(b"x" * 8)
    snps = pd.DataFrame({"name": ["rs1_A", "rs2_C"]})

    first = align.fingerprint(snps, cohort)
    assert first == align.fingerprint(snps, cohort) and len(first) == 16
    assert align.fingerprint(snps.iloc[:1], cohort) != first
    (tmp_path / "cohort.bed").write_bytes(b"x" * 9)
    assert align.fingerprint(snps, cohort) != first
