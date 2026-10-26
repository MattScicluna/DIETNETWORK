"""Matching a cohort to a model's SNPs: allele orientation, strand, missing SNPs.

Each test writes a variant of the kgp_mini cohort -- alleles swapped, strand
complemented, SNPs dropped -- and checks the matched dosages come back as the
original's, which is what the model was trained on.
"""

import h5py
import numpy as np
import pandas as pd
import pytest
from pyplink import PyPlink

from dietnet import align
from dietnet import dataset as ds
from dietnet.plink import Bed, read_bim

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def write_cohort(prefix, dosages, bim, fam_src):
    """A PLINK fileset; dosages count each row's A1 (as pyplink writes them)."""
    with PyPlink(str(prefix), "w") as bed:
        for j in range(dosages.shape[1]):
            bed.write_genotypes(dosages[:, j])
    bim.to_csv(f"{prefix}.bim", sep="\t", header=False, index=False)
    with open(f"{fam_src}.fam") as src, open(f"{prefix}.fam", "w") as dst:
        dst.write(src.read())


@pytest.fixture
def model(tmp_path, kgp_mini):
    """The model's view: SNP table and genotypes of the training data."""
    path = ds.build(kgp_mini.plink_prefix, tmp_path / "train.h5", class_labels=kgp_mini.labels)
    with h5py.File(path) as f:
        genotypes = np.array(f["inputs"])
    return ds.snp_table(path), genotypes


def raw_bim(prefix):
    return pd.read_csv(f"{prefix}.bim", sep="\t", header=None)


# Small enough that every fixture is written in many sample blocks, the last
# one partial: the path a biobank takes.
SMALL_BLOCKS = 37_000


def matched(tmp_path, model_snps, cohort):
    out = align.write_matched(cohort, model_snps, tmp_path / "matched.h5", block_bytes=SMALL_BLOCKS)
    with h5py.File(out) as f:
        return np.array(f["inputs"]), float(f["scale"][0])


def test_the_training_cohort_matches_itself(tmp_path, model, kgp_mini):
    snps, genotypes = model
    a = align.align(snps, kgp_mini.plink_prefix)

    assert a.counts["same"] == len(snps) and a.counts["missing"] == 0 and a.scale == 1.0
    inputs, scale = matched(tmp_path, snps, kgp_mini.plink_prefix)
    np.testing.assert_array_equal(inputs, genotypes)


def test_swapped_alleles_are_flipped_back(tmp_path, model, kgp_mini):
    """The cohort counts the other allele: every dosage is 2 - x there."""
    snps, genotypes = model
    bim = raw_bim(kgp_mini.plink_prefix)
    bim[[4, 5]] = bim[[5, 4]].to_numpy()
    flipped = np.where(genotypes >= 0, 2 - genotypes, -1).astype(np.int8)
    write_cohort(tmp_path / "swapped", flipped, bim, kgp_mini.plink_prefix)

    ambiguous = (snps["a1"].str.translate(COMPLEMENT) == snps["a2"]).to_numpy()
    a = align.align(snps, tmp_path / "swapped")
    assert a.counts["swapped"] == (~ambiguous).sum()

    inputs, _ = matched(tmp_path, snps, tmp_path / "swapped")
    np.testing.assert_array_equal(inputs[:, ~ambiguous], genotypes[:, ~ambiguous])
    # A swapped A/T or C/G SNP is indistinguishable from the other strand: dropped.
    assert (inputs[:, ambiguous] == -1).all()


def test_other_strand_is_recognised_except_for_ambiguous_snps(tmp_path, model, kgp_mini):
    snps, genotypes = model
    bim = raw_bim(kgp_mini.plink_prefix)
    bim[4] = bim[4].str.translate(COMPLEMENT)
    bim[5] = bim[5].str.translate(COMPLEMENT)
    write_cohort(tmp_path / "strand", genotypes, bim, kgp_mini.plink_prefix)

    ambiguous = (snps["a1"].str.translate(COMPLEMENT) == snps["a2"]).to_numpy()
    inputs, _ = matched(tmp_path, snps, tmp_path / "strand")
    np.testing.assert_array_equal(inputs[:, ~ambiguous], genotypes[:, ~ambiguous])
    assert (inputs[:, ambiguous] == -1).all()


def test_missing_snps_are_missing_and_set_the_published_scale(tmp_path, model, kgp_mini):
    """D / (D - |C|): 2,000 model SNPs, 1,400 in the cohort."""
    snps, genotypes = model
    keep = np.sort(np.random.default_rng(0).choice(len(snps), 1400, replace=False))
    bim = raw_bim(kgp_mini.plink_prefix).iloc[keep]
    write_cohort(tmp_path / "partial", genotypes[:, keep], bim, kgp_mini.plink_prefix)

    inputs, scale = matched(tmp_path, snps, tmp_path / "partial")
    assert scale == pytest.approx(2000 / 1400)
    np.testing.assert_array_equal(inputs[:, keep], genotypes[:, keep])
    dropped = np.setdiff1d(np.arange(len(snps)), keep)
    assert (inputs[:, dropped] == -1).all()


def test_cohort_order_and_extra_snps_do_not_matter(tmp_path, model, kgp_mini):
    """A cohort with its SNPs in another order, plus SNPs the model never saw."""
    snps, genotypes = model
    order = np.random.default_rng(1).permutation(len(snps))
    bim = raw_bim(kgp_mini.plink_prefix).iloc[order].reset_index(drop=True)
    extra = bim.iloc[:10].copy()
    extra[3] = extra[3] + 1  # positions no model SNP has
    extra[1] = extra[1] + "_extra"
    write_cohort(
        tmp_path / "shuffled",
        np.concatenate([genotypes[:, order], genotypes[:, order[:10]]], axis=1),
        pd.concat([bim, extra]),
        kgp_mini.plink_prefix,
    )

    inputs, scale = matched(tmp_path, snps, tmp_path / "shuffled")
    assert scale == 1.0
    np.testing.assert_array_equal(inputs, genotypes)


@pytest.mark.parametrize("load_bytes", [2**36, 0], ids=["in-memory", "memory-mapped"])
@pytest.mark.parametrize("block_bytes", [SMALL_BLOCKS, 2**32])
def test_bed_reader_matches_pyplink(kgp_mini, block_bytes, load_bytes):
    """The .bed decoder against pyplink: arbitrary SNP rows, any block size, either read path."""
    bed = Bed.open(kgp_mini.plink_prefix)
    rows = [5, 0, 1999, 17, 5]
    with PyPlink(str(kgp_mini.plink_prefix)) as reference:
        names = read_bim(kgp_mini.plink_prefix)["snp"].to_numpy()[rows]
        expected = np.stack([reference.get_geno_marker(n) for n in names], axis=1)
    blocks = list(bed.sample_blocks(rows, block_bytes, load_bytes))
    assert [start for start, _ in blocks] == list(range(0, bed.n_samples, len(blocks[0][1])))
    np.testing.assert_array_equal(np.concatenate([b for _, b in blocks]), expected)


def test_bed_is_read_whole_in_chunks(kgp_mini):
    """The page-cache-dropping reader returns the file's bytes, whatever the chunk size."""
    from dietnet.plink import _read_uncached

    path = f"{kgp_mini.plink_prefix}.bed"
    expected = np.fromfile(path, dtype=np.uint8, offset=3)
    got = _read_uncached(path, offset=3, size=len(expected), chunk=4099)
    np.testing.assert_array_equal(got, expected)


def test_blocks_are_sized_by_the_model_width_not_the_matched_snps(
    tmp_path, model, kgp_mini, monkeypatch
):
    """A cohort holding few of a wide model's SNPs: blocks still fit block_bytes.

    Budgeting by matched SNPs instead let one UK Biobank block span 184,808
    samples x 229,986 model SNPs (42 GB) when only 5,810 SNPs matched.
    """
    snps, genotypes = model
    keep = np.arange(0, len(snps), 40)  # 50 of 2,000 SNPs
    bim = raw_bim(kgp_mini.plink_prefix).iloc[keep]
    write_cohort(tmp_path / "sparse", genotypes[:, keep], bim, kgp_mini.plink_prefix)

    widths = []
    original = Bed.sample_blocks

    def spy(self, *args, **kwargs):
        for start, block in original(self, *args, **kwargs):
            widths.append(len(block) * len(snps))
            yield start, block

    monkeypatch.setattr(Bed, "sample_blocks", spy)
    inputs, scale = matched(tmp_path, snps, tmp_path / "sparse")

    assert max(widths) <= SMALL_BLOCKS + 4 * len(snps)
    assert scale == pytest.approx(2000 / 50)
    np.testing.assert_array_equal(inputs[:, keep], genotypes[:, keep])
