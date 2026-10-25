"""Rebuild the committed test fixtures. Not run by the test suite.

Two datasets, each committed as PLINK files *and* as the HDF5 file that
HussinLab/DIETNETWORK's own ``create_dataset.py`` makes from them:

``kgp_mini``
    Real data: 20 samples from each of the 24 populations of the 1KGP test set
    (``Dietnet.pretrained_models.TEST_DATA['1kgp_default']``, public) and 2,000 of
    its SNPs. Imputed, so it has no missing calls.

``synthetic``
    200 samples in 4 classes over 500 SNPs, with class-specific allele
    frequencies so a model can learn something, 3% of calls missing, and a
    quantitative trait for the regression task.

The HDF5 files are the point. Upstream never reads PLINK: it reads a genotype
table made by ``plink --recode A``. Building that table here, with plink 1.9 as
upstream did, and committing the result lets ``tests/parity`` check that this
repository's PLINK reader produces the same dosage matrix without plink being
installed wherever the tests run.

Usage::

    python tests/fixtures/build_fixtures.py \\
        --kgp-prefix /path/to/1000G...phased_imputed_V1 \\
        --kgp-labels /path/to/labels_pop_subsampleV1.tsv \\
        --plink /path/to/plink1.9 \\
        --upstream-dir /path/to/HussinLab/DIETNETWORK
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

KGP_PER_POPULATION = 20
KGP_N_SNPS = 2000
SYN_PER_CLASS = 50
SYN_N_CLASSES = 4
SYN_N_SNPS = 500
SYN_MISSING_RATE = 0.03
SEED = 20261025


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def build_kgp_mini(kgp_prefix: Path, kgp_labels: Path, plink: Path, work: Path) -> Path:
    rng = np.random.default_rng(SEED)

    labels = pd.read_csv(kgp_labels, sep="\t")
    sample_col, label_col = labels.columns[:2]
    keep = labels.groupby(label_col).sample(n=KGP_PER_POPULATION, random_state=SEED)
    keep = keep.sort_values(sample_col)

    fam = pd.read_csv(f"{kgp_prefix}.fam", sep=r"\s+", header=None)
    fam = fam[fam[1].isin(keep[sample_col])]
    (work / "keep.txt").write_text("".join(f"{fid}\t{iid}\n" for fid, iid in zip(fam[0], fam[1])))

    bim = pd.read_csv(f"{kgp_prefix}.bim", sep="\t", header=None)
    snps = np.sort(rng.choice(len(bim), size=KGP_N_SNPS, replace=False))
    (work / "extract.txt").write_text("".join(f"{s}\n" for s in bim[1].iloc[snps]))

    out = HERE / "kgp_mini"
    run(
        [
            plink,
            "--bfile",
            kgp_prefix,
            "--keep",
            work / "keep.txt",
            "--extract",
            work / "extract.txt",
            "--make-bed",
            "--out",
            out,
        ]
    )
    keep[[sample_col, label_col]].to_csv(HERE / "kgp_mini_labels.tsv", sep="\t", index=False)
    return out


def build_synthetic() -> Path:
    from pyplink import PyPlink

    rng = np.random.default_rng(SEED)
    n = SYN_PER_CLASS * SYN_N_CLASSES

    # Allele frequencies drawn around a shared ancestral frequency, as in the
    # Balding-Nichols model, so classes differ by a controllable amount.
    ancestral = rng.uniform(0.1, 0.9, size=SYN_N_SNPS)
    fst = 0.1
    a = ancestral * (1 - fst) / fst
    b = (1 - ancestral) * (1 - fst) / fst
    freqs = rng.beta(a, b, size=(SYN_N_CLASSES, SYN_N_SNPS))

    classes = np.repeat(np.arange(SYN_N_CLASSES), SYN_PER_CLASS)
    genotypes = rng.binomial(2, freqs[classes]).astype(np.int8)

    effects = rng.normal(0, 0.1, size=SYN_N_SNPS)
    trait = genotypes @ effects + rng.normal(0, 1, size=n)

    genotypes[rng.random(genotypes.shape) < SYN_MISSING_RATE] = -1

    samples = [f"SYN{i:04d}" for i in range(n)]
    out = HERE / "synthetic"
    with PyPlink(str(out), "w") as bed:
        for j in range(SYN_N_SNPS):
            bed.write_genotypes(genotypes[:, j])

    with open(f"{out}.fam", "w") as f:
        for s in samples:
            f.write(f"{s} {s} 0 0 0 -9\n")
    with open(f"{out}.bim", "w") as f:
        for j in range(SYN_N_SNPS):
            pos = 1000 * (j + 1)
            f.write(f"1\tchr1:{pos}:G:A\t0\t{pos}\tA\tG\n")

    pd.DataFrame({"Sample": samples, "Population": [f"POP{c}" for c in classes]}).to_csv(
        HERE / "synthetic_labels.tsv", sep="\t", index=False
    )
    pd.DataFrame({"Sample": samples, "Trait": np.round(trait, 6)}).to_csv(
        HERE / "synthetic_trait.tsv", sep="\t", index=False
    )
    return out


def upstream_hdf5(
    prefix: Path,
    labels: Path,
    plink: Path,
    upstream_dir: Path,
    work: Path,
    regression_labels: Path | None = None,
) -> None:
    """Build ``<prefix>_upstream.hdf5`` the way upstream's README describes."""
    raw = work / prefix.name
    run([plink, "--bfile", prefix, "--recode", "A", "--real-ref-alleles", "--out", raw])

    # .raw columns: FID IID PAT MAT SEX PHENOTYPE then one '<snp>_<counted allele>'
    # column per SNP. Upstream's table is the sample id, then one column per SNP.
    table = pd.read_csv(f"{raw}.raw", sep=r"\s+")
    snp_cols = table.columns[6:]
    # Nullable integers: a column holding a missing call would otherwise be read
    # as float and written as "0.0", which upstream's int() parser rejects.
    genotypes = table[["IID", *snp_cols]].astype({c: "Int8" for c in snp_cols})
    genotypes.columns = ["ID", *[c.rsplit("_", 1)[0] for c in snp_cols]]
    genotypes_tsv = work / f"{prefix.name}_genotypes.tsv"
    genotypes.to_csv(genotypes_tsv, sep="\t", index=False, na_rep="NA")

    cmd = [
        sys.executable,
        "create_dataset.py",
        "--exp-path",
        HERE,
        "--genotypes",
        genotypes_tsv,
        "--class-labels",
        labels,
        "--out",
        f"{prefix.name}_upstream",
    ]
    if regression_labels is not None:
        cmd += ["--regression-labels", regression_labels]
    run(cmd, cwd=upstream_dir / "Dietnet_train_and_evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kgp-prefix", type=Path, required=True)
    parser.add_argument("--kgp-labels", type=Path, required=True)
    parser.add_argument("--plink", type=Path, required=True, help="plink 1.9, as upstream used")
    parser.add_argument("--upstream-dir", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        kgp = build_kgp_mini(args.kgp_prefix, args.kgp_labels, args.plink, work)
        upstream_hdf5(kgp, HERE / "kgp_mini_labels.tsv", args.plink, args.upstream_dir, work)

        syn = build_synthetic()
        upstream_hdf5(
            syn,
            HERE / "synthetic_labels.tsv",
            args.plink,
            args.upstream_dir,
            work,
            regression_labels=HERE / "synthetic_trait.tsv",
        )

    for leftover in HERE.glob("*.log"):
        leftover.unlink()
    for leftover in HERE.glob("*.nosex"):
        leftover.unlink()


if __name__ == "__main__":
    main()
