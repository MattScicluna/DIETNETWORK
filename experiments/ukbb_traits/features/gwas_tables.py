"""Per-fold GWAS feature tables for `build-embedding`'s precomputed provider.

From each fold's GWAS on its training samples (regenie's columns: CHROM GENPOS ID
ALLELE0 ALLELE1 ... BETA SE, BETA per ALLELE1), writes one table per fold with
``chrom pos ref alt z beta signed_mlog10p``: ``ref``/``alt`` are the other and the
effect allele (ALLELE0, ALLELE1), so the values describe ``alt`` and a spec reads them
with ``orient: effect``. ``signed_mlog10p`` is -log10 of the two-sided normal p-value
of z, with z's sign, computed in log space (torch's log_ndtr) so large z keeps its
value instead of underflowing.

    python gwas_tables.py --gwas 'bench/fold{fold}/gwas_marginal_{label}.regenie' \
        --labels height,bmi --folds 5 --out 'features/gwas_{label}_fold{fold}.tsv.gz'
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def signed_mlog10p(z: np.ndarray) -> np.ndarray:
    """sign(z) * -log10(2 * Phi(-|z|)), stable for any |z|."""
    log_p = math.log(2.0) + torch.special.log_ndtr(-torch.from_numpy(np.abs(z))).numpy()
    return np.sign(z) * (-log_p / math.log(10.0))


def table(gwas: Path) -> pd.DataFrame:
    g = pd.read_csv(gwas, sep=r"\s+", dtype={"CHROM": str})
    g = g[(g["TEST"] == "ADD") & np.isfinite(g["BETA"]) & (g["SE"] > 0)]
    z = (g["BETA"] / g["SE"]).to_numpy(float)
    return pd.DataFrame(
        {
            "chrom": g["CHROM"].to_numpy(),
            "pos": g["GENPOS"].to_numpy(),
            "ref": g["ALLELE0"].to_numpy(),
            "alt": g["ALLELE1"].to_numpy(),
            "z": z,
            "beta": g["BETA"].to_numpy(float),
            "signed_mlog10p": signed_mlog10p(z),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gwas", required=True, help="path with {label} and {fold}")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--out", required=True, help="path with {label} and {fold}")
    args = parser.parse_args()
    for label in args.labels.split(","):
        sizes = set()
        for fold in range(args.folds):
            t = table(Path(args.gwas.format(label=label, fold=fold)))
            out = Path(args.out.format(label=label, fold=fold))
            out.parent.mkdir(parents=True, exist_ok=True)
            t.to_csv(out, sep="\t", index=False)
            sizes.add(len(t))
            print(f"{label} fold {fold}: {len(t)} SNPs, max |z| {np.abs(t['z']).max():.1f}")
        if len(sizes) > 1:  # build-embedding needs every fold to cover the same SNPs
            raise SystemExit(f"{label}: folds cover different numbers of SNPs: {sorted(sizes)}")


if __name__ == "__main__":
    main()
