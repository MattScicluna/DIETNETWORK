"""The eigenvalue spectrum of a feature table, read in chunks with little memory.

``features._pca`` runs a full SVD (hundreds of GB on the HM3+ table) and keeps only the
components. This streams the table once to build the tracks' covariance matrix, which is
small (tracks x tracks), and eigendecomposes it. It reports how many components reach a
share of the variance, on the covariance as ``_pca`` uses it (centred only) and on the
correlation (each track also scaled to unit variance), and each assay's share of the
total variance, since large-valued tracks dominate the covariance. All table rows count,
including any the cohort's alleles would not match.

    python pca_spectrum.py --table alphagenome/alphagenome_merged.tsv.gz \
        --out alphagenome/spectrum
"""

import argparse
import json

import numpy as np
import pandas as pd

KEYS = ("chrom", "pos", "ref", "alt")
SHARES = (0.8, 0.9, 0.95, 0.99)
COMPONENTS = (16, 64, 256, 1024)


def covariance(table, chunk_rows):
    """The tracks' names, means and covariance matrix, one chunk in memory at a time."""
    columns, n, total, cross = None, 0, None, None
    for chunk in pd.read_csv(
        table, sep="\t", usecols=lambda c: c not in KEYS, dtype=np.float32, chunksize=chunk_rows
    ):
        x = chunk.to_numpy(dtype=np.float64)
        if np.isnan(x).any():
            raise ValueError(f"{table}: NaN feature values")
        if columns is None:
            columns = list(chunk.columns)
            total, cross = np.zeros(x.shape[1]), np.zeros((x.shape[1], x.shape[1]))
        n += len(x)
        total += x.sum(axis=0)
        cross += x.T @ x
    return columns, n, total / n, (cross - np.outer(total, total) / n) / (n - 1)


def spectrum(matrix):
    """Eigenvalues (largest first, rounding below zero removed), eigenvectors in columns."""
    values, vectors = np.linalg.eigh(matrix)
    return np.clip(values[::-1], 0, None), vectors[:, ::-1]


def summary(values) -> dict:
    cumulative = np.cumsum(values) / values.sum()
    return {
        "total_variance": float(values.sum()),
        "components_for_share": {str(s): int(np.searchsorted(cumulative, s) + 1) for s in SHARES},
        "share_of_components": {
            str(k): float(cumulative[k - 1]) for k in COMPONENTS if k <= len(values)
        },
        "top_10_shares": (values[:10] / values.sum()).round(6).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", required=True)
    parser.add_argument("--out", required=True, help="Writes <out>.json and <out>.npz.")
    parser.add_argument("--chunk-rows", type=int, default=20000)
    args = parser.parse_args()

    columns, n, means, cov = covariance(args.table, args.chunk_rows)
    sds = np.sqrt(np.clip(np.diag(cov), 0, None))
    constant = sds == 0
    scale = np.where(constant, 1.0, sds)
    correlation = cov / np.outer(scale, scale)
    correlation[constant, :] = 0.0
    correlation[:, constant] = 0.0

    cov_values, cov_vectors = spectrum(cov)
    cor_values, cor_vectors = spectrum(correlation)

    variance = pd.Series(sds**2, index=columns)
    assay = variance.groupby(variance.index.str.split("|").str[0]).sum() / variance.sum()
    result = {
        "rows": n,
        "tracks": len(columns),
        "constant_tracks": int(constant.sum()),
        "covariance": summary(cov_values),
        "correlation": summary(cor_values),
        "variance_share_by_assay": assay.sort_values(ascending=False).round(6).to_dict(),
    }
    with open(f"{args.out}.json", "w") as f:
        json.dump(result, f, indent=2)
    # The eigenvectors and means allow projecting onto any number of components later.
    np.savez(
        f"{args.out}.npz",
        columns=np.array(columns),
        means=means,
        sds=sds,
        covariance_values=cov_values,
        covariance_vectors=cov_vectors,
        correlation_values=cor_values,
        correlation_vectors=cor_vectors,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
