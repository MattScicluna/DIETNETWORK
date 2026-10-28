"""A trait adjusted for covariates and standardised, as a regression label.

Height, for instance, depends strongly on sex, which a model of autosomal
genotypes cannot see, and on age. Training on the raw trait would ask the model
to explain variance it has no access to. The usual target is instead the
residual of a linear model of the trait on such covariates, standardised:

    trait ~ 1 + covariates     ->   residual / sd(residual)

which is what :func:`adjust` computes. Numeric covariates enter as they are;
text ones (e.g. assessment centre) as one indicator column per level. The
covariate model is fitted on every sample given: with a handful of parameters
and hundreds of thousands of samples, what it learns from a test fold is
negligible, and this is the common practice for polygenic scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def adjust(table: pd.DataFrame, trait: str, covariates: list[str]) -> tuple[pd.Series, dict]:
    """Standardised residuals of ``trait`` on ``covariates`` (plus an intercept).

    ``table`` is indexed by sample id. Samples missing the trait or a covariate
    are left out. Returns the adjusted trait and a summary (samples used and
    dropped, the covariates' R^2).
    """
    unknown = [c for c in [trait, *covariates] if c not in table.columns]
    if unknown:
        raise ValueError(f"columns not in the phenotype table: {unknown}")
    used = table[[trait, *covariates]].dropna()
    y = pd.to_numeric(used[trait], errors="raise").to_numpy(dtype=float)

    design = [np.ones(len(used))]
    names = ["intercept"]
    for c in covariates:
        values = pd.to_numeric(used[c], errors="coerce")
        if values.notna().all():
            design.append(values.to_numpy(dtype=float))
            names.append(c)
        else:  # a categorical covariate: indicators for all levels but the first
            levels = sorted(used[c].astype(str).unique())
            for level in levels[1:]:
                design.append((used[c].astype(str) == level).to_numpy(dtype=float))
                names.append(f"{c}={level}")
    x = np.column_stack(design)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    sd = residual.std(ddof=1)
    if not sd > 0:
        raise ValueError(f"{trait} has no variance left after the covariates")

    summary = {
        "trait": trait,
        "covariates": covariates,
        "samples": len(used),
        "dropped_missing": len(table) - len(used),
        "covariate_r2": float(1 - residual.var() / y.var()),
        "coefficients": dict(zip(names, beta.round(6).tolist())),
    }
    return pd.Series(residual / sd, index=used.index, name=trait), summary


def read_table(path) -> pd.DataFrame:
    """A phenotype TSV (header row, sample id first) indexed by sample id, as text."""
    table = pd.read_csv(path, sep="\t", dtype=str)
    return table.set_index(table.columns[0])
