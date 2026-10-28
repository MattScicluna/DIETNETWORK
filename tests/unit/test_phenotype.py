"""A trait adjusted for covariates and standardised (dietnet.phenotype, adjust-phenotype)."""

import json

import numpy as np
import pandas as pd
import pytest
from support import dietnet

from dietnet import phenotype


@pytest.fixture
def table():
    """Height with a sex effect (~13 cm), an age effect, a centre effect and noise."""
    rng = np.random.default_rng(0)
    n = 2000
    sex = rng.integers(0, 2, n)
    age = rng.uniform(40, 70, n)
    centre = rng.choice(["Leeds", "Oxford", "Glasgow"], n)
    noise = rng.normal(0, 6, n)
    height = 162 + 13 * sex - 0.1 * age + np.where(centre == "Glasgow", -2.0, 0.0) + noise
    t = pd.DataFrame(
        {"height": height, "sex": sex, "age": age, "centre": centre},
        index=pd.Index([f"s{i}" for i in range(n)], name="eid"),
    )
    t.loc["s0", "height"] = np.nan  # missing trait
    t.loc["s1", "age"] = np.nan  # missing covariate
    return t.astype(str).replace("nan", np.nan), noise


def test_the_residual_is_standardised_and_free_of_the_covariates(table):
    t, noise = table

    adjusted, summary = phenotype.adjust(t, "height", ["sex", "age", "centre"])

    assert len(adjusted) == len(t) - 2 and summary["dropped_missing"] == 2
    assert {"s0", "s1"}.isdisjoint(adjusted.index)
    assert abs(adjusted.mean()) < 1e-9 and abs(adjusted.std(ddof=1) - 1) < 1e-9
    kept = t.loc[adjusted.index]
    for column in ("sex", "age"):
        assert abs(np.corrcoef(adjusted, kept[column].astype(float))[0, 1]) < 1e-9
    # What is left is the noise, up to scale.
    assert np.corrcoef(adjusted, noise[2:])[0, 1] > 0.999
    assert summary["covariate_r2"] > 0.5
    assert "centre=Leeds" in summary["coefficients"]  # a text covariate became indicators


def test_unknown_columns_are_named(table):
    with pytest.raises(ValueError, match="height2"):
        phenotype.adjust(table[0], "height2", ["sex"])


def test_the_command_writes_regression_labels(tmp_path, table):
    t, _ = table
    pheno = tmp_path / "pheno.tsv"
    t.to_csv(pheno, sep="\t")

    out = dietnet(
        "adjust-phenotype",
        phenotypes=pheno,
        trait="height",
        covariates="sex,age",
        out=tmp_path / "h.tsv",
    )

    labels = pd.read_csv(tmp_path / "h.tsv", sep="\t")
    assert list(labels.columns) == ["sample", "height"] and len(labels) == len(t) - 2
    assert json.loads((tmp_path / "h.tsv.json").read_text())["covariates"] == ["sex", "age"]
    assert "covariates explain R^2" in out
