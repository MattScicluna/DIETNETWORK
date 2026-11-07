"""pca_spectrum.py: the streamed spectrum against a direct SVD of the same table."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[2] / "experiments" / "ukbb_traits" / "imputed" / "pca_spectrum.py"


@pytest.fixture
def table(tmp_path):
    """Correlated tracks on very different scales, one of them constant, as gzipped text."""
    rng = np.random.default_rng(0)
    n, tracks = 500, 12
    x = rng.normal(size=(n, 3)) @ rng.normal(size=(3, tracks)) + 0.1 * rng.normal(size=(n, tracks))
    x[:, :4] *= 100
    x[:, -1] = 7.0
    names = [f"RNA|t{i}" for i in range(6)] + [f"ATAC|t{i}" for i in range(6, tracks)]
    frame = pd.DataFrame(x, columns=names)
    keys = {"chrom": "chr1", "pos": np.arange(n), "ref": "A", "alt": "G"}
    path = tmp_path / "table.tsv.gz"
    pd.concat([pd.DataFrame(keys), frame], axis=1).to_csv(path, sep="\t", index=False)
    # The script reads float32, so the expected values use the same precision.
    return path, frame.to_numpy(dtype=np.float32).astype(np.float64)


@pytest.mark.parametrize("chunk_rows", [37, 10000])
def test_the_spectrum_matches_a_direct_svd_whatever_the_chunks(tmp_path, table, chunk_rows):
    path, x = table
    out = tmp_path / "spectrum"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--table", str(path), "--out", str(out)]
        + ["--chunk-rows", str(chunk_rows)],
        check=True,
        capture_output=True,
    )
    result = json.loads(Path(f"{out}.json").read_text())
    saved = np.load(f"{out}.npz")

    centred = x - x.mean(axis=0)
    expected = np.linalg.svd(centred, compute_uv=False) ** 2 / (len(x) - 1)
    np.testing.assert_allclose(saved["covariance_values"], expected, rtol=1e-6, atol=1e-6)

    keep = x.std(axis=0) > 0
    scaled = centred[:, keep] / centred[:, keep].std(axis=0, ddof=1)
    expected = np.linalg.svd(scaled, compute_uv=False) ** 2 / (len(x) - 1)
    np.testing.assert_allclose(saved["correlation_values"][: keep.sum()], expected, atol=1e-6)

    assert result["rows"] == 500 and result["tracks"] == 12 and result["constant_tracks"] == 1
    # The share reached by the reported number of components, and not by one fewer.
    values = saved["covariance_values"]
    k = result["covariance"]["components_for_share"]["0.95"]
    assert values[:k].sum() / values.sum() >= 0.95
    assert k == 1 or values[: k - 1].sum() / values.sum() < 0.95
    # The tracks scaled up by 100 are all RNA.
    assert result["variance_share_by_assay"]["RNA"] > 0.9


def test_refuses_missing_values(tmp_path):
    path = tmp_path / "table.tsv"
    path.write_text(
        "chrom\tpos\tref\talt\tRNA|a\tRNA|b\nchr1\t1\tA\tG\t1.0\t\nchr1\t2\tA\tG\t2.0\t3\n"
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--table", str(path), "--out", str(tmp_path / "s")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "NaN" in result.stderr
