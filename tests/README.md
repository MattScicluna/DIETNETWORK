# Tests

```bash
uv run pytest                                   # all but slow/network; parity tests skip without upstream
DIETNET_UPSTREAM_DIR=/path/to/upstream uv run pytest
DIETNET_UPSTREAM_DIR=... uv run pytest -m slow  # published models vs upstream inference.py (downloads Zenodo)
```

| Directory | What it pins |
|---|---|
| `unit/test_upstream_unchanged.py` | The upstream files this package runs are byte-identical to the commit in `src/dietnet/upstream_manifest.json`, which is also the commit CI clones. |
| `parity/test_dataset.py` | `dietnet create-dataset` writes exactly the HDF5 upstream's `create_dataset.py` writes. |
| `parity/test_predict.py` | `dietnet predict` scores equal upstream's `test_independent_dataset.py`: on a matched cohort, and with upstream's own SNP matching on a cohort missing 30% of SNPs. The published models' scores equal `Dietnet_inference/inference.py` (slow). |
| `unit/test_align.py` | Allele-aware matching: swapped alleles, other strand, ambiguous SNPs, missing SNPs and the D/(D−\|C\|) scale. |
| `integration/` | The CLI pipeline end to end on both fixtures, with accuracy floors, regression, recorded preprocessing and strict loading. |

## Upstream checkout

This repository keeps only the upstream files its API runs. The parity tests
compare it with upstream scripts it removed, so they need the full tree at the
pinned commit:

```bash
git init upstream && cd upstream
git fetch --depth 1 https://github.com/HussinLab/DIETNETWORK.git 5481556e472140c6d3a67363c1286f22109fb1ce
git checkout FETCH_HEAD
```

CI does this and sets `DIETNET_UPSTREAM_DIR`. Set `DIETNET_PUBLISHED_DIR` to
reuse a downloaded Zenodo release across slow runs.

## Fixtures

`fixtures/build_fixtures.py` documents and rebuilds them. It needs the 1KGP test
data, plink 1.9 and an upstream checkout; its docstring gives the command.

- `kgp_mini`: 480 samples (20 from each of 24 populations) × 2,000 SNPs of
  public 1KGP data.
- `synthetic`: 200 × 500, 4 classes, 3% missing calls, plus a trait in
  `synthetic_trait.tsv`.
- `*_upstream.hdf5`: what upstream's `create_dataset.py` makes from each via
  `plink --recode A`, with plink's `<id>_<A1>` SNP names, as in the published dataset.
