# Tests

```bash
uv run pytest                                  # everything that needs no network
uv run pytest -m "not integration"             # unit + parity only (~2 s)
DIETNET_UPSTREAM_DIR=/path/to/upstream uv run pytest -m upstream
```

| Directory | What it pins |
|---|---|
| `parity/` | Equality with HussinLab/DIETNETWORK at `UPSTREAM_SHA` (`tests/support.py`): dosage matrix, input statistics, imputation, embedding, forward pass, one training epoch, best-model rule. |
| `integration/` | The CLI pipeline end to end on both fixtures, with an accuracy floor. |
| `unit/` | Known bugs (`test_known_bugs.py`) and the sensitive-files guard. |

## Upstream checkout

The parity tests import upstream's code live instead of comparing against
stored outputs. Dropout masks and Adam updates depend on the installed torch,
so stored outputs would fail on a torch upgrade for reasons unrelated to this
code. Get the pinned commit:

```bash
git init upstream && cd upstream
git fetch --depth 1 https://github.com/HussinLab/DIETNETWORK.git 5481556e472140c6d3a67363c1286f22109fb1ce
git checkout FETCH_HEAD
```

Without `DIETNET_UPSTREAM_DIR` those tests skip. CI always sets it.

## Known bugs

Each bug in `docs/refactor-plan.md` that a test can show is a `strict` xfail
whose reason starts with its id (`B4: ...`). Strict means a fix that makes it
pass fails the run until the marker is removed, so the markers stay an accurate
list of what is still broken. `uv run pytest --runxfail` shows each one failing
for its stated reason.

## Fixtures

`fixtures/build_fixtures.py` documents and rebuilds them. It needs the 1KGP test
data, plink 1.9 and an upstream checkout. Its docstring has the command.

- `kgp_mini`: 480 samples (20 from each of 24 populations) × 2,000 SNPs of
  public 1KGP data. No missing calls (imputed).
- `synthetic`: 200 × 500, 4 classes, 3% missing calls, plus a trait for
  regression in `synthetic_trait.tsv`.
- `*_upstream.hdf5`: what upstream's `create_dataset.py` makes from each, via
  `plink --recode A`. This is the reference for the dosage-parity test.
