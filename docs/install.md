# Installation

From PyPI, from the first release (0.3.0) on:

```bash
pip install dietnetwork
dietnet info
```

From a clone, with the locked versions:

```bash
git clone https://github.com/MattScicluna/DIETNETWORK && cd DIETNETWORK
uv sync --frozen          # or: pip install -e .
uv run dietnet info
```

Or straight from GitHub, without a clone:

```bash
pip install git+https://github.com/MattScicluna/DIETNETWORK
```

The wheel carries HussinLab's `Dietnet_train_and_evaluate/` inside the package
(as `dietnet/_upstream/`), byte for byte, so an installed `dietnet` runs exactly
the published training and model code.

Tested on Python 3.10–3.13, on Linux and macOS. A GPU is optional: CPU is
enough for prediction, and training uses CUDA when torch finds it.

## Checking it works

```bash
dietnet info
```

prints the package and torch versions, whether CUDA is available, and the
upstream commit it runs. It also hashes every upstream file against
`src/dietnet/upstream_manifest.json`, and exits with status 1, naming the
files, if any is modified, missing or added:

```text
dietnet 0.3.0
upstream HussinLab/DIETNETWORK 5481556e472140c6d3a67363c1286f22109fb1ce at ...
upstream files: identical to the pinned commit
torch ..., cuda available: False
```

The test suite runs the whole pipeline on small fixtures in a few minutes:

```bash
uv sync --frozen --extra dev
uv run pytest -m "not slow and not network"
```

The parity tests skip unless a full upstream checkout is given; see
[parity with upstream](parity.md).

## Extras

| extra | for |
|---|---|
| `alphagenome` | `dietnet alphagenome`: the hosted AlphaGenome client |
| `tracking` | comet-ml |
| `docs` | this site |
| `dev` | tests, formatters, pre-commit |

```bash
uv sync --frozen --extra alphagenome      # in a clone, locked versions
pip install -e '.[alphagenome]'           # in a clone, with pip
pip install 'dietnetwork[alphagenome]'                 # from PyPI
pip install 'dietnetwork[alphagenome] @ git+https://github.com/MattScicluna/DIETNETWORK'
```

## Developing on the repository

```bash
uv sync --frozen --extra dev && uv run pre-commit install
uv run pytest
```

black, isort and flake8 run at line length 100. pre-commit also runs a check for
files that should not be shared (genotypes, HDF5 datasets, `.npy`/`.npz`
arrays, `.pt` weights, cluster paths) and gitleaks. `Dietnet_train_and_evaluate/`
is upstream's code and is never edited: `tests/unit/test_upstream_unchanged.py`
fails on any change.

To work on this site:

```bash
uv sync --frozen --extra docs
uv run mkdocs serve
```

The tutorial renders without its outputs, because the build does not execute
it; the docs workflow does that in a separate step. To see them locally:

```bash
uv run jupyter nbconvert --to notebook --inplace --execute docs/tutorial.ipynb
```

(and do not commit the executed notebook).
