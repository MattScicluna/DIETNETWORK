# Installation

Here we describe how to install the `dietnetwork` package, which provides the `dietnet` command.
It needs Python 3.10 to 3.13, and is tested on Linux (all four versions) and macOS (3.12).
A GPU is optional: a CPU is enough for prediction, and `dietnet` uses a GPU when PyTorch finds one.

```bash
pip install dietnetwork
```

To check the installation, run `dietnet info`:

```text
$ dietnet info
dietnet 0.6.0
upstream HussinLab/DIETNETWORK 5481556e472140c6d3a67363c1286f22109fb1ce at ...
upstream files: identical to the pinned commit
torch 2.15.0, cuda available: False
```

The second and third lines concern the code of the published models, which is included in the package.
If any of its files has been changed, `dietnet info` names them and exits with status 1.
The last line says whether PyTorch can use a GPU.

You can now go to the [quickstart](quickstart.md).

## AlphaGenome features

To score SNPs with AlphaGenome ([SNP features](features.md#alphagenome-variant-effects)), install its client as well:

```bash
pip install 'dietnetwork[alphagenome]'
```

## The development version

The latest code can be installed from GitHub:

```bash
pip install git+https://github.com/MattScicluna/DIETNETWORK
```

To work on the code, clone the repository and install the locked versions of the dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/MattScicluna/DIETNETWORK && cd DIETNETWORK
uv sync --frozen --extra dev
uv run pre-commit install
uv run pytest -m "not slow and not network"
```

The tests run the whole pipeline on small datasets.
Before each commit, pre-commit formats the code (black and isort), checks it (flake8, with lines of at most 100 characters) and refuses files that should not be shared, such as genotypes, model weights and passwords.
The code in `Dietnet_train_and_evaluate/` is the published code, and must not be edited: a test fails if any of it changes.

To preview this site, run:

```bash
uv sync --frozen --extra docs
uv run mkdocs serve
```

The tutorial is shown without its outputs, which are computed when the site is built on GitHub.
To see them, execute the notebook, but do not commit the result:

```bash
uv run jupyter nbconvert --to notebook --inplace --execute docs/tutorial.ipynb
```
