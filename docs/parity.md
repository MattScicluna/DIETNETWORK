# Parity with upstream

`dietnet` exists to run the published Diet Network, so its definition of
correct is *the same result as upstream*. Where this package does something
upstream's scripts also do, a test runs both and compares. A change that moves
a parity result is a bug, or a decision recorded in the
[plan](refactor-plan.md); it is never a tolerance to loosen.

## Upstream's code, unmodified

`Dietnet_train_and_evaluate/` holds only the upstream files the CLI runs (four
scripts and the `helpers` package they import), byte for byte as they are at
HussinLab/DIETNETWORK commit `5481556e`. The rest of upstream (deprecated code,
experiments, its own inference scripts) is not kept.

`src/dietnet/upstream_manifest.json` records the commit, a SHA-256 for every
kept file, and the list of removed ones. `tests/unit/test_upstream_unchanged.py`
fails if a kept file is modified or missing, if a file is added to upstream's
directories, if a removed one comes back, or if CI clones a different commit
from the one the manifest pins. `dietnet info` runs the same check on an
installed copy.

Where upstream's behaviour has to differ, it is changed *around* the call, never
in the file:

- **torch's safe loader.** `train.py` saves numpy scalars in its checkpoint
  and reloads it with a bare `torch.load`, which torch ≥ 2.6 refuses.
  `dietnet._launch` allowlists exactly those numpy types (keeping
  `weights_only=True`, the remedy torch's own error message gives) and then
  runs the script unmodified.
- **Hash seed.** `partition_data.py` iterates `set(labels)`, so the order of
  samples within a fold, and with it training under a fixed seed, depends on
  Python's per-process hash seed. Every upstream script runs with
  `PYTHONHASHSEED=0`.

Upstream's scripts run in a subprocess from their own directory, as upstream's
README runs them. For prediction, upstream's modules are imported in-process
instead, so that the model class and inference step are upstream's while the
I/O around them scales.

## What the parity tests compare

`tests/parity/` runs upstream's scripts from a full checkout at the pinned
commit (including the scripts this package removed) and requires:

| test | this package | against upstream's |
|---|---|---|
| `test_dataset.py` | `create-dataset` from the `.bed` | `create_dataset.py` from `plink --recode A` text: every dataset identical in dtype and value, built in one block and in many |
| `test_predict.py` | `predict` on a matched cohort | `test_independent_dataset.py` on the same cohort: same scores |
| `test_predict.py` | `predict` on a cohort missing 30% of the model's SNPs | `test_independent_dataset.py` doing its own SNP matching: same scores, so the missing-SNP handling and D/(D−\|C\|) scale are upstream's too |
| `test_predict.py` (slow) | `predict` with the 15 published models | `Dietnet_inference/inference.py`, the published inference path: same scores |
| `test_published.py` (slow) | the 15 published models on their own test folds | the published held-out accuracy, per model (mean 0.9214) |

Datasets must be identical. Scores are compared to floating-point rounding
(`rtol=1e-5`, `atol=1e-6`), since they come from separate processes; the
published accuracies may move by at most one test sample.

The HDF5 references in `tests/fixtures/` were built by upstream's
`create_dataset.py` with plink 1.9 (`tests/fixtures/build_fixtures.py`
rebuilds them).

## Running them

The parity tests skip unless `DIETNET_UPSTREAM_DIR` points at a full upstream
checkout of the pinned commit:

```bash
git init upstream && cd upstream
git fetch --depth 1 https://github.com/HussinLab/DIETNETWORK.git 5481556e472140c6d3a67363c1286f22109fb1ce
git checkout FETCH_HEAD
cd ..

DIETNET_UPSTREAM_DIR=$PWD/upstream uv run pytest
DIETNET_UPSTREAM_DIR=$PWD/upstream uv run pytest -m slow   # the published models; downloads Zenodo
```

CI does this on every pull request (without `slow`). Set
`DIETNET_PUBLISHED_DIR` to reuse a downloaded Zenodo release across slow runs.

## Beyond parity

Some checks have no upstream counterpart:

- **Allele-aware matching** (`tests/unit/test_align.py`): swapped alleles, the
  other strand, strand-ambiguous SNPs, missing SNPs and the scale. Upstream
  matches by SNP name only, which feeds mirror-coded genotypes for a SNP whose
  dosage counts the other allele. In the distributed 1KGP PLINK files that is
  56,195 of the published models' 229,986 SNPs.
- **The pipeline end to end** (`tests/integration/`): every command on both
  fixtures through the CLI, with accuracy floors, regression, the recorded
  preprocessing and strict loading.
- **Retraining** the 15 published models with `dietnet train`, on the
  published split and configs: held-out accuracy 0.9282 against the published
  0.9214, and 96.1% agreement with the published model of the same (seed,
  fold). Weights are not identical (GPU nondeterminism, a newer torch); see
  `reproduction/README.md`.

## Taking newer upstream code

```bash
git merge upstream/main
# delete again any file the manifest lists as removed
python scripts/upstream_manifest.py --commit upstream/main
uv run pytest
```

and move `UPSTREAM_SHA` in `.github/workflows/ci.yml` to the same commit, which
the manifest test checks.
