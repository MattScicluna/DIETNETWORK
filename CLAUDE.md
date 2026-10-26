# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## What this is

HussinLab's published Diet Network (AJHG 2026), with a `dietnet` CLI around it.
The branch history is HussinLab/DIETNETWORK `main` with this repository's
history merged in. Read `docs/refactor-plan.md` for the plan and where it stands.

## The rule that matters most

**`Dietnet_train_and_evaluate/` is upstream's code: never edit it.**
`tests/unit/test_upstream_unchanged.py` fails on any change, against
`src/dietnet/upstream_manifest.json`. If upstream behaviour has to differ, do it
in `src/dietnet/` around the call, as `_launch.py` does for torch's safe
loader. To take newer upstream code, merge `upstream/main`, delete again what
the manifest lists as removed, and regenerate the manifest with
`scripts/upstream_manifest.py`.

Keep only code the API uses. Upstream files the CLI does not run are removed
and listed in the manifest; nothing unused lives in `src/` either.

## Layout

- `src/dietnet/cli.py`: click CLI.
- `upstream.py`: runs upstream's scripts as subprocesses (`run`) and imports
  its modules (`modules`); also holds the manifest check.
- `_launch.py`: the launcher `run` uses.
- `dataset.py`: PLINK → upstream's HDF5 schema (contiguous, plink `<id>_<A1>`
  SNP names); `snp_table`; `memmap_inputs`.
- `plink.py`: vectorised `.bed` reader.
- `align.py`: allele-aware SNP matching, the D/(D−|C|) scale, and the matched
  dataset.
- `package.py`: model packages; loads via upstream's `DietNetworkHandler`,
  strictly.
- `training.py`: drives upstream `train.py` per (seed, fold) and packages the
  results.
- `predict.py`: memory-mapped matched cohort and upstream's `indep_test_step`
  for every model; ensemble table.
- `published.py`: the Zenodo release as packages.
- `benchmarks/`: I/O benchmark and SLURM jobs for biobank-scale runs (paths
  come from the environment).

## Development

```bash
uv sync --frozen --extra dev && uv run pre-commit install
uv run pytest
DIETNET_UPSTREAM_DIR=/path/to/full/upstream/checkout uv run pytest   # parity tests
```

- Parity with upstream (`tests/parity/`) is the definition of correct. A change
  that moves a parity result is a bug, or a decision to record in the plan; it
  is never a tolerance to loosen.
- black, isort and flake8 at line length 100. pre-commit also runs
  `scripts/check_sensitive_files.py` and gitleaks.
- Never commit genotypes, HDF5 datasets, `.npy`/`.npz` arrays or `.pt` weights
  (except the fixtures in `.sensitive-allow`), and never absolute cluster paths.
- On the cluster, heavy work goes to SLURM jobs on compute nodes, never the
  login node. Cohort data and outputs derived from it stay on the cluster.
- Work on a branch, with a PR into `master`. No PRs to HussinLab/DIETNETWORK
  for now.
