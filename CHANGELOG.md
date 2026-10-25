# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Refactor plan and bug ledger** (`docs/refactor-plan.md`): how this fork and
  HussinLab/DIETNETWORK diverged, what is verified equivalent, and fifteen known
  issues (B1–B16) with the phases that fix them.
- **Parity suite against upstream** (`tests/parity/`, `pytest -m upstream`).
  It imports HussinLab/DIETNETWORK at a pinned commit and checks that the dosage
  matrix, input statistics, imputation, genotype-frequency embedding, forward
  pass and one training epoch match. They do, except where the ledger says
  otherwise.
- **Known bugs as strict xfails**, each named after its ledger entry.
- **Fixtures**: `kgp_mini` (public 1KGP subset) and `synthetic` (with missing
  calls and a trait), each with upstream's HDF5 of it.
- **End-to-end CLI tests** with accuracy floors on both fixtures.
- **Genotype I/O benchmark** (`benchmarks/bench_genotype_io.py`), which backs the
  decision to make a contiguous HDF5 file the intermediate.
- **Tooling** as in manifold_genetics: committed `uv.lock`, black/isort/flake8,
  pre-commit with the sensitive-files guard (plus a rule for HDF5, `.npy`,
  `.npz` and `.pt` files), gitleaks and a size limit; CI with leak-check, lint +
  lock drift, and tests on Python 3.10–3.12 and macOS.

### Changed

- `CLAUDE.md` rewritten to describe the repository as it is.
- `pyproject.toml`: isort and flake8 replace ruff in the dev extra; pytest config
  and markers; URLs point to this repository.
- Pre-existing code under `Dietnet/` reformatted with black and isort. No
  behaviour change; the parity suite passes before and after.
