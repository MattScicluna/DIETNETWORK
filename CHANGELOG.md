# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-26

Rebuilt on HussinLab/DIETNETWORK `main` (`5481556e`). The training code is
upstream's, unmodified. The `dietnet` command line, model packages and
inference are re-applied around it, replacing this repository's 2022 copy of
the core, which trained differently from the published model (its main
dropout was fixed at 0.5; the published config uses 0.1).

### Added

- `dietnet build-embedding`: the auxiliary network's SNP features from several
  sources: the published genotype frequencies, any precomputed per-SNP table
  (oriented to the counted allele), and random or shuffled controls, with
  per-block centring and PCA. Packages record which blocks a model used.
- `dietnet match`: prepare a cohort on CPU; `predict` with the same `--work-dir`
  goes straight to inference (for a GPU job).
- `generate-embedding` and `train` refuse classes that have no training
  samples in some fold: upstream's embedding is NaN for them (or silently
  drops their columns).
- `reproduction/`: the published split (seed 124, verified against Zenodo), the
  published models' held-out accuracy (0.9214) pinned by a slow test, and the
  SLURM jobs that retrained all 15 (0.9282).
- `dietnet train --model mlp`: upstream's MLP baseline (no auxiliary network).
- `dietnet alphagenome`: AlphaGenome variant effects for every SNP through the
  hosted API, resumable, as a feature table for `build-embedding`
  (`pip install 'dietnetwork[alphagenome]'`).
- `dietnet create-dataset`: PLINK straight to upstream's HDF5 schema, identical
  to upstream's `create_dataset.py` output and without the text round-trip.
- `dietnet published`: the 15 published models (Zenodo
  doi:10.5281/zenodo.16943452) as model packages.
- Allele-aware SNP matching. Dosages that count the other allele are flipped,
  the other strand is recognised, and A/T and C/G SNPs are kept only on an
  exact match. The published D/(D−|C|) missing-SNP scale is applied.
- Biobank-scale prediction: the cohort is matched once into a contiguous HDF5
  under `--work-dir`, memory-mapped, and shared by every model.
- `dietnet info`, which reports the upstream commit and whether its files are
  unmodified.
- Parity tests against upstream: dataset builder, `predict` vs
  `test_independent_dataset.py`, and the published models vs
  `Dietnet_inference/inference.py`.
- Tooling as in manifold_genetics: uv lock, black/isort/flake8, pre-commit with
  a sensitive-files guard and gitleaks, CI on Python 3.10–3.13 and macOS.
- Released as `dietnetwork` on PyPI: `release.yml` publishes on a version tag
  through trusted publishing, after checking the tag, a dated CHANGELOG entry,
  the artefacts' contents and a clean-room install (`dietnet info` must find
  the bundled upstream code byte for byte the pinned commit).
- Tests of what would be published: package metadata
  (`tests/unit/test_package_metadata.py`) and the sdist and wheel themselves
  (`tests/integration/test_sdist_contents.py`: no untracked file, nothing the
  sensitive-files check would flag, upstream code identical), run in a CI
  `package` job.
- A Claude Code data-egress hook (`.claude/hooks/egress_guard.py`, run by the
  project's `.claude/settings.json`): agents are asked for approval before any
  copy from a remote host, any dump of a genotype or data file over ssh, or any
  upload of one to a host that is not an Alliance cluster, and cannot commit or
  push with `--no-verify`.
- `LICENSE` (BSD 3-Clause) and `NOTICE`, which keeps HussinLab's upstream code
  outside that licence; `py.typed`.
- A documentation site (mkdocs-material, `docs/`, the `docs` extra), built
  strictly on every pull request and deployed to GitHub Pages from `master`.
  Its tutorial runs the whole pipeline on the `kgp_mini` fixture on every
  build, and a test keeps `docs/cli.md` naming exactly the CLI's commands and
  options.

### Fixed

- `build-embedding` with a `class_genotype_freq` block failed when
  `--dataset` or `--partition` was a relative path: upstream's script runs from
  its own directory and could not find them.

### Changed

- `partition`, `generate-embedding`, `compute-input-stats` and `train` now run
  upstream's scripts. Partitions are label-balanced by default, as in the paper.
- Model packages (format 2) hold upstream's checkpoint, the fold's embedding and
  statistics, a training header and metadata (task, normalization, seed, fold,
  upstream commit). Weights load strictly.
- The predictions TSV has a header: `sample`, `prediction`, `probability`,
  `model_agreement` (regression: `prediction`, `sd_across_models`).
- `check` takes `--packages` (held-out accuracy per model) or `--predictions`
  with `--labels`.
- Commands use `--plink` and `--out`. The old CLI's `--exp-path`-relative file
  names are gone.
- `dietnet alphagenome` sends one request per SNP and retries transient API
  errors with backoff. About 2% of requests fail at random with gRPC
  `INTERNAL`, and the client's batch call failed a whole 200-SNP chunk on any
  one of them, so a run never got past its first chunk.

### Removed

- The 2022 core (`Dietnet/`): model, training loop and dataset code, replaced by
  upstream's.
- Upstream files the CLI does not run: deprecated code, experiments, upstream's
  inference scripts and `create_dataset.py` (listed in the manifest).
- `preprocess-plink`, `analyze-population` and the Dropbox model presets.
  Allele-aware matching covers the first; the others return if needed.
