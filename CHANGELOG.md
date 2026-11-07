# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Quantitative traits (height first):
  - `dietnet adjust-phenotype`: a trait regressed on covariates (numeric, or
    text as indicators) and standardised, as regression labels; for height,
    sex and age, which a model of autosomes cannot see.
  - `build-embedding` provider `genotype_freq`: genotype frequencies per fold
    over all training samples (upstream's definition with classes pooled),
    the auxiliary network's baseline for a task without classes; read a block
    of samples at a time, so it scales to a biobank.
  - `dietnet check` reports Pearson r, R² and RMSE for regression packages and
    numeric predictions.
  - `experiments/ukbb_traits/`: height and BMI on the UK Biobank array, as a
    chain of SLURM jobs (labels, datasets, partition, per-fold statistics in
    parallel, feature specs, pilot, grid), with a dry run on fake data.
- `build-embedding`: a block with `include: false` is built for a later
  `shuffled` block and left out of the matrix, so a shuffled control can
  stand without its original.

- `dietnet train --inputs auto|ram|memmap`: where training reads genotypes.
  Upstream's train.py copies the whole matrix into RAM first (~83 GB for a
  UK Biobank-sized one); `memmap` reads each batch's rows from the dataset
  file instead, through a genotype store (`dietnet.genotypes`) the launcher
  installs in train.py's `FoldDataset`. The model is bit-identical. `auto`
  memory-maps above 2 GiB. Stage the dataset to local disk: from network
  storage an epoch took 45x longer (benchmarks in `docs/scaling.md`).
- `dietnet pack-genotypes` and `train --inputs packed`: a 2-bit copy of the
  genotypes (four per byte, missing as 3), sample-major and memory-mapped, so
  training from disk reads a quarter of the bytes. Decoded exactly
  (bit-identical model); a copy whose dataset has changed since is refused.
  `auto` uses it when present.
- `dietnet train --loader fast` (the default) and `--workers`: upstream's
  train.py, unmodified, fed whole batches per read by loader workers, with
  missing values filled in one operation instead of a Python loop over the
  batch. The launcher gives train.py's `FoldDataset` a `__getitems__` and its
  `DataLoader` workers and pinned memory; sampler, shuffling and random-number
  use are untouched, so the model is bit-identical to train.py left alone
  (`--loader upstream`), from every genotype store and with workers.
- `dietnet train --engine dietnet` (and `--gpus N`, which implies it):
  upstream's model, task handlers, epoch logic (baseline, annealing, best-model
  rule, patience) and result files in a loop of dietnet's own
  (`dietnet.parallel`), so packages come out as from train.py. It reads the
  packed copy a batch at a time and decodes it on the GPU, skips train.py's
  per-batch bookkeeping and its per-epoch evaluation of the training set, and
  can run data-parallel (DDP; the config's `batch_size` is the global batch).
  On a UK Biobank-sized matrix: 13.5 s per epoch on one A100 against 65–80 s
  for upstream in RAM, in 8 GB instead of 82 GB. Not bit-identical to
  train.py; judged on held-out accuracy. Benchmarks and a batch-size study on
  the published 1KGP split are in docs/scaling.md.
- `benchmarks/`: training time by genotype store, and training throughput by
  loader, workers, missing-value fill and batch size, on real or simulated
  genotypes at any shape.

### Fixed

- `dietnet alphagenome` combines its parts one at a time instead of reading
  them all into memory and copying them three times. For a biobank array
  (~485,000 SNPs x ~3,850 tracks, ~15 GB as float64) the old combine needed
  several times a laptop's memory, at the end of a day of API calls.

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
