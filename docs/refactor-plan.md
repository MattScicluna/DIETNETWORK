# DietNetwork refactor plan

Living document: each pull request that moves the plan updates it.

## Where things stand (2026-10-25)

**Strategy: rebuild on upstream, not alongside it.** This repository is now
HussinLab/DIETNETWORK `main` (`5481556e`, the code behind the AJHG 2026 paper)
with this repository's history merged in (`git merge -s ours`, so the tree is
upstream's), plus `src/dietnet/`: the CLI, packaging and inference layer,
re-applied *around* upstream's code instead of a 2022 copy of it.

- Upstream's training code runs **unmodified**. Only the 12 files the API
  needs are kept; 30 unused ones are removed. All are listed with hashes in
  `src/dietnet/upstream_manifest.json`, and the suite fails on any edit.
- **Parity is tested, not assumed** (`tests/parity/`, against a full upstream
  checkout at the pinned commit):
  - `create-dataset` == upstream's `create_dataset.py`;
  - `predict` == upstream's `test_independent_dataset.py`, with upstream doing
    its own SNP matching too;
  - with the 15 published models, `predict` == upstream's
    `Dietnet_inference/inference.py`.
- The earlier Phase 0 harness (PR #7) compared the *old* core with upstream.
  It found the old core trained differently from the published model (B1: main
  dropout fixed at 0.5 instead of the configured 0.1; B3: best-model rule).
  Those findings are why the old core was dropped rather than fixed. PR #7 is
  superseded by this branch.

### Decisions

| Question | Decision |
|---|---|
| Base | Upstream `main`, published code unmodified; our CLI on top (2026-10-25). |
| Unused code | Removed: upstream files the API does not run, and everything of the old core. |
| Best-model rule | Upstream's, by construction (its `train.py` runs). |
| On-disk intermediate | Upstream's HDF5 schema, built straight from `.bed`, contiguous (benchmark below). |
| Upstream contributions | None for now. |
| AlphaGenome | Hosted API, behind a cached precompute step (Phase 4). |
| Best practices | As manifold_genetics: uv + lock, black/isort/flake8, pre-commit with sensitive-files guard and gitleaks, CI matrix, CHANGELOG. |
| Retrain | 1KGP with the published config, compared with the published models (Phase 1). |

### What became of the old bug ledger

The ledger (B1–B18) was written against the old core:
- **Gone with the old core:** B1, B3, B5, B9–B12, B15 (strict loading is now in
  `package.py`), B16 (regression now runs end to end through upstream), B17.
- **Fixed in the new layer:**
  - B2: fold-independent embeddings; upstream's `DietNetwork` takes them, and
    `training.fold_slice` passes them through.
  - B4: `preprocessing.normalize` is recorded in every package.
  - B6: allele-aware matching (`align.py`).
  - B8: the D/(D−|C|) missing-SNP scale, as published.
  - B18: the matched cohort goes to `--work-dir`.

New, both kept out of upstream's files rather than patched in:

- **U1.** Upstream's `train.py` reloads its checkpoint with a bare `torch.load`.
  torch ≥ 2.6 refuses it, because the checkpoint holds numpy scalars.
  `dietnet._launch` allowlists exactly those numpy types for torch's safe
  loader, keeping `weights_only=True`, then runs the script unmodified.
- **U2.** Upstream's `partition_data.py` iterates `set(labels)`, so the order
  of samples within a fold, and with it the training run under a fixed seed,
  depends on Python's per-process hash seed. `dietnet` runs upstream scripts
  with `PYTHONHASHSEED=0`.

### What the published paper and Zenodo record pin

Rochefort-Boulanger, Scicluna et al., AJHG 2026 (thesis chapter 4); Zenodo
record 21133813:
- **Data:** 1KGP WGS30x (GRCh38), 2,322 samples in 24 populations, 229,986 SNPs.
- **Protocol:** 5 folds × 3 seeds (78, 79, 80), 60/20/20 splits with labels
  balanced across them.
- **Config:** input dropout 0.995, main dropout 0.1, learning rates 6e-4
  (aux) and 3e-4 (main), batch 138, patience 2000.
- **Missing SNPs:** rescaled by D/(D−|C|).
- **Largest run:** All of Us, 447,278 people.

`dietnet published` turns the record into packages, and the slow parity test
runs all 15.

## On-disk intermediate: benchmark

`benchmarks/bench_genotype_io.py` on the full 1KGP test set (2,322 × 229,986,
509 MiB as int8). Mac, warm page cache:

| Format | Full load | 50 random batches of 138 | Size |
|---|---|---|---|
| `.bed`, pyplink per-marker | 2.36 s | — | 127 MiB |
| `.bed`, vectorised decode | 2.48 s | — | 127 MiB |
| `.npy` | 0.08 s | 0.31 s (memmap) | 509 MiB |
| **HDF5, contiguous** | **0.11 s** | **0.36 s** | 509 MiB |
| HDF5, row-chunked | 0.11 s | 43 s | 519 MiB |
| HDF5, chunked + lzf / gzip | 0.90 / 1.28 s | 86 / 102 s | 256 / 115 MiB |

- HDF5 is about 20× faster than decoding PLINK each time (Camille's point), and
  no faster than `.npy`.
- It is the intermediate because it is upstream's format and self-describing,
  and because, being contiguous, it can be memory-mapped. That is how
  `predict` serves a biobank without loading it.
- Chunked or compressed layouts are ruled out for minibatch reads.

UK Biobank on Narval (486,748 × 169,829; 77 GiB as int8), Lustre scratch, one
CPU node, repeats 1:

| | `.npy` | HDF5 contiguous |
|---|---|---|
| build from the decoded matrix | 58 s | 57 s |
| full load | 304 s | 96 s |
| 50 random batches of 138 | 293 s (memmap) | 63 s |

Decoding the `.bed` from Lustre took 370 s, with peak memory 154 GiB (the
benchmark holds two copies). HDF5 was read second, so some of its advantage
may be page cache. Either way, at this scale nothing should be loaded more
than once.

**Write pattern matters more than format.** The first UK Biobank `predict` run
filled the sample-major matched file a block of SNP *columns* at a time. Every
write touched all 486,748 rows, and the file was still unfinished after 2 h.
Both writers now go a block of *samples* at a time (`Bed.sample_blocks`):
bounded memory (4 GiB per block), and each block is one contiguous run of
rows. The rerun's timings go here.

## Phases from here

1. **Reproduce, then retrain** (in progress).
   - ✅ **The published split is recovered.** It is upstream's `partition_data.py
     --balance-class --seed 124`. The file was taken from the lab's 1KGP
     directory on the cluster and is now `reproduction/1kgp_partition_seed124.npz`.
     Its training indices reproduce Zenodo's per-fold means, SDs and
     embedding exactly. `dietnet partition --seed 124` regenerates its
     membership but not its within-fold order (U2), so the retrain uses the
     file.
   - ✅ **The published accuracy is reproduced.** Each of the 15 models, on its
     own test fold: mean 0.9214, sd 0.013 (per model in
     `reproduction/README.md`), matching the paper's Figure 1A (~0.92).
     Pinned by `tests/parity/test_published.py` (slow).
   - **Finding:** the published models count the ALT allele at all 229,986
     SNPs, but the distributed 1KGP PLINK files count the minor allele
     (plink 1.9's default). They differ at 56,195 SNPs (24%), which
     allele-aware matching flips back; same accuracy either way. Matching by
     SNP id alone would feed those SNPs mirror-coded.
   - **Retrain:** 15 models on GPUs with the published configs and split
     (`reproduction/submit_retrain.sh`), then compare per (seed, fold) with
     the published models. Identical weights are not expected (GPU
     nondeterminism, torch version), so the comparison is accuracy and
     prediction agreement.
2. **Scale.**
   - UK Biobank on Narval end to end: time and peak memory per step
     (`benchmarks/slurm/bench_predict.sbatch`).
   - GPU prediction.
   - Decide whether training on a biobank-sized *training* set is in scope:
     upstream's `train.py` loads the training set into RAM.
3. **Port what is still missing from the old CLI, only if used:**
   `analyze-population` plots and `preprocess-plink`. Allele-aware matching
   may make the latter unnecessary.
4. **Pluggable SNP features (the aux-net input).** Upstream's `DietNetwork`
   takes any SNPs × features matrix from `embedding.npz` (2-D or per-fold 3-D),
   normalizing columns itself. So new features need no change to upstream
   code; they need an embedding builder:
   - providers (`class_genotype_freq`, which is upstream's `generate_embedding`;
     `precomputed` tables keyed by `chr:pos:counted:other`; `random` /
     `shuffled` controls);
   - fold-dependence recorded explicitly;
   - per-block normalization.
5. **AlphaGenome via the hosted API.**
   - A resumable, cached precompute writes a `precomputed` table of ALT − REF
     variant-effect scores.
   - The sign is flipped per SNP where the counted allele is REF (24.5% of the
     1KGP SNPs).
   - Experiments: freq only, AlphaGenome only, both, random, shuffled, and an
     MLP (upstream's `MlpHandler` exists).
