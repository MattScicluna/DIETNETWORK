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
bounded memory (1 GiB per block), and each block is one contiguous run of
rows. Three further fixes came from runs and a memory profile on Narval:
- read the packed `.bed` once (it is a quarter of the decoded matrix);
- size blocks by the width they are written at;
- have `predict` read each batch once for all models.

UK Biobank (486,748 samples), after those fixes:

| Step | Where | Wall time | Peak RSS |
|---|---|---|---|
| `match` onto the HGDP-panel model (169,829 SNPs) | CPU, 8 cores | ~21 min (was > 2 h) | — |
| `match` onto the published models (229,986 SNPs) | CPU, 8 cores | 4 min 27 s | 23 GB (was 88 GB, OOM) |
| `predict`, 1 model, before batching | A100 | 628 s | — |
| `predict`, **15 published models** | A100 | **694 s** (12 min 56 s total) | memory-mapped file |

Before these fixes, the old code's pyplink read of this cohort alone took
6,339 s (1 h 45 min), on every command.

**UK Biobank data caveat.** The UK Biobank fileset used here was cut down to
the HGDP+1KGP panel's 169,829 SNPs. Only 5,810 of the published models'
229,986 SNPs are among them (both GRCh38; a different SNP selection, not a
build shift). So the run above measures scale only; its predictions use 2.5%
of the model's SNPs, and `match`/`predict` now warn below 1,000. Real UK
Biobank predictions from the published models need genotypes extracted at
the published SNPs from the raw UK Biobank data (GRCh37, so lifted over).

**Reference-panel caveat.** The HGDP+1KGP panel has 77 populations, some with
1-2 samples. Upstream's embedding cannot learn a class absent from a fold's
training split (NaN, or silently missing columns), so `dietnet` now stops
at `generate-embedding` with the classes named. Training on that panel
needs the small populations merged or dropped first.

## Phases from here

1. **Reproduce, then retrain** ✅.
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
   - ✅ **Retrained.** All 15 retrained with `dietnet train` on Narval A100s,
     using the published configs and split. Held-out accuracy 0.9282 against
     the published 0.9214. Paired by (seed, fold) that is +0.67 points (sd 1.1),
     higher in 11 of 15. Retrained and published models of the same
     (seed, fold) agree on 96.1% of samples. Weights are not identical (GPU
     nondeterminism, newer torch); the method's accuracy reproduces. Per-model
     table in `reproduction/README.md`.
   - **Phase 1 is done.**
2. **Scale.**
   - UK Biobank on Narval end to end: time and peak memory per step
     (`benchmarks/slurm/bench_predict.sbatch`).
   - GPU prediction.
   - Decide whether training on a biobank-sized *training* set is in scope:
     upstream's `train.py` loads the training set into RAM.
3. **Port what is still missing from the old CLI, only if used:**
   `analyze-population` plots and `preprocess-plink`. Allele-aware matching
   may make the latter unnecessary.
4. **Pluggable SNP features (the aux-net input)** ✅. `dietnet build-embedding`
   (`src/dietnet/features.py`) assembles `embedding.npz` from blocks:
   `class_genotype_freq` (upstream's script; alone it is bit-identical to the
   published embedding), `precomputed` (any per-SNP table, matched by position
   and allele on either strand, `orient: effect` flipped where the dataset
   counts REF, missing SNPs zeroed and flagged), and the `random` / `shuffled`
   controls. Upstream's model reads any SNPs x features matrix and
   L2-normalises each column itself, so no upstream change is needed, and
   per-column scaling is pointless. The transforms that matter are `center`
   and `pca: k`; the latter controls how many columns, and so how much weight,
   a block brings. All-zero columns (NaN in the model) are refused. Packages
   record the blocks.
5. **AlphaGenome via the hosted API**: code ✅, not yet run.
   `dietnet alphagenome` (`src/dietnet/alphagenome.py`, alphagenome client
   v0.10) scores each SNP with `score_variants` and keeps the signed `raw_score`
   per (output type, track). For gene-level tracks it keeps the gene with the
   largest effect. Scoring is resumable (one part file per chunk), with the key
   from `ALPHAGENOME_API_KEY` only. The output is a `precomputed` table, and
   `orient: effect` flips the sign per SNP where the counted allele is REF.
   Tested against a stand-in API and the real client's types; it has not been
   run against the API.
   - **To do:** score the 229,986 1KGP SNPs (resumable, rate-limited by the
     service), then run the experiment grid on the published split:
     freq only · AlphaGenome only · freq + AlphaGenome · freq + random (same
     width) · freq + shuffled AlphaGenome · MLP (`dietnet train --model mlp`).
