# DietNetwork refactor plan

Living document: each pull request that moves the plan updates it.

## Status and next steps (2026-10-28)

| Phase | State |
|---|---|
| Rebuild on upstream `main`, CLI on top | ✅ PR #8, merged |
| 1. Reproduce published models, retrain | ✅ split recovered (seed 124); published 0.9214, retrained 0.9282 |
| 2. Scale (UK Biobank, 486,748 samples) | ✅ match 4.5 min (CPU), 15 models 13 min (one A100) |
| 3. Port remaining old-CLI features | dropped unless needed (`analyze-population`, `preprocess-plink`) |
| 4. Pluggable SNP features (`build-embedding`) | ✅ |
| 5. AlphaGenome (`dietnet alphagenome`) | ✅ UK Biobank array scored (485,516 SNPs × 3,849 tracks), table on Narval |
| 6. Training at biobank scale | ✅ `--inputs memmap/packed`, fast loader (bit-identical), `--engine dietnet` (13.5 s/epoch vs 65–80 s, 8 GB vs 82 GB), DDP; batch study: 552 with √-scaled LR keeps accuracy (docs/scaling.md) |
| 7. Quantitative traits | ✅ `adjust-phenotype`, `genotype_freq` features, r/R²/RMSE in `check` (PR #17); UKB height & BMI runs set up (`experiments/ukbb_traits/`) |
| MLP baseline (published split) | ✅ 0.9117 vs Diet Network 0.9282 retrained (−1.65 points paired, 1 of 15 higher) |

### Resume here: UK Biobank height and BMI (2026-10-28)

Goal: predict height and BMI from autosomal genotypes with the Diet Network,
and test whether AlphaGenome features for the auxiliary network help.
Everything is set up in `experiments/ukbb_traits/` (README there); Narval
compute is down until 2026-11-03 04:00 (reservation POWEROUT20261027).

Decided with Matt (2026-10-28):
- **Genotypes: the UK Biobank array** (485,516 SNPs, GRCh38, AlphaGenome
  already scored), not Camille's 408,940 imputed SNPs. The two are different
  UKB applications with no id map between them: the array, grenier2's
  `variables/` and the Pan-UKB files share ids; Camille's genotypes, her
  phenotype tables and the lab's White British / kinship lists
  (`shared/UKBiobank_data/useful_ID_lists/`) are application 49731.
- **Traits: height and BMI**, each regressed on sex and age (main analysis),
  and on sex, age and Townsend deprivation (22189) as a variant. Covariates
  explain R^2 = 0.527 / 0.530 of height and 0.009 / 0.017 of BMI.
- **Samples:** Pan-UKB `EUR` with every field present: 424,768. Relatedness
  later: Pan-UKB's `related` flag (64,087 of them) is in
  `inputs/phenotypes.tsv`, and dropping them needs only a new partition.
- As fast as possible: dietnet engine, packed genotypes staged to node disk,
  one model per A100, per-fold statistics in parallel, everything chained.

State on Narval (user's scratch, `dietnet_traits/`):
- `repo/`: checkout of `feat/ukbb-traits` with its venv (the older
  `dietnet_refactor/` checkout is left alone for its queued benchmark jobs);
  `traits.env`: the paths and accounts.
- `exp/inputs/`: phenotypes, keep list, classes and the four label files
  (`prepare.sh` done).
- `submit.sh prep` queued (subset, 4 datasets, partition, 5 stats folds,
  merge, 5 embeddings, then the pilot: height / freq / fold 0 at batch 138 and
  552); job ids in `exp/logs/jobs.txt`.

When Narval is back:
1. Watch the chain (`squeue -u $USER`, `exp/logs/`). Every step was timed on
   Mila at full shape with simulated data (`dryrun/scale.sh`; table in the
   experiment's README): datasets 30 min (53 GB), statistics 11 min to 2 h a
   fold (124 GB), embeddings up to 45 min (117 GB), an epoch 26 s, so a model
   is at most ~1.5 h. That run caught create-dataset's memory (32G killed it;
   now 80G, and the pending Narval jobs were updated).
2. Pilot: best epoch and held-out r at batch 138 vs 552
   (`dietnet check --packages exp/runs/height/freq/<config>/packages`). Batch
   552 is only ~8% faster an epoch, so 138 (published) is the grid's default.
3. Grid, cut with Matt to what answers the question: `height bmi` x
   `freq freq_ag`, folds 0-4, seed 78, b138 (20 GPU jobs, ~7-30 GPU-hours):
   tasks 0-4, 10-14, 30-34, 40-44 of job array 7503767 (the other 40 tasks
   and the pilot were cancelled). Only if `freq_ag` beats `freq`: its
   `freq_random` control (the embedding is built anyway). Later, if wanted:
   `ag` alone, the `_tdi` label sets, the MLP, more seeds.
4. Later: relatedness (partition without Pan-UKB `related`), and Camille's
   results for comparison (her runs are in `cam27/DIETNET_EXP/HEIGHT/` and
   `BMI/`, on her imputed SNPs).

Lessons from the cluster work:
- Mila: `ssh mila`, SLURM only, nothing on the login node; benchmark data and
  checkout in the Mila scratch `dietnet_bench/`. The locked torch (CUDA 13) has
  no V100 kernels: request A100s. Never update a checkout from inside
  concurrent jobs that share it (they collided); update it once, then submit.
- Stage data to `$SLURM_TMPDIR` with `cp -p` (packed copies check mtimes);
  network storage is unusable for training.
- Narval benchmark jobs on real UKBB genotypes are queued for after the outage
  (`runs/bench_train` in the user's scratch).

Next, in order:
1. The UKBB height model, above.
2. Run the experiment grid on the published split, 15 models per
   configuration, A100s, about 6 GPU-hours each: freq · AlphaGenome · freq +
   AlphaGenome · freq + random · freq + shuffled AlphaGenome. The MLP
   baseline is done (`reproduction/README.md`); freq is the retrain.
3. Optional, needs data decisions: UK Biobank genotypes at the published SNPs,
   to apply the published models to UK Biobank; or the HGDP+1KGP panel with
   its smallest populations merged or dropped, to train a panel model.

Cluster notes: Narval jobs go through SLURM, never the login node. CPU is
fine up to about an hour; anything longer goes to a GPU. Inputs and runs from
this work are in the user's Narval scratch (`dietnet_refactor/`,
`dietnet_published/`).

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
