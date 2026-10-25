# DietNetwork refactor plan

Living document: each phase's pull request updates it. Bug ids (B1, B2, ...) are
referenced by the strict-xfail tests that pin them (`grep -rn "B1:" tests/`).

- **This repository** (`MattScicluna/DIETNETWORK`): `dietnet` CLI, PLINK-native,
  model packages, ensembles. The target user experience.
- **Upstream** (`HussinLab/DIETNETWORK`, pinned at `5481556e`): the lab's
  training code, `Dietnet_train_and_evaluate/` + `Dietnet_inference/`.

## Decisions (2026-10-25)

| Question | Decision |
|---|---|
| Best-model rule (B3) | Adopt upstream's (explained below). |
| Old packaged models (trained with dropout 0.5) | Retrain after Phase 1 and compare. Keep a loader shim for v1 packages. |
| On-disk intermediate | **HDF5 becomes the main path**, built once from PLINK, in place of the `.npy` cache. Layout from the benchmark below. |
| Contributions upstream | None for now; all work in this repository. |
| AlphaGenome | Hosted API, behind a cached, resumable precompute step (most maintainable: no weights or GPUs to manage). |
| Best practices | As manifold_genetics: uv + committed `uv.lock`, black/isort/flake8, pre-commit with the sensitive-files guard and gitleaks, CI matrix, strict-xfail bug ledger, parity tests, CHANGELOG. |

### The best-model rule, in plain terms

Training keeps the checkpoint from the epoch with the best validation accuracy.
When a later epoch *ties* that accuracy, the tie is broken on validation loss.
The question is: lower than *which* loss?

- **Upstream**: lower than the loss of the epoch that set the accuracy record.
- **Ours today**: lower than the lowest loss seen in *any* improving epoch, even
  one with worse accuracy.

Example. Baseline: loss 1.0, accuracy 50%. Epoch 1: loss 2.0, accuracy 60% →
both keep it. Epoch 2: loss 1.5, accuracy 60% (a tie). Upstream keeps epoch 2,
because 1.5 < 2.0. Ours does not, because it compares 1.5 with the baseline's
1.0, a loss from a different, worse model. Ours is stricter for no good reason
and can leave a worse-calibrated checkpoint in place. Upstream's rule is the
intended one. Pinned by
`tests/parity/test_model_parity.py::test_best_model_selection_agrees_on_accuracy_ties`.

## 1. How the two diverged

This repository branched from upstream at `2ec7fe3` (2022-01-07). Upstream has
had 104 commits since, this repository 39, and nothing was ported across.
Commit `82d749b "merge changes with main"` merges this repository's own
branches. So the core is the 2022 one, behind a new CLI, PLINK I/O and
packaging layer. Upstream rewrote its core in 2022–2024: model/task handlers,
configurable dropout, resume, MLP baseline, balanced splits, missing-SNP
scaling, an in-memory dataset and attribution models.

### 1.1 Verified equivalent (Phase 0 parity suite, live against upstream)

- **Dosage matrix**: the PLINK reader returns exactly upstream's HDF5 `inputs`,
  built by upstream's own `create_dataset.py` from `plink --recode A`. Same
  allele counted (the `.bim` A1), same missing calls. This closes the old open
  question B7.
- **Input means/SDs, missing-value imputation, genotype-class-frequency embedding.**
- **Forward pass**: identical logits given identical weights.
- **One full training epoch**: identical weights after Adam on identical batches
  from the same RNG state (at `dropout_main = 0.5`, see B1). Two optimizers
  upstream and two parameter groups here are numerically the same.

### 1.2 Bug ledger

✗ = pinned by a strict-xfail test; the fixing PR removes the marker.

| # | Issue | Where | Impact | Test |
|---|---|---|---|---|
| B1 | `dropout_main` is never read; `nn.Dropout()` is hardcoded (p = 0.5). | `helpers/model.py` | **Every model trained here used 0.5, whatever the config said** (shipped config: 0.1). | ✗ parity |
| B2 | A fold-independent 2-D embedding (n_snps × d) is mis-sliced: `load_embedding` always takes `emb[fold]`, and `_select_embedding_for_fold` slices axis 0 whenever `shape[0] > fold`. | `dataset_utils.load_embedding`, `cli._select_embedding_for_fold` | Blocks foundation-model embeddings. Fold-dependence is guessed from the array shape. | ✗ unit |
| B3 | Best-model rule (above). | `mainloop_utils.update_best_result` | Different checkpoint selection. | ✗ parity |
| B4 | The package does not record `normalize`; inference always z-scores when an SD is present. | `model_package.py`, `InferenceDataset` | `train --no-normalize` packages are fed differently scaled inputs at predict time. | ✗ integration |
| B5 | `convert_legacy_model` reads `stds_by_fold` (the file has `sd_by_fold`), so it centres but does not scale. It also ignores the `emb` key and packages every fold's embedding. `setup.sh` points to a `convert_legacy_models.py` that does not exist. | `model_package.py` | Legacy conversion produces unusable packages. Fix or delete. | ✗ unit ×2 |
| B6 | Predict-time SNP alignment matches on chr:pos or ID only, without checking alleles. Harmonization happens only via `preprocess-plink` (plink2 `--alt1-allele`). | `snp_alignment.create_snp_mapping` | Silent x → 2−x flips on datasets with swapped alleles. Upstream has the same gap; ours is ahead with `preprocess-plink`. Make allele-aware alignment the default. | Phase 2 |
| B8 | No missing-SNP compensation (upstream's `--scale`). | inference | Degrades on low-overlap arrays. | Phase 3 |
| B9 | Regression logs `Pearson_r = test_results[1]` (the predictions); r is `[2]`. | `train.py` | Logging only. | — |
| B10 | The genotype cache is written only above 2 GB; smaller data is re-decoded by every command. It is keyed by filename, with no content hash. | `load_plink_genotypes` | Speed; stale-cache risk. Superseded by the HDF5 intermediate. | ✗ unit |
| B11 | Train loss is the unweighted mean of minibatch means. | `mainloop_utils.train_step` | Reporting only. | — |
| B12 | `CombinedModel` registers its sub-networks twice (`feat_emb`/`disc_net` aliases), so its state dict has duplicate keys. | `model.py` | Package bloat; upstream checkpoints need key surgery to load strictly. | ✗ unit |
| B13 | `CLAUDE.md` described files that do not exist. | docs | Fixed in Phase 0. | — |
| B14 | No unit or parity tests. | `tests/` | Fixed in Phase 0. | — |
| B15 | `load_model_from_package` loads with `strict=False`. | `predict_with_plink.py` | A checkpoint missing a layer predicts with random weights, silently. | ✗ unit |
| B16 | Regression is broken end to end. `generate-embedding --task regression` computes per-class frequencies over the float trait and crashes. Predict builds a `len(label_mapping) == 0` output layer and assumes softmax. | `generate_embedding.py`, `predict_with_plink.py` | Regression unusable. Upstream builds the regression embedding from class labels stored beside the trait. The right regression embedding is a Phase 4 provider question. | ✗ integration |

| B17 | `get_train_dir` calls `sys.exit` without importing `sys`. | `helpers/test_external_utils.py` | The legacy `test_external_dataset.py` raises `NameError` instead of exiting cleanly when the training directory is missing. Found by flake8 (F821, exempted by name in `.flake8`). | — |

| B18 | Without `--temp-dir`, `predict` writes its genotype cache (an `.npy` the size of the int8 matrix: about 77 GiB for UK Biobank) next to the input PLINK files. | `predict_with_plink.py` | Fills, or fails on, shared or read-only cohort directories. The HDF5 intermediate should go to an explicit, user-chosen location. | — |

Q1 is **resolved** by the published model's Zenodo record (below): its three
configs are the shipped one with seeds 78, 79 and 80 (`lr_aux 6e-4`,
`lr_main 3e-4`, `input_dropout 0.995`, `dropout_main 0.1`, batch 138, patience
2000, annealing 0.99). The paper chose input dropout 0.995 by search ("retaining
an average of 1,150 SNPs per sample"). Upstream honours `dropout_main`, so the
published models trained with 0.1. Every model this repository trained used 0.5
(B1), which makes them a different model from the published one, not a
re-implementation of it.

### 1.4 What the published paper pins

Rochefort-Boulanger, Scicluna et al., *A Transparent and Generalizable Deep
Learning Framework for Genomic Ancestry Prediction*, AJHG 2026
(doi:10.1016/j.ajhg.2026.06.003). It is chapter 4 of the thesis. Weights,
configs, embeddings and the training HDF5 are on Zenodo:
doi:10.5281/zenodo.16943452, record 21133813 (`Dietnet_training.tar.gz`: 15
weight files `model_rep{1,2,3}_fold{0-4}.pt`, `config_rep{1,2,3}models.yaml`,
`SNPs_embedding.npz` (5 × 229,986 × 72), `normalization_stats.npz`,
`1KGP_dataset.hdf5`).

- **Data**: 1KGP WGS30x on GRCh38, 2,322 samples in 24 populations (CEU+GBR
  and ITU+STU merged and downsampled), 229,986 SNPs (GSA ∩ MHI arrays, MAF ≥ 5%,
  missingness ≤ 10%, LD-pruned, HLA removed, ShapeIT5-imputed). That is the
  1KGP test set already in `pretrained_models.py`.
- **Protocol**: 5 folds × 3 seeds, 60/20/20 train/valid/test, labels balanced
  across splits. That matches `partition --stratify` with the 0.75
  train/valid ratio. Embedding, means and SDs are per fold, from that fold's
  training split.
- **Missing-SNP rescaling is part of the published method.** SNPs absent from a
  cohort are set to 0 after standardization, and the observed ones are scaled
  by D/(D − |C|) (D SNPs in the model, |C| missing). This is applied "in all
  experiments". It is how the paper reaches 76%-missing arrays (CaG Axiom) and
  All of Us (11% missing). **B8 therefore moves into Phase 1**: without it this
  repository cannot reproduce the published cross-cohort results.
- **Ensemble**: the paper reports all 15 per-model labels. It does not combine
  them. This repository's majority vote and mean probability are a product
  choice to document, not a reproduction requirement.
- **Not stated** in the paper: the early-stopping and best-model rule, and
  whether dropout is active at evaluation time. Upstream's code is the
  reference for both.
- **Scale reached**: All of Us, 447,278 people on the GDA array. "Less than
  three minutes" CPU-only for 17,286 CaG samples from a pre-made HDF5; training
  takes about 22 minutes per model on an A100.

The Zenodo record also gives Phase 1 its strongest test: load the 15 published
weight files into this repository's model (via the strict upstream-checkpoint
loader, B12) and reproduce each fold's published test accuracy on
`1KGP_dataset.hdf5`. Mark it `slow` and `network`.

### 1.3 Upstream features absent here

- **Port (Phase 3):** task handlers; MLP baseline; resume from `last_model.pt`;
  missing-SNP `scale` and its experiment; a balanced-split option.
- **Later:** attribution model (`DietNetworkAttr` + captum), `get_last_layer`.
- **Skip:** `depricated/`, per-batch weight dumps, Theano init, upstream's
  genotype-text → HDF5 route (we build HDF5 straight from `.bed`).

## 2. On-disk intermediate: benchmark

`benchmarks/bench_genotype_io.py` on the full 1KGP test set (2,322 samples ×
229,986 SNPs, 509 MiB as int8). Mac, warm page cache, median of 3:

| Format | Full load | 50 random batches of 138 | Size |
|---|---|---|---|
| `.bed`, pyplink per-marker (today below 2 GB, every command) | 2.36 s | — | 127 MiB |
| `.bed`, vectorised decode | 2.48 s | — | 127 MiB |
| `.npy` (today's cache above 2 GB) | 0.08 s | 0.31 s (memmap) | 509 MiB |
| **HDF5, contiguous** (upstream's layout) | **0.11 s** | **0.36 s** | 509 MiB |
| HDF5, row-chunked | 0.11 s | 43 s | 519 MiB |
| HDF5, row-chunked + lzf | 0.90 s | 86 s | 256 MiB |
| HDF5, row-chunked + gzip | 1.28 s | 102 s | 115 MiB |

Reading:

- **Camille is right that HDF5 beats PLINK: about 20× per load.** But it is not
  faster than the `.npy` cache. The win comes from decoding once and reading
  raw int8 after that, and `.npy` does that too.
- HDF5 still wins as *the* intermediate, for a different reason: one
  self-describing file can carry genotypes, sample ids, SNP ids **with
  alleles**, labels and provenance. The `.npy` cache carries a bare matrix, with
  everything else in side files that can drift out of step (B10's stale-cache
  risk).
- Layout matters a lot: **contiguous and uncompressed.** Chunked or compressed
  layouts make random minibatch reads 100–300× slower under h5py's fancy
  indexing. Disk is cheap next to that.
- At UK Biobank scale (~488k samples, 112 GB int8) the 2.4 s decode becomes
  roughly 8 minutes per command, which makes building once matter more.
- These are laptop numbers. Rerun on the cluster filesystem, on the training
  node type, before Phase 2 fixes the layout:
  `python benchmarks/bench_genotype_io.py --plink-prefix ... --workdir $SLURM_TMPDIR`.

Schema (Phase 2), a superset of upstream's so that upstream HDF5 files still load:
`inputs` (int8, samples × SNPs, contiguous), `samples`, `snp_names`, plus
`snps/{chr,pos,counted_allele,other_allele}`, `class_labels`,
`class_label_names`, `regression_labels`, and attributes for source `.bed`
hash, builder version and creation time.

## 3. Strategy

Do not `git merge` upstream (104 commits, renamed trees, WIP). Keep this
repository's shell (CLI + package API) as the contract and rebuild the core
module by module. The parity suite turns every behavioural difference into an
explicit decision. Small PRs into `master`.

## 4. Phases

### Phase 0 — harness ✅ (this PR)

- Fixtures (`tests/fixtures/`, rebuilt by `build_fixtures.py`):
  - `kgp_mini`: 480 samples (20 × 24 populations) × 2,000 SNPs of the public
    1KGP test set.
  - `synthetic`: 200 × 500, 4 classes, 3% missing calls, a quantitative trait.
  - Each is committed as PLINK **and** as the HDF5 that upstream's own
    `create_dataset.py` builds from `plink --recode A`.
- Parity suite (`-m upstream`, live against upstream at the pinned commit; CI
  clones it): dosages, statistics, imputation, embedding, forward pass, one
  training epoch, best-model rule.
- Bug ledger as strict xfails (above).
- End-to-end CLI tests: synthetic must score ≥ 60% (chance 25%); `kgp_mini`
  must score ≥ 25% (chance 4.2%; calibrated at 40–45%).
- Tooling: uv lock, black/isort/flake8, pre-commit (sensitive-files guard with
  data-container rule, gitleaks, size limit), CI (leak-check, lint + lock drift,
  tests on 3.10–3.12 + macOS with CPU torch), CHANGELOG.
- I/O benchmark and decision (section 2).

### Phase 1 — correctness fixes, then retrain

One PR per fix, each flipping its xfail: B1, B15, B3, B8 (the paper's
D/(D − |C|) missing-SNP rescaling, §1.4), B4 (record
`preprocessing: {normalize, scale}` in the package), B2 (explicit
fold-dependence, see §5), B5 (fix or delete), B12 (+ a strict loader that accepts
upstream checkpoints), B9, B16 (or reject `--task regression` at `train`
until Phase 4 gives it an embedding).

First, reproduce the published accuracy from the Zenodo weights (§1.4). Then
**retrain 1KGP**: 5 folds × 3 seeds, this repository vs upstream, on the
same partition and the published config. Report per-fold accuracy, mean ± sd, and
against the shipped v1 packages. This answers "did my changes affect core
functionality" with numbers. Needs the cluster; add a
`scripts/cluster/retrain_1kgp.sh` with that PR.

### Phase 2 — restructure, no behaviour change

```
src/dietnet/
  cli/            click, thin: parse → call api
  api.py          fit(), predict(), load_package(), build_dataset(), compute_features()
  config.py       one validated, versioned schema
  data/
    hdf5.py       build-once HDF5 intermediate (schema above) from .bed, vectorised decoder
    snps.py       SnpTable: chr, pos, counted_allele, other_allele, id
    alignment.py  ID + allele harmonization (flip, strand, drop ambiguous), missing mask
    labels.py, partition.py, stats.py
  features/       SNP feature providers (§5)
  models/         aux_net.py, main_net.py, dietnet.py, mlp.py
  training/       trainer.py, tasks.py, checkpoint.py (resume)
  inference/      package.py (manifest v2), predictor.py, ensemble.py
```

Manifest v2 records the code version, the resolved config, preprocessing, the
SnpTable with alleles, the feature spec + hash, the label map, seed and fold. A
shim loads v1 packages. Allele-aware alignment becomes the default (B6). Docs
are checked against the CLI (`tests/unit/test_docs_match_cli.py`, as in
manifold_genetics). `CLAUDE.md` and the README are rewritten.

### Phase 3 — port upstream features

Task handlers, MLP baseline, resume, balanced split, missing-SNP `scale` (B8)
and its experiment as a command.

### Phase 4 — pluggable SNP features (§5)

Migrate the class-frequency embedding onto the provider interface first. Its
output must be **bit-identical** to Phase 1's.

### Phase 5 — AlphaGenome provider and experiments (§5.3–5.4)

## 5. Pluggable SNP-feature design

The aux net consumes a per-SNP matrix **F** (n_snps × d). Today F is the
genotype × class frequencies. Generalize F to a concatenation of blocks, each
from a **provider**:

```python
class SnpFeatureProvider(Protocol):
    name: str
    fold_dependent: bool          # fit per fold on that fold's training samples only
    uses_labels: bool             # leakage guard
    def fit(self, ctx: FeatureContext, fold: int | None) -> None: ...
    def transform(self, snps: SnpTable) -> FeatureBlock   # values (n × d) + coverage mask (n,)
    def spec(self) -> dict        # params + version: cache key and manifest entry
```

Built-ins: `class_genotype_freq` (today's), `allele_freq` (unsupervised; a
regression candidate, B16), `precomputed` (any table keyed by
`chr:pos:counted:other`: the generic hook), `alphagenome` (fills a
`precomputed` table), and `random` / `shuffled` controls.

Config:

```yaml
features:
  - {provider: class_genotype_freq, norm: l2_col}
  - {provider: precomputed, path: alphagenome_grch38.parquet, norm: zscore, project: 64}
```

- Per-block `norm` ∈ {`l2_col` (today's, the default), `zscore`, `none`}.
  Optional learned per-block `project: k`, so a wide foundation embedding does
  not swamp a 78-wide frequency block.
- SNPs a block does not cover get zeros plus an indicator column. Coverage is
  logged.
- Storage: each block declares `fold_dependent`, with values shaped
  `(n_folds, n, d)` or `(n, d)`, plus `meta.json`. Never inferred from the
  shape (B2).
- Packages embed the final per-fold F, so predict needs neither AlphaGenome nor
  the training data.

### 5.3 AlphaGenome via the API

- A `dietnet features alphagenome` precompute command: batched, rate-limited,
  resumable (it skips variants already in the output), writing
  `alphagenome_<scorer>_grch38.parquet`. The API key comes from an environment
  variable, never from config or repo files. Never called during training.
- Per-SNP vector: variant-effect scores (ALT − REF) over a fixed, recorded set
  of scorers and tracks. Interpretable, with controllable width.
- Orientation must be decided **per SNP**. In the 1KGP training data (WGS30x,
  GRCh38; IDs `chr:pos:REF:ALT`), the counted allele A1 is ALT for 173,642
  SNPs and REF for 56,344 (24.5%). That is plink 1.9 making A1 the minor
  allele, since `--keep-allele-order` was not used. The provider therefore
  flips the sign of ALT − REF wherever the counted allele is REF. It reads that
  from the SnpTable, never from the ID's position. (Check REF against the
  GRCh38 reference while doing this, in case the IDs are not REF:ALT.)
- The same mechanism makes B6 concrete: a cohort with different allele
  frequencies can get a different A1 at the same SNP, so allele-aware alignment
  is required, not optional.

### 5.4 Experiments

Same folds and seeds as the Phase 1 reference: freq only · AlphaGenome only ·
freq + AlphaGenome · freq + random (same width) · freq + shuffled AlphaGenome ·
MLP (no aux). The random and shuffled controls separate "AlphaGenome carries
signal" from "more aux capacity". Report per-fold and per-population accuracy,
plus accuracy under missing-SNP scenarios (Phase 3 tool).
