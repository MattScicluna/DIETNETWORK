# Formats

Every file the pipeline reads and writes. Where upstream defines a format, it
is upstream's, unchanged: its scripts read the files `dietnet` writes.

## Inputs

**PLINK 1 binary filesets** (`prefix.bed/.bim/.fam`), SNP-major. Commands take
the prefix (`--plink cohort` for `cohort.bed`). Dosages count the `.bim` A1
allele: 0, 1 or 2, and missing calls become −1.

**Label tables** (`--labels`, `--regression-labels`, `check --labels`): tab
separated, a header row, then sample id (the `.fam` IID) and label. Further
columns are ignored. Duplicated sample ids are refused, and `create-dataset`
needs a label for every sample in the `.fam`.

```text
Sample	Population
HG00096	CEUGBR
HG00130	CEUGBR
```

**Training config** (`train --config`): upstream's YAML; see
[training](training.md#inputs).

**Feature spec** (`build-embedding --spec`) and **precomputed tables**: see
[SNP features](features.md).

## The HDF5 dataset

`dietnet create-dataset` writes upstream's schema, the one `partition_data.py`,
`generate_embedding.py`, `compute_input_features_statistics.py` and `train.py`
read:

| dataset | type | |
|---|---|---|
| `inputs` | int8, samples × SNPs | dosage of the `.bim` A1, −1 missing; contiguous, uncompressed |
| `samples` | bytes | sample ids (`.fam` IID), in `.fam` order |
| `snp_names` | bytes | `<.bim id>_<A1>`, the `plink --recode A` column names, so the counted allele travels with the name (as in the published 1KGP dataset: `chr1:858952:G:A_A`) |
| `class_labels` | int64 | index into `class_label_names` (with `--labels`) |
| `class_label_names` | bytes | the sorted unique labels |
| `regression_labels` | float64 | (with `--regression-labels`) |

Added here and ignored by upstream:

| | |
|---|---|
| `snps/chrom`, `snps/pos`, `snps/a1`, `snps/a2` | what matching a cohort by position and allele needs |
| attributes `source_plink_prefix`, `source_bed_fingerprint`, `built_by`, `built_at` | provenance |

Datasets made by upstream's `create_dataset.py`, including the published
`1KGP_dataset.hdf5`, have no `snps/` group. For those, the counted allele is
read from the `_A1` suffix of `snp_names`, and position and the other allele
from ids of the form `chr:pos:ref:alt`.

`inputs` is contiguous and uncompressed on purpose: chunked or compressed
layouts make the random minibatch reads of training 100 to 300 times slower
(benchmark in the [plan](refactor-plan.md#on-disk-intermediate-benchmark)), and
a contiguous array can be memory-mapped.

## Per-fold training inputs

All `.npz`, all upstream's:

| file | written by | keys |
|---|---|---|
| `partition.npz` | `partition` | `folds_indexes`: per fold, row indices of [train, valid, test]; `folds_samples`: the same as sample ids |
| `embedding.npz` | `generate-embedding`, `build-embedding` | `emb`: folds × SNPs × features (or SNPs × features, shared by all folds) |
| `stats.npz` | `compute-input-stats` | `means_by_fold`, `sd_by_fold`: folds × SNPs |

`build-embedding` also writes `embedding.npz.json`: each block's provider, spec,
column range and details (coverage, sign flips, a SHA-256 of a precomputed
table), plus every column name and the matrix's shape.

## Model packages

One trained (seed, fold) model, self-contained: prediction never needs the
training data or the experiment directory. `train` writes them to
`exp/packages/seed_<seed>/fold_<fold>/`, and `published` to
`DEST/packages/seed_<seed>/fold_<fold>/`.

| file | |
|---|---|
| `model.pt` | upstream's `best_model.pt`, as `train.py` saved it |
| `config.yaml` | the training config, seed included |
| `embedding.npz` | `emb`: this fold's SNP embedding (SNPs × features); absent for the MLP baseline |
| `input_features_stats.npz` | `means_by_fold` and `sd_by_fold` for this fold only |
| `train_header.h5` | `snp_names`, `class_label_names` and `snps/` from the training dataset: what upstream's classes read |
| `metadata.json` | see below |
| `test_predictions.tsv` | `train` only: the fold's held-out samples, with `sample`, `prediction`, `label` |

`metadata.json`:

```json
{
  "format_version": 2,
  "model": "Dietnet",
  "task": "classification",
  "classes": ["ACB", "ASW", "..."],
  "preprocessing": {"normalize": true, "missing_snps": "scale D/(D-|C|)"},
  "seed": 1,
  "fold": 0,
  "n_snps": 2000,
  "upstream_commit": "5481556e472140c6d3a67363c1286f22109fb1ce",
  "dietnet_version": "0.3.0",
  "created": "2026-10-26T15:20:23-0400",
  "results_dir": "...",
  "embedding": null
}
```

`model` is `Dietnet` or `Mlp`. `classes` is empty for regression. Packages from
`train` add `results_dir` (upstream's run directory) and `embedding` (the
`build-embedding` description, or null); the published ones add `source` (the
Zenodo record) and `replicate`.

A package is loaded with upstream's own classes (`DietNetworkHandler`, or
`MlpHandler`) and its weights loaded **strictly**: a checkpoint that does not fit
the model fails instead of leaving layers at their random initialisation.
`--model` takes a package or any directory above packages; all of them must
share the same SNPs, task and classes.

## The matched cohort

`match` and `predict` write the cohort on the models' SNPs to
`WORK_DIR/<cohort>.<fingerprint>.matched.h5`, where the fingerprint covers the
model SNP names, the cohort's `.bim` and `.fam`, and the `.bed`'s size. It
exists already on a rerun with the same inputs, and is reused. Upstream's
matched-dataset schema, so upstream's `test_independent_dataset.py` reads it
with `--matched-test-dataset`:

| | |
|---|---|
| `inputs` | int8, samples × model SNPs, contiguous; −1 where the SNP is missing |
| `samples` | the cohort's sample ids |
| `snp_names` | the model's |
| `scale` | D/(D−\|C\|): model SNPs over SNPs found |
| attribute `alignment` | JSON counts: how many SNPs were copied (`same`), flipped (`swapped`), found on the other strand (`strand`, `strand_swapped`), kept although strand-ambiguous (`ambiguous_kept`), present with other alleles (`found_with_other_alleles`) and `missing`, out of `model_snps`, and whether matching was by `position` or `id` |

Matching is by chromosome and position when the model records them, by SNP id
otherwise. These counts are what `match` and `predict` print. Built from cohort
genotypes, the file is cohort data: keep it where the cohort is.

## Predictions

`predict --out` writes one row per sample, tab separated. Classification:

| column | |
|---|---|
| `sample` | the `.fam` IID |
| `prediction` | the class with the highest mean probability across models |
| `probability` | that mean probability |
| `model_agreement` | the fraction of models whose own top class it is |

Regression: `sample`, `prediction` (the mean over models) and
`sd_across_models`.

### Per-model scores

`predict --scores scores.npz` adds:

| key | |
|---|---|
| `samples` | sample ids |
| `models` | `seed<seed>_fold<fold>` per model |
| `classes` | class names (empty for regression) |
| `scores` | models × samples × classes probabilities, or models × samples for regression |
