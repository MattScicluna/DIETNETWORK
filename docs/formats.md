# Formats

Here we describe the files that the commands read and write, in the order in which they are used.
To train a model, see [Training](training.md), and to predict with one, see the [Quickstart](quickstart.md).

## Inputs

**Genotypes**: PLINK 1 files, `prefix.bed`, `prefix.bim` and `prefix.fam`.
Commands take the prefix (`--plink cohort` for `cohort.bed`).
The genotype of an individual at a SNP is its dosage: the number of copies of the A1 allele in the `.bim`, so 0, 1 or 2.

**Labels**: a tab-separated file with a header row, then one row per individual with its id (the IID in the `.fam`) and its label.
Other columns are ignored, duplicated ids are refused, and `create-dataset` needs a label for every individual in the `.fam`.

```text
Sample	Population
HG00096	CEUGBR
HG00130	CEUGBR
```

The [training config](training.md#inputs) and the [descriptions of SNP features](features.md) are on their own pages.

## The HDF5 dataset

`create-dataset` writes two files.
`dataset.h5` describes the individuals, the SNPs and the labels, and `dataset.h5.packed.bin` holds the genotypes.

| in `dataset.h5` | contents |
|---|---|
| `samples` | the ids of the individuals, in the order of the `.fam` |
| `snp_names` | `<id>_<A1>` for each SNP, so that the name says which allele is counted (for example `chr1:858952:G:A_A`) |
| `snps/chrom`, `snps/pos`, `snps/a1`, `snps/a2` | the position and alleles of each SNP, needed to match a new cohort |
| `class_labels`, `class_label_names` | the class of each individual, as a position in the sorted list of class names (with `--labels`) |
| `regression_labels` | the value of each individual (with `--regression-labels`) |
| `inputs` | individuals × SNPs: has the shape of the genotypes, but holds them only with `--int8` (−1 where missing) |

The packed file stores each genotype in 2 bits: 0, 1 and 2 are the dosages and 3 is missing.
A small file next to it, `dataset.h5.packed.json`, records the size and modification time of `dataset.h5`.
If these change, the packed file is refused, since it may no longer describe the same genotypes.
Copying `dataset.h5` changes its modification time, so copy the three files together and keep the times:

```bash
cp --preserve=timestamps dataset.h5 dataset.h5.packed.* destination/
```

Datasets made by the published code, such as the published `1KGP_dataset.hdf5`, can also be used.
They hold the genotypes in `inputs`, and have no `snps/` group.
The alleles and positions of their SNPs are read from names of the form `chr:pos:ref:alt_A1`.
With other names, a cohort is matched by SNP id.

## Files for training

| file | written by | contents |
|---|---|---|
| `partition.npz` | `partition` | `folds_indexes`: for each fold, the rows of the train, validation and test individuals. `folds_samples`: the same as ids |
| `embedding.npz` | `generate-embedding`, `build-embedding` | `emb`: the SNP features, folds × SNPs × features, or SNPs × features if all folds share them |
| `stats.npz` | `compute-input-stats` | `means_by_fold`, `sd_by_fold`: the mean and standard deviation of each SNP, folds × SNPs |

`build-embedding` also writes `embedding.npz.json`, which describes each block of features and names each feature.

## Model packages

A package is a directory that holds one trained model and all that is needed to predict with it.
`train` writes packages to `exp/packages/seed_<seed>/fold_<fold>/`, and `published` to `DEST/packages/seed_<seed>/fold_<fold>/`.

| file | contents |
|---|---|
| `model.pt` | the weights of the best epoch |
| `config.yaml` | the training config, with the seed |
| `embedding.npz` | the SNP features of this fold (absent for the MLP) |
| `input_features_stats.npz` | the mean and standard deviation of each SNP in this fold |
| `train_header.h5` | the SNPs and class names of the training dataset |
| `metadata.json` | how the model was trained (below) |
| `test_predictions.tsv` | the id, prediction and label of each individual in the test set (only from `train`) |

For example, this is the `metadata.json` of a model with [free SNP weights](training.md#free-snp-weights):

```json
{
  "format_version": 3,
  "model": "Dietnet",
  "task": "classification",
  "classes": ["ACB", "ASW", "..."],
  "preprocessing": {"normalize": true, "missing_snps": "scale D/(D-|C|)"},
  "seed": 78,
  "fold": 0,
  "n_snps": 2000,
  "upstream_commit": "5481556e472140c6d3a67363c1286f22109fb1ce",
  "dietnet_version": "0.5.0",
  "created": "2026-11-25T11:52:17-0500",
  "results_dir": "...",
  "embedding": null,
  "free_snp_weights": {"lambda": 1.0, "lr": 0.0003}
}
```

`model` is `Dietnet` or `Mlp`, and `classes` is empty for regression.
`upstream_commit` is the version of the published code.
`embedding` is the description written by `build-embedding`, or null.
`free_snp_weights` is present only if the model has them, and `snp_sample` and `chunk_rows` only if it was trained with these options.
The published packages have `source` (the Zenodo record) and `replicate` in place of `results_dir` and `embedding`.

**Check a package before sharing it.** `test_predictions.tsv` contains the ids and labels of individuals, and `results_dir` contains a path on your machine.

`--model` takes a package, or a directory that contains packages.
The packages used together must have the same SNPs, task and classes.
If the weights do not fit the model exactly, loading fails.

## The matched cohort

`match` and `predict` save the genotypes of a cohort at the models' SNPs, in `<cohort>.<fingerprint>.matched.h5`.
The fingerprint depends on the models' SNPs and on the cohort, so a second run on the same inputs reuses the file.

| in the file | contents |
|---|---|
| `inputs` | the genotypes, individuals × SNPs (−1 where missing) |
| `samples`, `snp_names` | the ids of the individuals and the names of the models' SNPs |
| `scale` | the number of SNPs in the models divided by the number found, used to compensate for the missing ones |
| `alignment` (an attribute) | counts of how the SNPs were matched, also printed by `match` and `predict` |

The counts in `alignment` are:

| count | SNPs that were |
|---|---|
| `same` | found, with the same alleles |
| `swapped` | found, with the alleles swapped, so that their genotypes were flipped |
| `strand`, `strand_swapped` | found on the other strand, with the same or swapped alleles |
| `ambiguous_kept` | A/T or C/G SNPs, kept because their alleles were the same (also counted in `same`) |
| `found_with_other_alleles` | at the same position, but with alleles that do not match |
| `missing` | not found, or found only with other alleles |
| `model_snps` | in the models |

`matched_by` says whether the SNPs were found by `position`, or by `id` if the models do not record positions.

**This file contains the cohort's genotypes.** Keep it where the cohort is.

## Predictions

`predict --out` writes a tab-separated file with one row per individual.
For classes, its columns are:

| column | contents |
|---|---|
| `sample` | the id of the individual (the IID in the `.fam`) |
| `prediction` | the class with the highest probability, averaged over the models |
| `probability` | that average probability |
| `model_agreement` | the fraction of models whose own most likely class is `prediction` |

For regression they are `sample`, `prediction` (the mean over the models) and `sd_across_models`.

### Per-model scores

`predict --scores scores.npz` also saves the scores of each model:

| key | contents |
|---|---|
| `samples` | the ids of the individuals |
| `models` | the name of each model, `seed<seed>_fold<fold>` |
| `classes` | the names of the classes (empty for regression) |
| `scores` | probabilities, models × individuals × classes, or predictions, models × individuals, for regression |
