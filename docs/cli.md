# Command line

Here we list every command and its options.
To learn how to use them, start with the [Quickstart](quickstart.md) or [Training](training.md).
`dietnet --help` and `dietnet <command> --help` print the same information, and `dietnet --version` prints the version.

| to predict | |
|---|---|
| [`published`](#dietnet-published) | downloads the published models |
| [`match`](#dietnet-match) | finds the models' SNPs in a cohort, ahead of `predict` |
| [`predict`](#dietnet-predict) | predicts a cohort with one or more models |
| [`check`](#dietnet-check) | reports how accurate predictions or models are |

| to train | |
|---|---|
| [`create-dataset`](#dietnet-create-dataset) | reads the genotypes and labels |
| [`partition`](#dietnet-partition) | splits the individuals into folds |
| [`generate-embedding`](#dietnet-generate-embedding) | computes the published SNP features |
| [`compute-input-stats`](#dietnet-compute-input-stats) | computes the mean and standard deviation of each SNP |
| [`train`](#dietnet-train) | trains one model per seed and fold |

| less often | |
|---|---|
| [`build-embedding`](#dietnet-build-embedding) | builds other SNP features |
| [`alphagenome`](#dietnet-alphagenome) | gets AlphaGenome variant effects, for `build-embedding` |
| [`adjust-phenotype`](#dietnet-adjust-phenotype) | adjusts a trait for covariates |
| [`pack-genotypes`](#dietnet-pack-genotypes) | packs the genotypes of an older dataset |
| [`info`](#dietnet-info) | prints versions and checks the published code |

`--seeds` and `--folds` take several values, either separated by commas (`--seeds 78,79,80`) or by repeating the option (`--seeds 78 --seeds 79`).

## `dietnet published`

Downloads the 15 published models from Zenodo, checks the download, and saves the models as [packages](formats.md#model-packages) in `DEST/packages/`.

| option | | default |
|---|---|---|
| `--dest PATH` | where the download and the packages go | required |
| `--release PATH` | a `Dietnet_training/` directory that you already have, instead of downloading it | |

## `dietnet match`

Finds the models' SNPs in a cohort and saves their genotypes, so that `predict` does not have to.
It does not use a GPU, so on a cluster it can be a separate job.
It prints how many SNPs were found, and warns you if there are fewer than 1,000.

| option | | default |
|---|---|---|
| `--model PATH` | a package, or a directory of packages | required |
| `--plink TEXT` | prefix of the cohort's PLINK files | required |
| `--work-dir PATH` | where the genotypes are saved | required |
| `--seeds TEXT` | use only these seeds | all |
| `--folds TEXT` | use only these folds | all |

## `dietnet predict`

Predicts a cohort with every selected model and combines the predictions ([output](formats.md#predictions)).
It finds the models' SNPs first, unless `match` already has, and warns you if there are fewer than 1,000.

| option | | default |
|---|---|---|
| `--model PATH` | a package, or a directory of packages | required |
| `--plink TEXT` | prefix of the cohort's PLINK files | required |
| `--out PATH` | the predictions (TSV) | required |
| `--work-dir PATH` | where the cohort's genotypes are saved, or were saved by `match` | next to `--out` |
| `--seeds TEXT` | use only these seeds | all |
| `--folds TEXT` | use only these folds | all |
| `--batch-size INTEGER` | individuals predicted at a time | 1024 |
| `--device TEXT` | `cpu`, `cuda`, ... | `cuda` if available |
| `--scores PATH` | also save the scores of each model ([format](formats.md#per-model-scores)) | |

## `dietnet check`

Reports accuracy in one of two ways.
With `--packages`, it reports the accuracy of each model on its own test fold.
With `--predictions` and `--labels`, it compares predictions with known labels, overall and for each class.
For regression, it reports Pearson's r, R² and the RMSE.

| option | | default |
|---|---|---|
| `--packages PATH` | a directory of packages from `train` | |
| `--predictions PATH` | the output of `predict` | |
| `--labels PATH` | a TSV of sample id and label, with a header row | |
| `--min-accuracy FLOAT` | exit with status 1 if the accuracy is lower (classification only) | |

## `dietnet create-dataset`

Reads PLINK files and labels, and writes the [dataset](formats.md#the-hdf5-dataset) and, next to it, a packed file that holds the genotypes.
At least one of `--labels` and `--regression-labels` is required.

| option | | default |
|---|---|---|
| `--plink TEXT` | prefix of the PLINK files | required |
| `--labels PATH` | a TSV of sample id and class, with a header row | |
| `--regression-labels PATH` | a TSV of sample id and value, with a header row | |
| `--out PATH` | the dataset (`.h5`) | required |

## `dietnet partition`

Splits the individuals into folds, each with train, validation and test individuals.

| option | | default |
|---|---|---|
| `--dataset PATH` | the dataset | required |
| `--out PATH` | the partition (`.npz`) | required |
| `--nb-folds INTEGER` | number of folds | 5 |
| `--seed INTEGER` | random seed | 23 |
| `--balanced / --unbalanced` | split each class separately, as in the paper, or ignore the classes | balanced |
| `--train-valid-ratio FLOAT` | with `--unbalanced`, the fraction of the individuals outside the test set that are used to train | 0.75 |

## `dietnet generate-embedding`

Computes the published SNP features: the genotype frequencies of each SNP in each class, from the training individuals of each fold.
It stops if a class has no training individuals in some fold, and names the class.

| option | | default |
|---|---|---|
| `--dataset PATH` | the dataset | required |
| `--partition PATH` | the partition | required |
| `--out PATH` | the SNP features (`.npz`) | required |

## `dietnet compute-input-stats`

Computes the mean and standard deviation of each SNP, from the training individuals of each fold.

| option | | default |
|---|---|---|
| `--dataset PATH` | the dataset | required |
| `--partition PATH` | the partition | required |
| `--out PATH` | the statistics (`.npz`) | required |

## `dietnet train`

Trains one model for each seed and fold, and saves each as a package in `EXP_DIR/packages/seed_<seed>/fold_<fold>/`.
See [Training](training.md#train).

| option | | default |
|---|---|---|
| `--dataset PATH` | the dataset | required |
| `--partition PATH` | the partition | required |
| `--embedding PATH` | the SNP features (required, except for the MLP, which does not use them) | |
| `--input-stats PATH` | the statistics | required |
| `--config PATH` | the [config](training.md#inputs) (YAML) | required |
| `--exp-dir PATH` | where the packages, logs and results go | required |
| `--seeds TEXT` | for example `78,79,80`; each replaces the `seed` of the config | required |
| `--folds TEXT` | for example `0,1,2,3,4` | required |

Options that change the model:

| option | | default |
|---|---|---|
| `--task` | `classification` or `regression` | classification |
| `--model` | `dietnet`, or `mlp`, the [baseline](training.md#the-mlp-baseline) without the auxiliary network | dietnet |
| `--normalize / --no-normalize` | standardize the genotypes with the mean and standard deviation of each SNP | normalize |
| `--free-snp-weights FLOAT` | \(\lambda\), for [free SNP weights](training.md#free-snp-weights) (not for the MLP) | 0 (none) |
| `--free-snp-lr FLOAT` | the learning rate of the free SNP weights | the config's `lr_main` |
| `--snp-penalty-weights PATH` | \(v_j\), a value [for each SNP](training.md#a-different-penalty-for-each-snp) that multiplies \(\lambda\) (`.npy`) | 1 for every SNP |
| `--snp-sample FLOAT` | train each batch on this fraction of the SNPs, chosen again for each batch, instead of dropping SNPs for each individual (not for the MLP). Faster, because only these SNPs are computed. The config's `input_dropout` is then ignored; `1 - input_dropout` keeps its rate | 1 (off) |
| `--bn-recalibrate INTEGER` | before each validation, recompute the batch normalization statistics on this many training individuals, without dropout | 0 (off) |
| `--valid-subset N` | validate on N of the validation individuals, chosen at random once. The test set is not affected | 0 (all) |

Options that change how the genotypes are read:

| option | | default |
|---|---|---|
| `--inputs` | `packed` reads each batch from the packed file, `memmap` from the dataset (if it holds the genotypes), and `ram` copies the genotypes of the dataset into memory first. `auto` is `packed` if there is a packed file, and otherwise `memmap` above 2 GiB and `ram` below | auto |
| `--workers INTEGER` | processes that read the genotypes | the CPUs available less one, at most 8 |
| `--gpus INTEGER` | GPUs for each model, which share each batch (see [Scaling](scaling.md#time-and-memory)) | 1 |
| `--chunk-rows INTEGER` | read the individuals in chunks of this many neighbors, so that the packed file does not have to fit in memory. Individuals are then shuffled within a buffer of chunks, and not over the whole training set | 0 (off) |
| `--chunks-per-buffer INTEGER` | with `--chunk-rows`, the number of chunks in a buffer | 4 |
| `--snp-blocks INTEGER` | with `--chunk-rows` and `--snp-sample`, cut the SNPs into this many blocks of neighbors, and read a fraction `--snp-sample` of the blocks for each buffer | 0 (off) |

## `dietnet build-embedding`

Builds [SNP features](features.md) from a list of blocks.
It writes the features and a description of them, `<out>.json`.

| option | | default |
|---|---|---|
| `--dataset PATH` | the dataset | required |
| `--partition PATH` | the partition | required |
| `--spec PATH` | the list of blocks (YAML) | required |
| `--out PATH` | the SNP features (`.npz`) | required |

## `dietnet alphagenome`

Gets the AlphaGenome variant effects of every SNP, as a table for the `precomputed` block of `build-embedding`.
The coordinates and alleles of every SNP are sent to Google's API, and the API key is read from `ALPHAGENOME_API_KEY`.
If it is interrupted, run the same command again to continue.
See [SNP features](features.md#alphagenome-variant-effects).

| option | | default |
|---|---|---|
| `--dataset PATH` | the dataset; its SNP ids must be `chr:pos:REF:ALT` on GRCh38 | one of these two |
| `--bim PATH` | a PLINK `.bim` with such ids, when there is no dataset | one of these two |
| `--out PATH` | the table (`.tsv`) | required |
| `--scorers TEXT` | AlphaGenome's recommended scorers to use, separated by commas | `ATAC,DNASE,CHIP_TF,CHIP_HISTONE,CAGE,RNA_SEQ` |
| `--sequence-length` | context around each variant: `16KB`, `100KB`, `500KB` or `1MB` | `1MB` |
| `--chunk INTEGER` | SNPs saved at a time | 200 |
| `--max-workers INTEGER` | API requests at a time | 5 |

## `dietnet adjust-phenotype`

Adjusts a trait for covariates, and writes the result as labels for `create-dataset`.
The trait is regressed on the covariates, and the residuals are divided by their standard deviation.
For example, height is usually adjusted for sex and age, since a model of the autosomes cannot see sex.

Covariates that are text, such as an assessment center, become indicators.
Individuals who are missing the trait or a covariate are left out.
A summary is written to `<out>.json`.

| option | | default |
|---|---|---|
| `--phenotypes PATH` | a TSV with a header row and the sample id first | required |
| `--trait TEXT` | the column to predict | required |
| `--covariates TEXT` | the columns to adjust for, separated by commas | none |
| `--out PATH` | the labels (TSV) | required |

## `dietnet pack-genotypes`

Writes the packed file of a dataset that holds its genotypes, such as one made by version 0.4 or earlier or by the published code.
`create-dataset` writes this file itself.

| option | | default |
|---|---|---|
| `--dataset PATH` | the dataset | required |

## `dietnet info`

Prints the versions, and whether the [copy of the published code](parity.md#the-copy-of-the-published-code) is unmodified.
If it is not, it names the files and exits with status 1.
It has no options.
