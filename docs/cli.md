# Command line

```bash
dietnet --help
dietnet <command> --help
dietnet --version
```

| command | does |
|---|---|
| [`create-dataset`](#dietnet-create-dataset) | PLINK + labels → upstream's HDF5 dataset |
| [`partition`](#dietnet-partition) | dataset → folds (upstream `partition_data.py`) |
| [`generate-embedding`](#dietnet-generate-embedding) | per-fold genotype × class frequencies (upstream `generate_embedding.py`) |
| [`build-embedding`](#dietnet-build-embedding) | the auxiliary network's input from several feature sources |
| [`alphagenome`](#dietnet-alphagenome) | AlphaGenome variant effects per SNP, for `build-embedding` |
| [`compute-input-stats`](#dietnet-compute-input-stats) | per-fold SNP means and SDs (upstream `compute_input_features_statistics.py`) |
| [`train`](#dietnet-train) | upstream `train.py` per (seed, fold) → model packages |
| [`match`](#dietnet-match) | a cohort on the models' SNPs, ahead of `predict` |
| [`predict`](#dietnet-predict) | a cohort through every selected model, combined |
| [`check`](#dietnet-check) | accuracy against labels, or each package's held-out accuracy |
| [`published`](#dietnet-published) | the 15 published models, as packages |
| [`info`](#dietnet-info) | versions, and whether the upstream code is intact |

Options that take several integers (`--seeds`, `--folds`) accept
comma-separated values (`--seeds 78,79,80`) or the flag repeated
(`--seeds 78 --seeds 79`). Paths marked *existing* must exist.

## `dietnet create-dataset`

Build upstream's HDF5 dataset from PLINK, without the text round-trip
([schema](formats.md#the-hdf5-dataset)). At least one of `--labels` and
`--regression-labels` is required.

| option | | default |
|---|---|---|
| `--plink TEXT` | PLINK prefix (`.bed/.bim/.fam`) | required |
| `--labels PATH` | *existing* TSV: sample id, class label | |
| `--regression-labels PATH` | *existing* TSV: sample id, value | |
| `--out PATH` | output `.h5` | required |

## `dietnet partition`

Split samples into folds (upstream `partition_data.py`).

| option | | default |
|---|---|---|
| `--dataset PATH` | *existing* dataset | required |
| `--out PATH` | output `.npz` (`.npz` is appended if missing) | required |
| `--nb-folds INTEGER` | number of folds | 5 |
| `--seed INTEGER` | random seed | 23 |
| `--train-valid-ratio FLOAT` | train share of train + valid; unbalanced split only | 0.75 |
| `--balanced / --unbalanced` | balance class labels across splits, as in the paper | balanced |

## `dietnet generate-embedding`

Per-fold genotype × class frequency embedding (upstream
`generate_embedding.py`). Stops, naming them, if a class has no training
samples in some fold.

| option | | default |
|---|---|---|
| `--dataset PATH` | *existing* dataset | required |
| `--partition PATH` | *existing* partition | required |
| `--out PATH` | output `.npz` | required |
| `--ncpus INTEGER` | processes | 1 |

## `dietnet build-embedding`

SNP features for the auxiliary network from several sources; writes
`embedding.npz` and `embedding.npz.json`. See [SNP features](features.md).

| option | | default |
|---|---|---|
| `--dataset PATH` | *existing* dataset | required |
| `--partition PATH` | *existing* partition | required |
| `--spec PATH` | *existing* YAML list of feature blocks | required |
| `--out PATH` | output `.npz` | required |

## `dietnet alphagenome`

AlphaGenome variant effects for every SNP, through the hosted API, with the key
in `ALPHAGENOME_API_KEY`. Writes a table for `build-embedding`'s `precomputed`
provider (`orient: effect`). Resumable: rerun the same command after an
interruption. Needs the `alphagenome` extra, and sends every SNP's coordinates
and alleles to Google's API; see [SNP features](features.md#alphagenome-variant-effects).

| option | | default |
|---|---|---|
| `--dataset PATH` | *existing* dataset; SNP ids must be `chr:pos:REF:ALT` on GRCh38 | one of these two |
| `--bim PATH` | a PLINK `.bim` instead: only its SNP ids are read (same `chr:pos:REF:ALT` form), so no genotypes are needed | one of these two |
| `--out PATH` | feature table (`.tsv`) | required |
| `--scorers TEXT` | AlphaGenome recommended scorers, comma-separated | `ATAC,DNASE,CHIP_TF,CHIP_HISTONE,CAGE,RNA_SEQ` |
| `--sequence-length` | context around each variant: `16KB`, `100KB`, `500KB` or `1MB` | `1MB` |
| `--chunk INTEGER` | SNPs per saved part | 200 |
| `--max-workers INTEGER` | parallel API requests | 5 |

## `dietnet pack-genotypes`

Write a 2-bit copy of the dataset's genotypes next to it
(`<dataset>.packed.npy`, with a `.packed.json` recording what it was built
from), for `train --inputs packed`: a quarter of the bytes, read a batch at a
time. Rerun it after the dataset changes; a stale copy is refused.

| option | | default |
|---|---|---|
| `--dataset PATH` | *existing* dataset | required |

## `dietnet compute-input-stats`

Per-fold SNP means and SDs (upstream `compute_input_features_statistics.py`).

| option | | default |
|---|---|---|
| `--dataset PATH` | *existing* dataset | required |
| `--partition PATH` | *existing* partition | required |
| `--out PATH` | output `.npz` | required |
| `--ncpus INTEGER` | processes | 1 |

## `dietnet train`

Train each (seed, fold) with upstream `train.py`; package the results under
`EXP_DIR/packages/seed_<seed>/fold_<fold>/`. See [training](training.md#train).

| option | | default |
|---|---|---|
| `--dataset PATH` | *existing* dataset | required |
| `--partition PATH` | *existing* partition | required |
| `--embedding PATH` | *existing* auxiliary-network input: required for the Diet Network, refused for the MLP | |
| `--input-stats PATH` | *existing* per-fold statistics | required |
| `--config PATH` | *existing* upstream training config (YAML) | required |
| `--exp-dir PATH` | runs, logs, configs and packages | required |
| `--seeds TEXT` | e.g. `78,79,80`; each overrides the config's `seed` | required |
| `--folds TEXT` | e.g. `0,1,2,3,4` | required |
| `--task` | `classification` or `regression` | classification |
| `--normalize / --no-normalize` | standardise inputs with the fold's means and SDs | normalize |
| `--model` | `dietnet`, upstream's Diet Network, or `mlp`, its MLP baseline (no auxiliary network) | dietnet |
| `--inputs` | where training reads genotypes: `ram` copies them all into memory first (upstream), `memmap` reads each batch from the dataset file, `packed` from its 2-bit copy (`pack-genotypes`); same model in every case. `auto` uses the packed copy if there is one, else memory-maps above 2 GiB | auto |
| `--loader` | `fast` reads each batch in one call with loader workers and fills missing values in one operation; `upstream` leaves train.py's loader alone. Same model | fast |
| `--workers INTEGER` | loader processes for `--loader fast` | the CPUs available, less one, at most 8 |
| `--gpus INTEGER` | above 1, train each model data-parallel over this many GPUs on one node (CPU processes where there are none): upstream's model and epoch logic, the config's `batch_size` split across them; not bit-identical to train.py, see Scaling | 1 |

## `dietnet match`

Match a cohort to the models' SNPs, ahead of `predict`. CPU and I/O only: run it
as its own job, then `dietnet predict` with the same `--work-dir` on a GPU,
which finds the matched cohort and goes straight to inference. Prints the
matching counts, and warns if fewer than 1,000 model SNPs were found.

| option | | default |
|---|---|---|
| `--model PATH` | *existing* package, or a directory of packages | required |
| `--plink TEXT` | cohort PLINK prefix | required |
| `--work-dir PATH` | where the matched cohort HDF5 goes | required |
| `--seeds TEXT` | only these seeds | all |
| `--folds TEXT` | only these folds | all |

## `dietnet predict`

Predict a cohort with every selected model and combine them
([output](formats.md#predictions)). Warns if fewer than 1,000 model SNPs were
found.

| option | | default |
|---|---|---|
| `--model PATH` | *existing* package, or a directory of packages | required |
| `--plink TEXT` | cohort PLINK prefix | required |
| `--out PATH` | predictions TSV | required |
| `--work-dir PATH` | where the matched cohort HDF5 goes, or already is | next to `--out` |
| `--seeds TEXT` | only these seeds | all |
| `--folds TEXT` | only these folds | all |
| `--batch-size INTEGER` | samples read and scored at a time | 1024 |
| `--device TEXT` | `cpu`, `cuda`, ... | `cuda` if available |
| `--scores PATH` | also save per-model scores ([`.npz`](formats.md#per-model-scores)) | |

## `dietnet check`

Accuracy of predictions against labels, overall and per class; or, with
`--packages`, each package's accuracy on its own held-out test fold (from the
`test_predictions.tsv` that `train` writes). Give `--packages`, or
`--predictions` with `--labels`.

| option | | default |
|---|---|---|
| `--predictions PATH` | *existing* output of `dietnet predict` | |
| `--packages PATH` | *existing* directory of packages from `train` | |
| `--labels PATH` | *existing* TSV: sample id, label | |
| `--min-accuracy FLOAT` | exit with status 1 below this | |

## `dietnet published`

Fetch the 15 published models (Zenodo record 21133813) as model packages, in
`DEST/packages/`. The archive is checked against Zenodo's checksum.

| option | | default |
|---|---|---|
| `--dest PATH` | where the download and packages go | required |
| `--release PATH` | *existing* already unpacked `Dietnet_training/` (skips the download) | |

## `dietnet info`

Versions, and whether the upstream code is exactly the pinned commit. Exits
with status 1, naming the files, if not. No options.
