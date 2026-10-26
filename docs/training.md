# Training your own models

Each step writes one file the next reads. Apart from `create-dataset` (and the
packaging at the end of `train`), every step runs one of upstream's scripts,
unmodified, in a subprocess:

| command | writes | upstream script |
|---|---|---|
| `dietnet create-dataset` | `dataset.h5` | (replaces `create_dataset.py`, same output) |
| `dietnet partition` | `partition.npz` | `partition_data.py` |
| `dietnet generate-embedding` | `embedding.npz` | `generate_embedding.py` |
| `dietnet compute-input-stats` | `stats.npz` | `compute_input_features_statistics.py` |
| `dietnet train` | model packages | `train.py`, once per (seed, fold) |
| `dietnet check` | (prints accuracy) | |

```bash
dietnet create-dataset      --plink panel --labels labels.tsv --out dataset.h5
dietnet partition           --dataset dataset.h5 --out partition.npz          # 5 folds, labels balanced
dietnet generate-embedding  --dataset dataset.h5 --partition partition.npz --out embedding.npz
dietnet compute-input-stats --dataset dataset.h5 --partition partition.npz --out stats.npz
dietnet train --dataset dataset.h5 --partition partition.npz --embedding embedding.npz \
              --input-stats stats.npz --config config.yaml --exp-dir exp \
              --seeds 78,79,80 --folds 0,1,2,3,4
dietnet check --packages exp/packages            # each model's held-out test accuracy
```

The [tutorial](tutorial.ipynb) runs exactly this on the bundled 1KGP fixture.

## Inputs

**PLINK files.** `panel.bed/.bim/.fam`, read directly (no `plink --recode A`
text round-trip). The dataset's dosages count each SNP's `.bim` A1 allele.

**Labels.** `labels.tsv`: a header row, then `sample id<TAB>label`. Every
sample in the `.fam` needs a label; duplicated ids are refused.

**Config.** `config.yaml` is upstream's training config. The published one is
the place to start:

```yaml
batch_size: 138
epochs: 8000
input_dropout: 0.995
dropout_main: 0.1
lr_aux: 0.0006
lr_main: 0.0003
learning_rate_annealing: 0.99
nb_hidden_u_aux: [100, 100]
nb_hidden_u_main: [100]
patience: 2000
seed: 78          # overridden per --seeds
uniform_init_limit: 0.02
```

`train` writes one copy per seed to `exp/configs/config_seed<seed>.yaml`,
because upstream's `train.py` reads its seed from the config.

## The steps

### `create-dataset`

Writes upstream's HDF5 schema ([formats](formats.md#the-hdf5-dataset)) from the
`.bed`, a block of samples at a time. The result equals what upstream's
`create_dataset.py` writes from `plink --recode A` output, which the parity
tests check. It also stores each SNP's chromosome, position and alleles, which
matching a cohort later needs.

### `partition`

Upstream's `partition_data.py`: `--nb-folds` folds (default 5), each split
into train, validation and test. With `--balanced` (the default, as in the
paper) every class is split separately, 1/`nb-folds` of it to validation and
1/`nb-folds` to test, rotating with the fold (60/20/20 at five folds).
`--unbalanced` ignores labels and divides what is not test by
`--train-valid-ratio`. `--seed` defaults to 23. Partitioning needs class
labels, so a regression dataset still needs `--labels`.

!!! note "Order within a fold"
    `partition_data.py` iterates over `set(labels)`, whose order depends on
    Python's per-process hash seed, and that order decides which samples share
    a minibatch. `dietnet` runs every upstream script with `PYTHONHASHSEED=0`,
    so the same seed gives the same partition from run to run. It is still not
    the order the published partition was made with: to retrain the published
    models bit for bit, use `reproduction/1kgp_partition_seed124.npz` (see
    [scaling](scaling.md#reproducing-the-published-models)).

### `generate-embedding` and `compute-input-stats`

Per fold, from that fold's training samples only: the genotype frequencies per
class that the auxiliary network reads (`emb`, folds × SNPs × 3·classes), and
each SNP's mean and standard deviation (`means_by_fold`, `sd_by_fold`). Both
take `--ncpus`.

`generate-embedding` stops, naming them, if a class has no training samples in
some fold. Upstream's script divides by that count, so such a class gives NaN
(which spreads through the whole model) or, if it is the last class, silently
drops its columns. Merge or drop classes too small to appear in every fold's
training split (at least `--nb-folds` samples each), rebuild the dataset and
rerun. `train` makes the same check, and also refuses NaN in the statistics.

To replace or extend the embedding, see [SNP features](features.md).

### `train`

Runs upstream's `train.py` for every combination of `--seeds` and `--folds`.
Values are comma-separated (`--seeds 78,79,80`) or the flag is repeated. The
model, loop, early stopping and choice of best checkpoint are all upstream's.
Each run's log goes to `exp/logs/train_seed<seed>_fold<fold>.log` and its
results to `exp/runs/`; if a run fails, the error shows the last lines of its
log.

Each finished (seed, fold) becomes a [model package](formats.md#model-packages)
in `exp/packages/seed_<seed>/fold_<fold>/`, with that fold's held-out
predictions in `test_predictions.tsv`. A directory of packages is what
`predict --model` takes.

Other options:

- `--normalize/--no-normalize` (default on): standardise inputs with the
  fold's means and SDs. Recorded in the package, and applied the same way at
  prediction.
- `--task regression`: see [below](#regression).
- `--model mlp`: see [below](#the-mlp-baseline).

### `check`

```bash
dietnet check --packages exp/packages
```

prints each package's accuracy on its own test fold, from its
`test_predictions.tsv`, and their mean and SD. With `--predictions` and
`--labels` instead, it scores a `predict` output against known labels. Either
way, `--min-accuracy` makes it exit with status 1 below a threshold, which is
how the integration tests and the reproduction use it.

## The MLP baseline

```bash
dietnet train --model mlp --dataset dataset.h5 --partition partition.npz \
              --input-stats stats.npz --config config.yaml --exp-dir exp_mlp \
              --seeds 78,79,80 --folds 0,1,2,3,4
```

Upstream's MLP, trained by the same `train.py`: no auxiliary network, so no
`--embedding` (giving one is an error). It is the comparison point for any
change to the SNP features. Its packages have no `embedding.npz`, and `predict`
runs them like any other.

## Regression

Give `create-dataset` both `--labels` (class labels: upstream builds the
embedding from them) and `--regression-labels` (a TSV of `sample id<TAB>value`),
then train with `--task regression`. A regression ensemble's `predictions.tsv`
has `sample`, `prediction` (the mean over models) and `sd_across_models`.
