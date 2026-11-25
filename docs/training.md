# Training your own models

This page trains Diet Networks on your own PLINK files and labels.
The output is a directory of [model packages](formats.md#model-packages), which `dietnet predict` can then apply to new cohorts.

Training takes five commands, and each one writes files that a later one reads.
A change to one step, such as a new set of SNP features, therefore does not require redoing the steps before it.
A sixth command, `check`, reports the accuracy of the result.

| command | writes | upstream script |
|---|---|---|
| `dietnet create-dataset` | `dataset.h5` | replaces `create_dataset.py` |
| `dietnet partition` | `partition.npz` | `partition_data.py` |
| `dietnet generate-embedding` | `embedding.npz` | `generate_embedding.py` |
| `dietnet compute-input-stats` | `stats.npz` | `compute_input_features_statistics.py` |
| `dietnet train` | model packages | `train.py`, once per (seed, fold) |
| `dietnet check` | (prints accuracy) | |

By default, the four commands from `partition` to `train` run upstream's scripts, unmodified.
We replaced only `create_dataset.py`, and the [parity tests](parity.md) check that the replacement gives the same dataset.

```bash
dietnet create-dataset      --plink panel --labels labels.tsv --out dataset.h5 --int8
dietnet partition           --dataset dataset.h5 --out partition.npz          # 5 folds, labels balanced
dietnet generate-embedding  --dataset dataset.h5 --partition partition.npz --out embedding.npz
dietnet compute-input-stats --dataset dataset.h5 --partition partition.npz --out stats.npz
dietnet train --dataset dataset.h5 --partition partition.npz --embedding embedding.npz \
              --input-stats stats.npz --config config.yaml --exp-dir exp \
              --seeds 78,79,80 --folds 0,1,2,3,4
dietnet check --packages exp/packages            # each model's held-out test accuracy
```

The [tutorial](tutorial.ipynb) runs this pipeline on the bundled 1KGP fixture.

## Inputs

**PLINK files**: `panel.bed`, `panel.bim` and `panel.fam`.
They are read directly, without the `plink --recode A` text files that upstream needs.
The dosages count each SNP's A1 allele, as listed in the `.bim`.

**Labels**: `labels.tsv`, with a header row and then `sample id<TAB>label`.
Every sample in the `.fam` needs a label, and duplicated ids are refused.

**Config**: `config.yaml`, in upstream's format.
We suggest starting from the published one:

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

The high `input_dropout` is deliberate.
At 0.995, training keeps an average of 1,150 of the published model's 229,986 SNPs per sample, which prevents the model from relying on a small set of informative SNPs.
In the paper, this made predictions more robust to missing SNPs.

Upstream's `train.py` reads its seed from the config.
For this reason, `dietnet train` writes one copy of the config per seed, to `exp/configs/config_seed<seed>.yaml`.

## The steps

### `create-dataset`

This step converts the PLINK files into the [HDF5 dataset](formats.md#the-hdf5-dataset) that the other steps read, a block of samples at a time.
It also stores each SNP's chromosome, position and alleles, which are needed later to match a new cohort to the model.

The genotypes are always written to a packed 2-bit file next to the dataset.
With `--int8`, they are also written inside `dataset.h5`, which takes four times the disk space of the packed file.
Upstream's `generate-embedding` and `compute-input-stats` scripts can only read the second form, and the commands stop with an error if it is missing.
If that copy is too large, use `compute-input-stats --engine dietnet` and the `genotype_freq` block of [`build-embedding`](features.md) (with `classes:` for frequencies per class), which read the packed file.

### `partition`

This step splits the samples into folds: `--nb-folds` of them (5 by default), each with its own train, validation and test samples.

With `--balanced`, the default and the setting used in the paper, every class is split separately.
At five folds, each class contributes 60% of its samples to training, 20% to validation and 20% to test.
With `--unbalanced`, labels are ignored, and the samples that are not in the test set are divided by `--train-valid-ratio`.
Both settings need class labels in the dataset, so a regression dataset needs `--labels` too.
`--seed` defaults to 23.

!!! note "Reproducing the published partition"
    The same seed gives the same partition from run to run, but it does not give the partition that the published models were trained on.
    `partition_data.py` iterates over `set(labels)`, whose order depends on Python's per-process hash seed, and that order decides which samples share a minibatch.
    `dietnet` fixes the hash seed (`PYTHONHASHSEED=0`) for every upstream script, and the published partition was made with a different order.
    To retrain the published models bit for bit, use `reproduction/1kgp_partition_seed124.npz` (see [scaling](scaling.md#reproducing-the-published-models)).

### `generate-embedding` and `compute-input-stats`

These two steps summarize each SNP.
`generate-embedding` computes its genotype frequencies in each class, which is what the auxiliary network reads (`emb`, folds × SNPs × 3·classes).
`compute-input-stats` computes its mean and standard deviation (`means_by_fold`, `sd_by_fold`).
Both are computed separately for each fold, from its training samples only, and both take `--ncpus`.

Every class needs at least one training sample in every fold.
Upstream's script divides by the number of training samples in the class, so an empty class gives NaN, which spreads through the whole model.
If the empty class is the last one, its columns are silently dropped instead.
`generate-embedding` therefore stops and names the empty classes.
When this happens, merge or drop the classes that are too small (each needs at least `--nb-folds` samples), rebuild the dataset and rerun.
`train` makes the same check, and also refuses NaN in the statistics.

To replace or extend the embedding, see [SNP features](features.md).

### `train`

This step trains one model for every combination of `--seeds` and `--folds`.
The model, the training loop, early stopping and the choice of best checkpoint are all upstream's.
Values are comma-separated (`--seeds 78,79,80`), or the flag is repeated.

Each finished model becomes a [package](formats.md#model-packages) in `exp/packages/seed_<seed>/fold_<fold>/`, with its held-out predictions in `test_predictions.tsv`.
A directory of packages is what `predict --model` takes.
Logs go to `exp/logs/train_seed<seed>_fold<fold>.log` and results to `exp/runs/`.
If a run fails, the error shows the last lines of its log.

The options that change the model are:

- `--normalize/--no-normalize` (default on): standardize the inputs with the fold's means and standard deviations. The choice is recorded in the package and applied again at prediction.
- `--task regression`: predict a number instead of a class. See [Regression](#regression).
- `--model mlp`: train without the auxiliary network. See [The MLP baseline](#the-mlp-baseline).
- `--free-snp-weights`: let SNPs depart from what the auxiliary network predicts. See [Free SNP weights](#free-snp-weights).

The [command line reference](cli.md#dietnet-train) lists every option.

### `check`

This step reports how well the models do on samples they were not trained on.

```bash
dietnet check --packages exp/packages
```

It prints the accuracy of each package on its own test fold, and then the mean and standard deviation over packages.
With `--predictions` and `--labels` instead, it scores the output of `predict` against known labels.
In both cases, `--min-accuracy` makes it exit with status 1 below a threshold, which is how the integration tests and the reproduction use it.

## The MLP baseline

The MLP baseline is upstream's network without the auxiliary network, trained by the same `train.py`.
It is the comparison point for any change to the SNP features.

```bash
dietnet train --model mlp --dataset dataset.h5 --partition partition.npz \
              --input-stats stats.npz --config config.yaml --exp-dir exp_mlp \
              --seeds 78,79,80 --folds 0,1,2,3,4
```

Since there is no auxiliary network, giving `--embedding` is an error and the packages have no `embedding.npz`.
`predict` runs them like any other package.

## Regression

To predict a quantitative trait, give `create-dataset` two label files: `--regression-labels`, a TSV of `sample id<TAB>value`, and `--labels`, the class labels.
The class labels are still needed because upstream builds the partition and the embedding from them.
Then train with `--task regression`.

The `predictions.tsv` of a regression ensemble has three columns: `sample`, `prediction` (the mean over models) and `sd_across_models`.

## Free SNP weights

In a Diet Network, the first layer's weights are not learned directly.
Instead, the auxiliary network computes the weights of SNP j from that SNP's features, `W_j = aux(features_j)`: a column with one weight per hidden unit (100 by default).
This keeps the model small.
For the published models, learning the first layer directly would mean fitting about 23 million weights (229,986 SNPs × 100 hidden units) with a dataset of 2,322 samples.

However, it also ties every SNP to what its features say.
Two SNPs with similar features get similar weights, even when one of them only tags the other through linkage disequilibrium.

Free SNP weights loosen this tie by adding a term that each SNP can learn for itself:

    W_j = aux(features_j) + delta_j

They are turned on by `--free-snp-weights LAMBDA`, which needs `--engine dietnet` because upstream's `train.py` has no such term:

```bash
dietnet train --engine dietnet --free-snp-weights 1 ...      # LAMBDA = 1
```

`delta` has one entry per first-layer weight, starts at zero and has its own Adam optimizer (`--free-snp-lr`).
After each update, a group-lasso proximal step shrinks each SNP's column as a whole, `delta_j <- delta_j * max(0, 1 - lr * LAMBDA / ||delta_j||)`, and a column that is short enough becomes exactly zero.
The penalty therefore pulls a SNP's weights toward what the auxiliary network predicts, not toward zero.
A SNP whose `delta_j` is zero is left entirely to the auxiliary network.
Packages carry `delta` (package format 3), and `predict` applies it.

### Choosing LAMBDA

LAMBDA sets how far SNPs may depart from the auxiliary network.

When LAMBDA is large, most columns stay at zero and the model is close to the plain Diet Network.
When it is small, every SNP can move away from what its features predict.
The model then fits the SNP effects jointly, as penalized regression does, with the auxiliary network acting as a prior.
When it is too small, the model overfits, and a best epoch that comes early is a sign of this.

Useful values are of order 0.1 to 10, rather than the small values usual for a lasso.
This is because Adam moves each entry by about the learning rate whatever the gradient, so LAMBDA is relative to the learning rate.
Choose it on the validation samples.

### Penalizing some SNPs more than others

If there is prior evidence about which SNPs are causal, from functional annotations for example, the penalty can reflect it.
`--snp-penalty-weights WEIGHTS.npy` multiplies LAMBDA by a separate weight for each SNP, in the dataset's SNP order:

| weight | effect on the SNP |
|---|---|
| 0 | not penalized |
| below 1 | departs from the auxiliary network more easily |
| 1 | unchanged |
| above 1 | held closer to the auxiliary network |

This is the adaptive group lasso.
Annotations can then decide which SNPs the free weights may pick, in the same way that they decide the auxiliary network's weights.
Packages record the file's path and SHA-256.
