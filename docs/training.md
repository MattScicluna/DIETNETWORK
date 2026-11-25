# Training your own models

Here we describe how to train a Diet Network on your own genotypes (PLINK files) and labels.
The trained models are saved as [model packages](formats.md#model-packages), which `dietnet predict` can then apply to new cohorts.

Training takes five commands, and each one writes a file that a later one reads.
A change to one of them, such as a new set of SNP features, therefore does not require rerunning the commands before it.
A sixth command, `check`, reports the accuracy of the trained models.

| command | what it does | output |
|---|---|---|
| `dietnet create-dataset` | reads the genotypes and labels | `dataset.h5` |
| `dietnet partition` | splits the samples into folds | `partition.npz` |
| `dietnet generate-embedding` | computes the SNP features that the auxiliary network reads | `embedding.npz` |
| `dietnet compute-input-stats` | computes each SNP's mean and standard deviation | `stats.npz` |
| `dietnet train` | trains one model per seed and fold | model packages |
| `dietnet check` | reports the accuracy of each model | printed |

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
The dosages count each SNP's A1 allele, as listed in the `.bim`.

**Labels**: `labels.tsv`, with a header row and then `sample id<TAB>label`.
Every sample in the `.fam` needs a label, and duplicated ids are refused.

**Config**: `config.yaml`, the settings of the network and its training.
We suggest starting from the published ones:

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

The `seed` in the config is replaced by each value of `--seeds`, and the config used for each seed is saved to `exp/configs/config_seed<seed>.yaml`.

## The steps

### `create-dataset`

`create-dataset` converts the PLINK files into the [HDF5 dataset](formats.md#the-hdf5-dataset) that the other steps read, a block of samples at a time.
It also stores each SNP's chromosome, position and alleles, which are needed later to match a new cohort to the model.

The genotypes are always written to a packed 2-bit file next to the dataset.
With `--int8`, they are also written inside `dataset.h5`, which takes four times the disk space of the packed file.
By default, `generate-embedding` and `compute-input-stats` read the genotypes from `dataset.h5`, so without `--int8` they stop with an error.
If that copy is too large, use `compute-input-stats --engine dietnet` and the `genotype_freq` block of [`build-embedding`](features.md) (with `classes:` for frequencies per class), which read the packed file.

### `partition`

`partition` splits the samples into folds: `--nb-folds` of them (5 by default), each with its own train, validation and test samples.

With `--balanced`, the default and the setting used in the paper, every class is split separately.
At five folds, each class contributes 60% of its samples to training, 20% to validation and 20% to test.
With `--unbalanced`, labels are ignored, and the samples that are not in the test set are divided by `--train-valid-ratio`.
Both settings need class labels in the dataset, so a regression dataset needs `--labels` too.
`--seed` defaults to 23.

!!! note "Reproducing the published partition"
    The same seed gives the same partition from run to run, but not the partition that the published models were trained on.
    The samples are in a different order within each fold, and the order decides which samples share a minibatch.
    To retrain the published models bit for bit, use `reproduction/1kgp_partition_seed124.npz` (see [scaling](scaling.md#reproducing-the-published-models)).

### `generate-embedding` and `compute-input-stats`

`generate-embedding` computes each SNP's genotype frequencies in each class, which is what the auxiliary network reads (`emb`, folds × SNPs × 3·classes).
`compute-input-stats` computes each SNP's mean and standard deviation (`means_by_fold`, `sd_by_fold`).
Both are computed separately for each fold, from its training samples only, and both take `--ncpus`.

Every class needs at least one training sample in every fold.
Without one, its genotype frequencies are undefined, so `generate-embedding` stops and names the class.
When this happens, merge or drop the classes that are too small (each needs at least `--nb-folds` samples), rebuild the dataset and rerun.

To replace or extend the embedding, see [SNP features](features.md).

### `train`

`train` fits one model for every combination of `--seeds` and `--folds`.
By default, it runs the [published code](parity.md), with the same training loop, early stopping and choice of best checkpoint.
Values are comma-separated (`--seeds 78,79,80`), or the flag is repeated.

Each finished model becomes a [package](formats.md#model-packages) in `exp/packages/seed_<seed>/fold_<fold>/`, with its held-out predictions in `test_predictions.tsv`.
A directory of packages is what `predict --model` takes.
Logs go to `exp/logs/train_seed<seed>_fold<fold>.log` and results to `exp/runs/`.
If a run fails, the error shows the last lines of its log.

The options that change the model are:

- `--normalize/--no-normalize` (default on): standardize the inputs with the fold's means and standard deviations. The choice is recorded in the package and applied again at prediction.
- `--task regression`: predict a number instead of a class. See [Regression](#regression).
- `--model mlp`: train without the auxiliary network. See [The MLP baseline](#the-mlp-baseline).
- `--free-snp-weights`: add a learned term to the weights that the auxiliary network computes. See [Free SNP weights](#free-snp-weights).

The [command line reference](cli.md#dietnet-train) lists every option.

### `check`

`check` reports how accurate the models are on samples they were not trained on.

```bash
dietnet check --packages exp/packages
```

It prints the accuracy of each package on its own test fold, and then the mean and standard deviation over packages.
With `--predictions` and `--labels` instead, it scores the output of `predict` against known labels.
In both cases, `--min-accuracy` makes it exit with status 1 below a threshold, so it can be used in a script.

## The MLP baseline

The MLP baseline is a network that learns the weights of its first layer directly, without the auxiliary network.
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
The class labels are still needed because `partition` and `generate-embedding` use them.
Then train with `--task regression`.

The `predictions.tsv` of a regression ensemble has three columns: `sample`, `prediction` (the mean over models) and `sd_across_models`.

## Free SNP weights

For individual \(i\), the first layer of the network computes the input to hidden unit \(k\) as

\[
z_{ik} = b_k + \sum_j x_{ij} \left( W_{jk} + \delta_{jk} \right), \qquad W_{jk} = f_k(e_j),
\]

where \(x_{ij}\) is the standardized genotype of individual \(i\) at SNP \(j\) and \(b_k\) is a bias.
The weights \(W_{jk}\) are not learned directly.
They are computed by the auxiliary network \(f\) from the features \(e_j\) of SNP \(j\), such as its genotype frequencies in each population.
In the standard Diet Network, \(\delta_{jk} = 0\).

Computing the weights this way keeps the number of parameters small: the published model has about 23 million weights \(W_{jk}\) (229,986 SNPs and 100 hidden units) and a dataset of 2,322 individuals.
However, two SNPs with similar features get similar weights, even when one of them is associated with the trait only because it is in linkage disequilibrium with the other.

With `--free-snp-weights`, \(\delta_{jk}\) is learned from the data, so a SNP's weights can differ from what its features predict.
Since this adds one parameter per weight, \(\delta\) is penalized with a group lasso.
Training minimizes

\[
L + \lambda \sum_j v_j \lVert \delta_j \rVert_2,
\]

where \(L\) is the prediction loss and \(\delta_j\) is the vector of the values \(\delta_{jk}\) of SNP \(j\) over all hidden units \(k\).
The option's value is \(\lambda\), and \(v_j\) is 1 unless it is [set for each SNP](#a-different-penalty-for-each-snp).
The group lasso sets all the values in \(\delta_j\) to zero together, and the weights of that SNP are then exactly those of the standard Diet Network.

```bash
dietnet train --engine dietnet --free-snp-weights 1 ...      # lambda = 1
```

This needs `--engine dietnet`, since the default engine does not have \(\delta\).
Packages store \(\delta\) (package format 3), and `predict` uses it.

### Choosing lambda

When \(\lambda\) is large, \(\delta_j = 0\) for most SNPs and the model is close to the standard Diet Network.
When \(\lambda\) is small, \(\delta_j \neq 0\) for most SNPs.
The weights of all SNPs are then fitted jointly, as in penalized regression, with \(W_{jk}\) acting as a prior.
When \(\lambda\) is too small, the model overfits, and a best epoch that comes early is a sign of this.

Useful values are of order 0.1 to 10, and they do not depend on the learning rate \(\eta\) (`--free-snp-lr`, by default the config's `lr_main`).
At each step, Adam moves \(\delta_j\) by at most about \(\eta \sqrt{K}\), where \(K\) is the number of hidden units, and the penalty then pulls it back toward zero by \(\eta \lambda v_j\).
When \(\lambda\) approaches \(\sqrt{K}\) (10 for 100 hidden units, with \(v_j = 1\)), the penalty cancels the updates and \(\delta_j\) stays at zero.
Choose \(\lambda\) on the validation samples.

### A different penalty for each SNP

Prior evidence about which SNPs are causal, such as functional annotations, can be used to set \(v_j\).
`--snp-penalty-weights WEIGHTS.npy` takes one value per SNP, in the dataset's SNP order.
A value below 1 weakens the penalty, so \(\delta_j\) is more likely to be nonzero, 0 removes it, and a value above 1 strengthens it.
This is the adaptive group lasso.
Packages record the file's path and SHA-256.
