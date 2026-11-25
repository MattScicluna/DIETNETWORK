# Parity with the published code

Here we describe which results of `dietnet` are the same as those of the published code ([HussinLab/DIETNETWORK](https://github.com/HussinLab/DIETNETWORK)), which are different on purpose, and how this is tested.

## What is the same

| command | compared with the published code |
|---|---|
| `create-dataset` | the same individuals, SNPs, labels and genotypes, value for value |
| `partition` | runs the published script |
| `generate-embedding` | the same numbers |
| `compute-input-stats` | the same means, and standard deviations equal to floating-point rounding |
| `train` | the same network, loss, optimizers, early stopping and choice of best epoch, but [not the same weights](#training) |
| `predict` | the same scores, to floating-point rounding, when both read the same genotypes |

`predict` also treats missing SNPs as the published code does: the standardized genotypes of the SNPs that were found are multiplied by the total number of SNPs divided by the number found.

## What is different

### Training

The published `train.py` reads one individual at a time from a copy of all the genotypes in memory.
On a biobank, this takes too much memory and leaves the GPU idle most of the time (see [Scaling](scaling.md#training)).
`dietnet train` therefore reads whole batches from the packed file.

As a result, the individuals are grouped into different batches, and a model trained with `dietnet train` does not have the same weights as one trained with `train.py`, even with the same seed.
The published models themselves are not affected, since `predict` uses their weights as they are.

### Matching the SNPs of a cohort

The published code matches SNPs by name only.
When a cohort counts the other allele of a SNP, the two homozygous genotypes are then read the wrong way around.
This is common: it is the case for 56,195 of the 229,986 SNPs of the published models in the distributed 1KGP PLINK files.
`dietnet` also compares the positions and alleles, and flips the genotypes of these SNPs.

### The order of the samples in a fold

The published `partition_data.py` orders the individuals in a way that depends on Python's hash seed, which changes each time Python starts.
`dietnet partition` fixes this seed, so it gives the same partition each time.
The partition of the published models has another order, which cannot be obtained this way, so it is included as [a file](scaling.md#reproducing-the-published-models).

## How it is tested

Tests run the published code and `dietnet` on the same inputs and compare the results.

| test | what is compared |
|---|---|
| `tests/parity/test_dataset.py` | the dataset, with files that the published `create_dataset.py` made (the genotypes are written out in full for this comparison) |
| `tests/unit/test_features.py` | the SNP features, with `generate_embedding.py`: equal |
| `tests/unit/test_input_stats.py` | the statistics, with `compute_input_features_statistics.py`: means equal, standard deviations within a relative error of 1e-12 |
| `tests/parity/test_predict.py` | the scores, with `test_independent_dataset.py`, on a full cohort and on one that is missing 30% of the SNPs: within a relative error of 1e-5 |
| `tests/parity/test_predict.py` (slow) | the scores of the 15 published models, with the published `inference.py`: within a relative error of 1e-5 |
| `tests/parity/test_published.py` (slow) | the accuracy of each published model on its own test fold, with the published value (0.9214 on average): within about one individual |

If a change to `dietnet` moves one of these results, it is a bug, or a decision recorded in the [changelog](https://github.com/MattScicluna/DIETNETWORK/blob/master/CHANGELOG.md).
The tolerances are never loosened.

The tests in `tests/parity/test_predict.py` need a copy of the published code at the commit we compare with:

```bash
git init upstream && cd upstream
git fetch --depth 1 https://github.com/HussinLab/DIETNETWORK.git 5481556e472140c6d3a67363c1286f22109fb1ce
git checkout FETCH_HEAD
cd ..

DIETNET_UPSTREAM_DIR=$PWD/upstream uv run pytest
DIETNET_UPSTREAM_DIR=$PWD/upstream uv run pytest -m slow   # the published models, downloaded from Zenodo
```

Without `DIETNET_UPSTREAM_DIR`, they are skipped.
They run on every pull request, except for the slow ones.
To reuse a downloaded release between slow runs, set `DIETNET_PUBLISHED_DIR`.

## The copy of the published code

The files of the published code that we keep are in `Dietnet_train_and_evaluate/`, and they are not edited.
`dietnet` uses the network, the data handling and the partition script.
It no longer runs `train.py`, `generate_embedding.py` and `compute_input_features_statistics.py`, but we keep them for comparison.

`src/dietnet/upstream_manifest.json` lists the commit, a checksum for each file, and the files that we removed.
A test fails if a file was changed, added or brought back, and `dietnet info` makes the same check on an installed copy.

To update to a newer version of the published code:

```bash
git merge upstream/main
# delete again any file that the manifest lists as removed
python scripts/upstream_manifest.py --commit upstream/main
uv run pytest
```

Then set `UPSTREAM_SHA` in `.github/workflows/ci.yml` to the same commit, which a test checks.
