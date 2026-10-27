# Reproducing the published 1KGP models

Rochefort-Boulanger, Scicluna et al., AJHG 2026. Inputs are the Zenodo release
`Dietnet_training/` (`dietnet published --dest DIR` downloads it) and the
partition in this directory.

## The partition: `1kgp_partition_seed124.npz`

The published models' 5-fold split: upstream's `partition_data.py` with
`--balance-class --seed 124` over the published `1KGP_dataset.hdf5`, the seed
recorded in the original experiment names (`DATASPLITSEED124_MODELSEED78/79/80`).
Taken from the lab's 1KGP directory on the cluster, and verified, not assumed:

- recomputing each fold's SNP means and SDs and genotype×class embedding from
  this file's training indices reproduces Zenodo's `normalization_stats.npz`
  and `SNPs_embedding.npz` exactly (differences 0, SDs to 1e-14). The seed-123
  partition from the same directory does not;
- `dietnet partition --seed 124` regenerates the same fold *membership*, but
  not the order within folds: upstream iterates `set(labels)`, whose order
  depends on Python's hash seed. Order changes which samples share a minibatch,
  so a bit-for-bit retrain needs this file, not a regenerated one.

SHA-256 `e8f7799b00d0667fbedeaf26f498d96d4c6d0801c9e53b92ac73320c98790d71`.
Indices are rows of the published `1KGP_dataset.hdf5`.

## Published held-out accuracy

The 15 published models on their own test folds (the published inputs,
`tests/parity/test_published.py`):

| fold | seed 78 | seed 79 | seed 80 |
|---|---|---|---|
| 0 | 0.9185 | 0.9206 | 0.9356 |
| 1 | 0.9224 | 0.9073 | 0.9289 |
| 2 | 0.9336 | 0.9315 | 0.9465 |
| 3 | 0.9289 | 0.9095 | 0.9246 |
| 4 | 0.9089 | 0.9002 | 0.9046 |

Mean 0.9214 (sd 0.013), matching the paper's Figure 1A (input dropout 0.995,
0% missing, averaged over 15 models: about 0.92).

Through `dietnet predict` on the distributed 1KGP PLINK files instead, the mean
is the same to four decimals: 56,195 of the 229,986 SNPs count the other
allele there and are flipped back by allele-aware matching, and 149 whose
alleles disagree are treated as missing. The published models count the ALT
allele at every SNP.

## Retraining

On a SLURM cluster (paths from the environment; nothing cluster-specific is
committed):

```bash
RELEASE=/path/to/Dietnet_training DIETNET_VENV=/path/to/.venv EXP_DIR=/path/to/run_dir \
  ACCOUNT=my-account bash reproduction/submit_retrain.sh
```

`prep.sbatch` recomputes the embedding and statistics with `dietnet` and stops
unless they equal Zenodo's; `retrain.sbatch` is a 15-task array, one
(seed, fold) per GPU, with the published configs. Then:

```bash
dietnet check --packages $EXP_DIR/exp/packages
```

## Retrained with dietnet (2026-10-26)

All 15 retrained on Narval A100s (19-58 min each) with `submit_retrain.sh`:
upstream's train.py through `dietnet train`, the published configs and this
split. The recomputed per-fold embedding, means and SDs equalled Zenodo's
exactly. Held-out accuracy on the same test folds, and how often the retrained
and published model of the same (seed, fold) predict the same population:

| seed | fold | published | retrained | agreement |
|---|---|---|---|---|
| 78 | 0 | 0.9185 | 0.9442 | 0.9528 |
| 78 | 1 | 0.9224 | 0.9246 | 0.9612 |
| 78 | 2 | 0.9336 | 0.9400 | 0.9593 |
| 78 | 3 | 0.9289 | 0.9332 | 0.9806 |
| 78 | 4 | 0.9089 | 0.8980 | 0.9436 |
| 79 | 0 | 0.9206 | 0.9356 | 0.9592 |
| 79 | 1 | 0.9073 | 0.9353 | 0.9569 |
| 79 | 2 | 0.9315 | 0.9422 | 0.9700 |
| 79 | 3 | 0.9095 | 0.9138 | 0.9461 |
| 79 | 4 | 0.9002 | 0.9132 | 0.9219 |
| 80 | 0 | 0.9356 | 0.9335 | 0.9850 |
| 80 | 1 | 0.9289 | 0.9332 | 0.9526 |
| 80 | 2 | 0.9465 | 0.9400 | 0.9807 |
| 80 | 3 | 0.9246 | 0.9203 | 0.9784 |
| 80 | 4 | 0.9046 | 0.9154 | 0.9631 |
| | **mean** | **0.9214** (sd 0.013) | **0.9282** (sd 0.013) | **0.9608** |

Retrained minus published, paired by (seed, fold): +0.0067 (sd 0.011),
retrained higher in 11 of 15. Identical weights were not expected -- GPU
nondeterminism, and torch 2.14 here against the original's older torch -- and
the reproduction is of the method's accuracy, which holds.

## MLP baseline (2026-10-26)

Upstream's MLP on the same split, configs and per-fold statistics, with no
auxiliary network: `MODEL=mlp` on `retrain.sbatch`, with `stats.npz` and
`configs/` from the retrain run and no embedding. One hidden layer of 100, input
dropout 0.995, as configured for the Diet Network. All 15 trained on Narval
A100s in 20-64 min, early-stopped at epochs 2,024-6,235.

| seed | fold | published | retrained | MLP | MLP agrees with retrained |
|---|---|---|---|---|---|
| 78 | 0 | 0.9185 | 0.9442 | 0.9313 | 0.9464 |
| 78 | 1 | 0.9224 | 0.9246 | 0.9181 | 0.9353 |
| 78 | 2 | 0.9336 | 0.9400 | 0.9379 | 0.9358 |
| 78 | 3 | 0.9289 | 0.9332 | 0.9224 | 0.9353 |
| 78 | 4 | 0.9089 | 0.8980 | 0.9262 | 0.9306 |
| 79 | 0 | 0.9206 | 0.9356 | 0.9034 | 0.9120 |
| 79 | 1 | 0.9073 | 0.9353 | 0.9159 | 0.9246 |
| 79 | 2 | 0.9315 | 0.9422 | 0.9036 | 0.9443 |
| 79 | 3 | 0.9095 | 0.9138 | 0.8966 | 0.9116 |
| 79 | 4 | 0.9002 | 0.9132 | 0.9024 | 0.9111 |
| 80 | 0 | 0.9356 | 0.9335 | 0.9185 | 0.9227 |
| 80 | 1 | 0.9289 | 0.9332 | 0.9095 | 0.9353 |
| 80 | 2 | 0.9465 | 0.9400 | 0.8994 | 0.9229 |
| 80 | 3 | 0.9246 | 0.9203 | 0.8879 | 0.9009 |
| 80 | 4 | 0.9046 | 0.9154 | 0.9024 | 0.9241 |
| | **mean** | **0.9214** | **0.9282** | **0.9117** (sd 0.014) | **0.9262** |

Paired by (seed, fold), the MLP is 1.65 points below the retrained Diet Network
(sd 1.7; MLP higher in 1 of 15) and 0.97 below the published models (sd 1.8;
higher in 5 of 15). This is the reference for the SNP-feature experiments: a
feature set earns its place by how much of that gap it keeps or widens.
