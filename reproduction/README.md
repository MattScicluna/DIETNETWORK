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
