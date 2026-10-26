# Scaling and clusters

`dietnet` is built to apply the published models to biobank cohorts, and the
work splits along the hardware it needs:

| step | needs | why |
|---|---|---|
| `match` | CPU and I/O | reads the `.bed` once, a block of samples at a time (1 GiB per block), and writes the matched cohort as contiguous runs of rows |
| `predict` | a GPU, ideally | memory-maps the matched cohort and reads each batch of samples once for every model |
| `train` | a GPU | upstream's `train.py`, which loads the training set into memory |

Measured on UK Biobank (486,748 samples) on Narval:

| step | where | wall time | peak memory |
|---|---|---|---|
| `match` onto the published models (229,986 SNPs) | 8 CPUs | 4 min 27 s | 23 GB |
| `predict`, the 15 published models | one A100 | 694 s | memory-mapped |

!!! note
    That fileset holds only 5,810 of the published models' SNPs, so the run
    measures scale, not accuracy. Real predictions from the published models
    need genotypes at the published SNPs (see the [plan](refactor-plan.md)).

## Rules for cluster work

- Heavy work goes to SLURM jobs on compute nodes, never the login node.
- Cohort data, and anything derived from it (datasets, matched cohorts,
  predictions, trained packages), stays on the cluster. The repository's
  pre-commit hook refuses genotype files, HDF5 datasets, `.npy`/`.npz` arrays,
  `.pt` weights and absolute cluster paths.
- No cluster path is committed: every job below takes its paths from
  environment variables and stops with a message naming any that is unset.
  Pass the account to `sbatch`.

## Prediction jobs: `benchmarks/slurm/`

`bench_predict.sbatch` times every pipeline step (wall time and peak RSS, with
GNU `time`): it trains a throwaway one-fold model for 3 epochs on a reference
panel (the work per step is the same; accuracy is not the point), then matches
a cohort to it.

```bash
TRAIN_PREFIX=/path/panel TRAIN_LABELS=/path/panel_labels.tsv \
TEST_PREFIX=/path/cohort DIETNET_VENV=/path/.venv RUN_DIR=/path/to/run_dir \
sbatch --account=ACCOUNT benchmarks/slurm/bench_predict.sbatch
```

`predict_gpu.sbatch` is the GPU half: inference after `match` has written the
matched cohort to `WORK_DIR`.

```bash
MODEL_DIR=/path/packages TEST_PREFIX=/path/cohort WORK_DIR=/path/to/run_dir/work \
OUT=/path/to/run_dir/predictions.tsv DIETNET_VENV=/path/.venv \
sbatch --account=ACCOUNT benchmarks/slurm/predict_gpu.sbatch
```

To apply the published models, run `dietnet match` as a CPU job with the same
`--model` and `--work-dir`, then this.

## The genotype I/O benchmark

`benchmarks/bench_genotype_io.py` settled which on-disk format the pipeline
builds once from PLINK and reads in every later step: PLINK decoded each time,
`.npy`, or HDF5 contiguous, row-chunked, or compressed. It times a full load
and random minibatch reads, and reports the first (cold) run separately from
the median.

```bash
PLINK_PREFIX=/path/to/cohort OUT_JSON=/path/to/results.json \
DIETNET_VENV=/path/to/.venv [WORKDIR=/path/to/workdir] [FORMATS=...] \
sbatch --account=ACCOUNT benchmarks/slurm/bench_io.sbatch
```

`WORKDIR` defaults to the node-local `$SLURM_TMPDIR`; point it at scratch to
time the shared filesystem. It needs about 2.2× the int8 matrix (samples × SNPs
bytes) in memory. The outcome, contiguous HDF5, and the numbers are in the
[plan](refactor-plan.md#on-disk-intermediate-benchmark).

## Reproducing the published models

`reproduction/` retrains the 15 published models on a SLURM cluster from the
Zenodo release (`dietnet published --dest DIR` downloads it to
`DIR/download/Dietnet_training`) and the published split:

```bash
RELEASE=/path/to/Dietnet_training DIETNET_VENV=/path/to/.venv EXP_DIR=/path/to/run_dir \
  ACCOUNT=my-account bash reproduction/submit_retrain.sh
```

- `prep.sbatch` (CPU) recomputes the per-fold embedding and statistics with
  `dietnet generate-embedding` and `compute-input-stats`, and stops unless
  they equal Zenodo's.
- `retrain.sbatch` (GPU), once prep succeeds, is a 15-task array: one
  (seed, fold) per task, with the published config for that seed.

Then:

```bash
dietnet check --packages $EXP_DIR/exp/packages
```

The split is `reproduction/1kgp_partition_seed124.npz`: upstream's
`partition_data.py --balance-class --seed 124` over the published
`1KGP_dataset.hdf5`. Its training indices reproduce Zenodo's per-fold
statistics and embedding exactly. `dietnet partition --seed 124` regenerates the
same fold membership but not the order within folds, which decides minibatches,
so a retrain uses the file.

Retrained on Narval A100s (19 to 58 minutes per model), the mean held-out
accuracy was 0.9282 against the published 0.9214; per-model results are in
`reproduction/README.md`.
