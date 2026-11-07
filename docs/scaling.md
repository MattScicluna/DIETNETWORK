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

## Training on a biobank

Upstream's train.py copies every genotype into RAM before training (~83 GB for
UK Biobank's 486,748 × 169,829), then reads one sample per Python call in one
process and fills missing values in a Python loop over each batch. On an A100
that left the GPU ~87% idle even with everything in memory. dietnet offers two
ways around it:

- **`--engine upstream`** (the default on one GPU): train.py itself,
  unmodified, fed from a genotype store (`--inputs memmap|packed`) by the fast
  loader (`--loader fast`, the default: whole batches per read, loader workers,
  one vectorised missing-value fill). The model is **bit-identical** to
  train.py left alone, and it needs ~5 GB instead of the whole matrix.
- **`--engine dietnet`**: upstream's model, loss, optimisers and epoch logic
  (baseline, annealing, best-model rule, patience, the test pass, the same
  result files) in a loop of dietnet's own, which reads the packed 2-bit copy
  and decodes it on the GPU, and skips train.py's per-batch bookkeeping and
  its evaluation pass over the training set. Optionally data-parallel
  (`--gpus N`). Not bit-identical to train.py, so it is checked on held-out
  accuracy.

### Benchmarks

Simulated genotypes of UK Biobank's shape (292,048 training samples ×
169,829 SNPs), one A100 40 GB unless stated, dataset staged to the node's
local NVMe unless stated (`benchmarks/slurm/bench_train_inputs.sbatch`, Mila,
2026-10-27):

| engine | genotypes | GPUs | batch | memory used | before training | whole epoch |
|---|---|---|---|---|---|---|
| upstream, as published | RAM | 1 | 138 | 82 GB | 346 s load | 65–80 s |
| upstream | memmap, network storage | 1 | 138 | — | — | 2,923 s |
| upstream | memmap | 1 | 138 | 4.4 GB | 217 s staging | 131 s |
| upstream, fast loader | packed | 1 | 138 | 5.3 GB | 280 s staging | 82 s |
| **dietnet** | packed | 1 | 138 | 8.3 GB | 244 s staging | **13.5–14 s** |
| **dietnet** | packed | 1 | 2,048 | 8.6 GB | 244 s staging | **8.9–9.5 s** |
| dietnet | packed | 4 | 2,048 | 5.8 GB | 217 s staging | 7.8–8.6 s |

Training throughput by data path, upstream's model, samples/s
(`benchmarks/bench_train_throughput.py`):

| batch | upstream's data path | batched memmap, 8 workers | packed, decoded on CPU | packed, decoded on GPU |
|---|---|---|---|---|
| 512 | 1,811 | 11,840 | 7,847 | 31,409 |
| 2,048 | 1,580 | 11,405 | 7,775 | 40,711 |

What they say:

- Network storage is unusable for training; stage the dataset (and its packed
  copy, with `cp --preserve=timestamps`; `cp -p` fails on Narval's node disk) to `$SLURM_TMPDIR` first. Staging ~100 GB takes 3–5 min.
- The dietnet engine at the published batch size is 5–6× faster per epoch than
  upstream in RAM, in a tenth of the memory. At ~1,000–2,000 epochs per model
  that is hours, not days.
- On one node, more GPUs barely help: the epoch is bound by reading the packed
  rows (~50,000 samples/s from 8 CPUs). For an ensemble, train **one model per
  GPU** (a SLURM array, as for the published models) rather than one model on
  several.
- Larger batches are faster still, but batch size is a hyperparameter; see
  below.

### Batch size

Held-out accuracy on the published 1KGP split, seed 78, published config with
the learning rates scaled for a batch k times 138
(`benchmarks/slurm/batch_size_study.sbatch`):

| batch | learning rates | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean (sd) |
|---|---|---|---|---|---|---|---|
| 138 | published | 0.9270 | 0.9332 | 0.9272 | 0.9310 | 0.9111 | **0.926** (0.009) |
| 552 | × 2 (√4) | 0.9378 | 0.9267 | 0.9358 | 0.9181 | 0.9154 | **0.927** (0.010) |
| 552 | × 4 (linear) | 0.9378 | 0.9224 | 0.9272 | 0.9095 | 0.9197 | 0.923 (0.010) |
| 2,208 | × 4 (√16) | 0.9013 | 0.8987 | 0.9143 | 0.8750 | 0.8850 | 0.895 (0.015) |
| 2,208 | × 16 (linear) | 0.8197 | 0.7823 | 0.8522 | 0.8125 | 0.7939 | 0.812 (0.027) |

1KGP has only ~1,400 training samples per fold, so a batch of 2,208 is one
step per epoch: full-batch gradient descent, which these results penalise. A
batch four times the published one with the learning rates scaled by √4 kept
accuracy. At UK Biobank's ~290,000 training samples even a batch of 2,048 is
~140 steps per epoch; check a larger batch on one UK Biobank fold against 138
before using it for the ensemble.

### Recipe

```bash
dietnet create-dataset --plink cohort --labels labels.tsv --out dataset.h5   # CPU job
dietnet pack-genotypes --dataset dataset.h5                                # CPU job, ~30 min for UKBB
# then, per (seed, fold), one GPU job (8 CPUs, 32 GB):
cp --preserve=timestamps dataset.h5 dataset.h5.packed.* "$SLURM_TMPDIR/"
dietnet train --dataset "$SLURM_TMPDIR/dataset.h5" ... --engine dietnet --inputs packed \
    --seeds 78 --folds 0
```

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
