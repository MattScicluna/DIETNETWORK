# Scaling and clusters

Here we describe how much time and memory `dietnet` needs on a biobank, how to run it on a SLURM cluster, and how to retrain the published models.

## Prediction

Prediction has two steps that need different hardware, so they can be two jobs.
`match` finds the models' SNPs in the cohort, which is limited by the CPU and the disk.
`predict` then runs the models, which is faster on a GPU.

We measured both on UK Biobank (486,748 individuals) with the 15 published models:

| step | hardware | time |
|---|---|---|
| `match` | 8 CPUs, 23 GB of memory | 4 min 27 s |
| `predict` | one A100 | 11 min 34 s |

That cohort had only 5,810 of the models' 229,986 SNPs, so these numbers say nothing about accuracy, and `match` can take longer on a cohort with more of the SNPs.

`match` can be run in any CPU job:

```bash
dietnet match --model models/packages --plink cohort --work-dir work
```

`benchmarks/slurm/predict_gpu.sbatch` is a GPU job that runs `predict` on its result:

```bash
MODEL_DIR=/path/models/packages TEST_PREFIX=/path/cohort WORK_DIR=/path/work \
OUT=/path/predictions.tsv DIETNET_VENV=/path/.venv \
sbatch --account=ACCOUNT benchmarks/slurm/predict_gpu.sbatch
```

The file that `match` writes holds genotypes, so keep it where the cohort is allowed to be.

## Training

A training set the size of UK Biobank does not fit in the memory of most jobs.
At one byte per genotype, 486,748 individuals and 169,829 SNPs take about 83 GB.
`dietnet train` reads the packed file, which takes 2 bits per genotype, one batch at a time, and unpacks each batch on the GPU.

To train on a cluster, first copy the dataset to the local disk of the node:

```bash
dietnet create-dataset --plink cohort --labels labels.tsv --out dataset.h5   # a CPU job

# then one GPU job for each seed and fold (8 CPUs, 32 GB of memory):
cp --preserve=timestamps dataset.h5 dataset.h5.packed.* "$SLURM_TMPDIR/"
dietnet train --dataset "$SLURM_TMPDIR/dataset.h5" ... --seeds 78 --folds 0
```

Reading the genotypes from network storage is far too slow: in the measurements below, an epoch of the published loop took 2,923 s instead of 131 s.
`--preserve=timestamps` is needed because the packed file is refused when the modification time of `dataset.h5` changes.

If the packed file is larger than the memory of the job, add `--chunk-rows 2048`, the value that we used for [height and BMI](uk-biobank.md).
The individuals are then read in chunks of neighbors instead of one at a time (see the [command line reference](cli.md#dietnet-train)).

### Time and memory

We compared `dietnet train` with the published `train.py` on simulated genotypes of the size of UK Biobank (292,048 training individuals and 169,829 SNPs).
Every run used A100 GPUs with 40 GB of memory.

| loop | genotypes | GPUs | batch size | memory | before training | one epoch |
|---|---|---|---|---|---|---|
| published | in memory | 1 | 138 | 82 GB | 346 s (load) | 65–80 s |
| published | read from local disk | 1 | 138 | 4.4 GB | 217 s (copy) | 131 s |
| published | read from network storage | 1 | 138 | not measured | none | 2,923 s |
| `dietnet train` | packed, local disk | 1 | 138 | 8.3 GB | 244 s (copy) | **13.5–14 s** |
| `dietnet train` | packed, local disk | 1 | 2,048 | 8.6 GB | 244 s (copy) | 8.9–9.5 s |
| `dietnet train` | packed, local disk | 4 | 2,048 | 5.8 GB | 217 s (copy) | 7.8–8.6 s |

At the published batch size, an epoch is about five times faster than the published loop with the genotypes in memory, and takes a tenth of the memory.

Using more GPUs for one model barely helps, because the time is then spent reading the packed file.
To train several models, use one GPU for each.

### Batch size

A larger batch is faster, but it can change the accuracy.
To see by how much, we trained on the 1KGP dataset with the published split and seed 78, with the config's learning rates multiplied as shown.
These models were trained with the published loop, which `dietnet` used up to version 0.5.0.

| batch size | learning rates | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean (sd) |
|---|---|---|---|---|---|---|---|
| 138 | published | 0.9270 | 0.9332 | 0.9272 | 0.9310 | 0.9111 | **0.926** (0.009) |
| 552 | × 2 | 0.9378 | 0.9267 | 0.9358 | 0.9181 | 0.9154 | **0.927** (0.010) |
| 552 | × 4 | 0.9378 | 0.9224 | 0.9272 | 0.9095 | 0.9197 | 0.923 (0.010) |
| 2,208 | × 4 | 0.9013 | 0.8987 | 0.9143 | 0.8750 | 0.8850 | 0.895 (0.015) |
| 2,208 | × 16 | 0.8197 | 0.7823 | 0.8522 | 0.8125 | 0.7939 | 0.812 (0.027) |

A batch four times larger kept the accuracy when the learning rates were multiplied by two, the square root of four.
A batch of 2,208 lost accuracy.
However, each fold has only about 1,400 training individuals, so this is a single step per epoch.
On a biobank, a batch of 2,048 is still about 140 steps per epoch, and the result can be different.
Before training many models with a larger batch, compare it with 138 on one fold.

## Reproducing the published models

`reproduction/` has the jobs that retrain the 15 published models on a SLURM cluster, from the Zenodo release (which `dietnet published --dest DIR` downloads to `DIR/download/Dietnet_training`):

```bash
RELEASE=/path/Dietnet_training DIETNET_VENV=/path/.venv EXP_DIR=/path/run \
  ACCOUNT=my-account bash reproduction/submit_retrain.sh

dietnet check --packages /path/run/exp/packages    # once the jobs are done
```

The first job computes the SNP features and statistics, and stops if they differ from the published ones.
The second trains one model for each seed and fold, with the published configs.

The jobs read the partition from `reproduction/1kgp_partition_seed124.npz`.
`dietnet partition --seed 124` puts the same individuals in each fold, but in a different order, which changes the batches.

The published models have a mean accuracy of 0.9214 on their test folds.
When we retrained them with the published loop, it was 0.9282, and each took 19 to 58 minutes on an A100.
`reproduction/README.md` has the results of each model.
Since then, `dietnet train` uses a [different loop](parity.md#training), so the accuracy of each model will not be exactly the same.

## Benchmarks

Most of the numbers on this page come from scripts in `benchmarks/`, which take their paths from environment variables:

| script | what it measures |
|---|---|
| `slurm/bench_predict.sbatch` | every step from PLINK files to a matched cohort: time and memory |
| `slurm/bench_train_inputs.sbatch` | training, by how the genotypes are read |
| `slurm/bench_train_throughput.sbatch` | individuals per second, by how the genotypes are read and by batch size |
| `slurm/batch_size_study.sbatch` | accuracy by batch size and learning rate |
| `slurm/bench_io.sbatch` | reading genotypes from different file formats |

The comments at the top of each script say what to set.
The rows of the published loop in the tables above were measured with earlier versions of these scripts.
