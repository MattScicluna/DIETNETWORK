# UK Biobank traits: height and BMI

Diet Network regression on the UK Biobank array (485,516 SNPs, GRCh38), and
whether AlphaGenome variant effects help the auxiliary network. Every step is a
SLURM job that takes its paths from the environment; no cluster path is in the
repository, and cohort data and everything derived from it stay on the cluster.

## Design

- **Samples:** genotyped, Pan-UKB ancestry `EUR`, and none of height (50), BMI
  (21001), sex (31), age at recruitment (21022) or Townsend deprivation
  (22189) missing at the first visit: 424,768 of 447,946. All label sets use
  the same samples, so they share one partition and one set of statistics.
  Pan-UKB's `related` samples (64,087) are still in; dropping them later needs
  only a new partition, not new datasets.
- **Label sets:** the trait regressed on covariates and standardised
  (`dietnet adjust-phenotype`). Covariates explain R² = 0.527 of height and
  0.009 of BMI with sex and age; 0.530 and 0.017 with Townsend as well.

  | name | trait | covariates |
  |---|---|---|
  | `height` | height | sex, age |
  | `bmi` | BMI | sex, age |
  | `height_tdi` | height | sex, age, Townsend |
  | `bmi_tdi` | BMI | sex, age, Townsend |

  Townsend is partly heritable and genetically correlated with BMI, so
  adjusting for it removes some genetic signal along with environmental
  noise: the `_tdi` sets are the variant, the others the main analysis.
- **Features** (`specs/`): `freq` (genotype frequencies per fold, the
  baseline), `ag` (AlphaGenome effects, 64 PCs), `freq_ag`, and its controls
  `freq_random` (64 random columns) and `freq_shuffled` (AlphaGenome's values
  on permuted SNPs); `mlp` in `TRAIN_SPECS` trains upstream's MLP instead.
- **Classes:** one for everyone. Partitioning needs class labels; with pooled
  frequencies nothing else reads them, so the folds are a plain random split.

## Speed

- Training: `--engine dietnet --inputs packed`, one model per A100, the 2-bit
  copy staged to the node's disk (~50 GB; the `.h5` stays on Lustre, linked,
  since training reads only labels from it).
- Upstream's statistics script loads a fold's training genotypes (~125 GB) and
  works through the SNPs serially (hours per fold). `stats.sbatch` runs the
  five folds side by side, on one-fold partitions; `merge_stats.sbatch` joins
  them into the file one run over all folds writes (the dry run checks they
  are identical).
- The four datasets, the five embeddings and the five statistics folds are
  array tasks; each job waits only on what it reads, so `submit.sh prep` can
  be queued while the cluster is down.
- Epochs: the published 8,000 epochs and patience 2,000 were ~80,000 and
  ~20,000 steps on 1KGP; here an epoch is ~1,850 steps. The pilot runs at most
  200 epochs with patience 20, at batch 138 and at 552 (learning rates ×2,
  which kept accuracy on 1KGP): its logs give the epoch time and where
  validation loss bottoms out, which set the grid's config.

## Running it

On the cluster, an env file (not in git):

```bash
export GENOTYPES=/path/to/array_plink_prefix
export FIELDS=/path/to/variables          # <field>.csv.gz per UKB field
export PAN_UK=/path/to/variables/pan-uk   # Pan-UKB table and bridge to these ids
export ALPHAGENOME=/path/to/alphagenome_ukbb.tsv.gz
export DIETNET_VENV=/path/to/repo/.venv
export EXP_DIR=/path/to/run_dir
export ACCOUNT=cpu-account GPU_ACCOUNT=gpu-account
```

```bash
source traits.env
bash prepare.sh            # labels; seconds, fine on a login node
bash submit.sh prep        # subset -> datasets -> partition -> stats, embeddings -> pilot
# after the pilot (runs/height/freq/b138 and b552):
TRAIN_LABELS="height bmi" TRAIN_SPECS="freq ag freq_ag freq_random freq_shuffled mlp" \
TRAIN_CONFIGS=b552 SEEDS=78 FOLDS="0 1 2 3 4" bash submit.sh train
$DIETNET_VENV/bin/dietnet check --packages $EXP_DIR/runs/height/freq_ag/b552/packages
```

`submit.sh train` passes further arguments to `sbatch` (e.g.
`--dependency=afterok:<id>`). Job ids are appended to `$EXP_DIR/logs/jobs.txt`.

## Dry run

```bash
bash experiments/ukbb_traits/dryrun/dryrun.sh /tmp/ukbb_traits_dryrun   # from the repo root
```

runs every step on fake data around the `kgp_mini` fixture (plink2 replaced by
a copy, configs cut to 3 epochs, training on CPU). Run it after changing any
script here.
