# UK Biobank: height and BMI

This tutorial predicts two quantitative traits, standing height and body mass index (BMI),
from 1.29 million imputed SNPs in 424,142 UK Biobank participants. It uses the penalized
Diet Network (`dietnet train --free-snp-weights`, see [Training](training.md#free-snp-weights)) and compares
it with two standard polygenic scores, lassosum2 and LDpred2-auto, on the same folds.

The steps are SLURM jobs in
[`experiments/ukbb_traits/`](https://github.com/MattScicluna/DIETNETWORK/tree/master/experiments/ukbb_traits).
They need access to UK Biobank genotypes and phenotypes through an approved application,
and they take every path from the environment, so nothing about the cohort is in the
repository.

## Design

- **Samples:** genotyped participants of European ancestry (Pan-UKB), with height, BMI,
  sex and age at recruitment recorded at the first visit.
- **Traits:** each trait regressed on sex and age and standardized
  (`dietnet adjust-phenotype`). R² is therefore the share of the variance left after sex
  and age.
- **SNPs:** the 1,290,149 HapMap3+ variants with minor allele frequency at least 1% and at
  most 10% missing calls, as hard calls from the imputed genotypes.
- **Folds:** five folds. In each, 254,474 samples train the model, 84,846 choose the best
  epoch and calibrate the scores, and 84,822 test it.
- **Features:** `dec`, each SNP's genotype frequencies in each decile of the trait, from
  the training samples of the fold (`specs/dec.yaml`). They tell the auxiliary network how
  each SNP's genotypes vary along the trait.
- **Model:** each SNP's first-layer weights are the auxiliary network's prediction plus a
  free term, `W_j = aux(features_j) + delta_j`, with a group-lasso penalty of strength
  lambda on `delta_j`. The model therefore fits per-SNP weights jointly, as penalized
  regression does, and shrinks them toward what the features predict rather than toward
  zero. Lambda 0.3 was chosen on fold 0 from 0.03 to 6.
- **Polygenic scores:** lassosum2 and LDpred2-auto (bigsnpr), from a marginal GWAS of the
  training samples of each fold, with LD from 10,000 of them. The model is chosen on the
  validation samples.
- **Evaluation:** every score is calibrated by a linear fit of the trait on the score in
  the validation samples, then compared with the test samples. Calibration does not
  change r.

## Results

Test r and R², mean over the five folds:

| Model | Height r | Height R² | BMI r | BMI R² |
|---|---|---|---|---|
| **Penalized Diet Network, `dec`** | **0.540** | **0.292** | **0.318** | **0.101** |
| lassosum2 | 0.517 | 0.267 | 0.296 | 0.087 |
| LDpred2-auto | 0.396[^ldpred2] | 0.157 | 0.308 | 0.095 |
| Diet Network, `dec` (no free weights) | 0.394 | 0.155 | 0.260 | 0.068 |

[^ldpred2]: Most of LDpred2-auto's sampler chains failed its convergence check on height,
    so its height score is likely below its best.

The penalized Diet Network beats the better polygenic score for each trait on every fold:
by 0.022 to 0.027 in R² for height and by 0.002 to 0.011 for BMI. The free weights are what
make the difference; without them the Diet Network falls well short of both scores.

Each model was trained once, with one seed, and related participants were not removed, so
relatives can fall in both the training and test samples. This may inflate the accuracy of
every method, not necessarily by the same amount.

On fold 0, none of the following improved on these results by more than about 0.01 in r:
AlphaGenome variant-effect predictions as features, 9.1 million imputed SNPs in place of
HapMap3+, functional annotations as per-SNP penalty weights (`--snp-penalty-weights`), and
gene-level embeddings of nearby genes as features.

## Running it

Paths and accounts go in an env file on the cluster, as described in the experiment's
`README.md`. The HapMap3+ cohort reuses the labels, folds and benchmark sample lists of a
run on the genotyping array, so that run comes first.

```bash
source traits.env
cd experiments/ukbb_traits

# 1. Labels, folds and benchmark sample lists, on the array
bash prepare.sh
bash submit.sh prep
bash submit.sh bench

# 2. The HapMap3+ cohort from the imputed genotypes, with the same samples and folds
export SRC_EXP="$EXP_DIR" EXP_DIR=/path/to/hm3_run
export LABELS="height=height:sex,age bmi=bmi:sex,age"
IMPUTED=/path/to/imputed_chr{chr} HM3_MAP=/path/to/map_hm3_plus.rds bash submit.sh imputed

# 3. Datasets, input statistics and the dec features, then a one-fold pilot
RECIPE="--bn-recalibrate 20000 --snp-sample 0.5 --chunk-rows 2048 --chunks-per-buffer 4"
RECIPE+=" --snp-blocks 64 --free-snp-weights 0.3"
SPECS="dec@height dec@bmi" STAGE=0 TRAIN_FLAGS="$RECIPE" bash submit.sh prep-imputed

# 4. The penalized Diet Network, five folds, both traits (one A100 per model)
TRAIN_LABELS="height bmi" TRAIN_SPECS=dec TRAIN_CONFIGS=b138_d50 SEEDS=78 \
FOLDS="0 1 2 3 4" STAGE=0 TRAIN_FLAGS="$RECIPE" RUN_SUFFIX=_fw0.3 \
    bash submit.sh train --mem=128G --time=12:00:00

# 5. Calibrated test r and R² (rescale/<label>_dec_b138_d50_fw0.3.json)
sbatch --account="$ACCOUNT" --chdir="$EXP_DIR" \
    --export=ALL,BENCH="$PWD/benchmark",RUNS=dec/b138_d50_fw0.3 benchmark/rescale.sbatch

# 6. lassosum2 and LDpred2-auto on the same folds (bench/fold<k>/<label>.ldpred2.plink.*)
GWAS=plink2 EXTRA=1 bash submit.sh bench-ldpred2 --mem=128G
```

A penalized model takes 4 to 9 hours per fold on one A100 with 128 GB of memory. The free
weights add about 129 million parameters at 1.29 million SNPs. `--snp-sample 0.5` uses a
random half of the SNPs in each batch in place of input dropout, and the `--chunk-rows`
options read the genotypes in large sequential blocks; the
[command line reference](cli.md) describes every option.
