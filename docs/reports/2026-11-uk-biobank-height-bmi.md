# Height and BMI in the UK Biobank (November 2026)

This report summarizes the Diet Network experiments on height and body mass index (BMI) in
the UK Biobank, run in November 2026. It states what we found, how it compares with two
polygenic scores, and what the results do not show. The code is on the `feat/ukbb-traits`
branch, and the experiment scripts are in `experiments/ukbb_traits/`.

## Summary

- A penalized Diet Network predicts height and BMI better than the two polygenic scores we
  compared it with, on all five folds. The mean held-out R² is 0.292 for height, against
  0.267 for lassosum2, and 0.101 for BMI, against 0.095 for LDpred2-auto.
- The gain comes from free per-SNP weights. The published Diet Network, whose SNP weights
  come only from SNP features, matches a simple marginal score (height R² 0.155).
- AlphaGenome variant-effect predictions did not improve prediction in any form we tried.

## Data and models

- **Samples:** 424,142 UK Biobank participants of European ancestry (Pan-UKB `EUR`) with
  every field present.
- **Traits:** height and BMI, each regressed on sex and age and standardized. R² is
  therefore the share of the variance left after sex and age.
- **Genotypes:** 1,290,149 imputed HapMap3+ SNPs, as hard calls (genotype probability at
  least 0.9) with minor allele frequency at least 1%.
- **Folds:** five folds. In each, 254,474 samples train the model, 84,846 choose the best
  epoch and calibrate the scores, and 84,822 test it. Every method uses the same folds.
- **Features for the auxiliary network:**
  - `dec`: each SNP's genotype frequencies in each trait decile, computed on the training
    samples of the fold. This is, in effect, the SNP's marginal association with the trait.
  - `gwas`: per-fold GWAS statistics (z, effect size and signed -log10 p).
  - AlphaGenome: 3,849 predicted molecular effects per SNP (accessibility, binding,
    histone marks, CAGE and RNA-seq across cell types), reduced by principal components.
- **Penalized Diet Network:** each SNP's first-layer weights are the auxiliary network's
  output plus a free term, `W_j = aux(features_j) + delta_j`. A group-lasso penalty
  (strength lambda) shrinks each `delta_j` toward zero, so toward what the features
  predict. This fits per-SNP weights jointly, as penalized regression does, with the
  features acting as a prior, as annotations do in annotation-informed scores. It is not a
  linear score: each SNP has 100 weights that feed hidden layers. The option is
  `dietnet train --free-snp-weights LAMBDA`.
- **Polygenic scores:** lassosum2 and LDpred2-auto, from a GWAS of each fold's training
  samples, with the model chosen on validation.
- **Evaluation:** every model's predictions are calibrated by a linear fit of the trait on
  the score in the validation samples, as is usual for polygenic scores, then compared
  with the test samples. Calibration does not change r.

## Results

Mean over the five folds (test r and R²):

| Model | Height r | Height R² | BMI r | BMI R² |
|---|---|---|---|---|
| **Penalized Diet Network, `dec`, lambda 0.3** | **0.540** | **0.292** | **0.318** | **0.101** |
| Penalized Diet Network, `dec` + AlphaGenome (64 PCs) | 0.539 | 0.291 | 0.316 | 0.100 |
| Penalized Diet Network, AlphaGenome only (200 PCs) | 0.513 | 0.263 | 0.250 | 0.063 |
| lassosum2 | 0.517 | 0.267 | 0.296 | 0.087 |
| LDpred2-auto | 0.396 | 0.157 | 0.308 | 0.095 |
| Diet Network, `dec` | 0.394 | 0.155 | 0.260 | 0.068 |
| Diet Network, `dec` + AlphaGenome (64 PCs) | 0.394 | 0.155 | 0.257 | 0.066 |
| Diet Network, `gwas` | 0.392 | 0.154 | 0.230 | 0.053 |
| Diet Network, AlphaGenome only (200 PCs)* | 0.045 | 0.002 | 0.009 | 0.000 |

\* Not calibrated on validation. Calibration changed no other Diet Network by more than
0.013 in R².

The penalized Diet Network beats the better polygenic score for each trait on every fold.
For height, the R² gain over lassosum2 is 0.022 to 0.027 by fold (mean 0.025). For BMI,
the gain over LDpred2-auto is 0.002 to 0.011 (mean 0.006). The height gain is consistent,
while the BMI gain is small and varies between folds.

**Lambda.** On fold 0, lambda 0.3 was best for both traits, whether the features were
`dec` or AlphaGenome alone:

| lambda | 0.03 | 0.1 | 0.3 | 1 | 3 | 6 |
|---|---|---|---|---|---|---|
| Height r, `dec` | 0.442 | 0.490 | **0.543** | 0.445 | 0.411 | 0.398 |
| BMI r, `dec` | 0.183 | 0.237 | **0.320** | 0.289 | 0.279 | 0.271 |
| Height r, AlphaGenome | | 0.490 | **0.514** | 0.320 | 0.149 | |
| BMI r, AlphaGenome | | 0.225 | **0.252** | 0.115 | 0.053 | |

A smaller lambda overfits: the best epoch comes early. A larger lambda pulls the weights
toward the features' prediction and loses accuracy. The loss is faster with AlphaGenome,
whose prediction carries little information about either trait.

**AlphaGenome.** AlphaGenome added nothing to `dec`, and on its own it predicted almost
nothing (height r 0.045, BMI r 0.009). In the penalized model, AlphaGenome alone reached
height r 0.513, close to lassosum2, but almost all of that comes from the free weights. For
the 200-component version, the tracks were standardized first: without standardizing,
RNA-seq accounts for only 0.65% of the variance, so its tracks barely shape the components.
200 standardized components keep 94.9% of the variance.

A likely reason for the null result is how the model uses the features. The auxiliary
network predicts each SNP's weights, so the features must predict the direction and size of
an effect. `dec` measures that directly. AlphaGenome predicts molecular effects in cells,
which say how likely a variant is to matter but not whether it raises or lowers height.
Annotation-informed scores such as SBayesRC use annotations for exactly that: they set how
large an effect is likely to be, and leave the direction to the data. A version of the
penalized Diet Network in which AlphaGenome sets each SNP's penalty, rather than its prior
weights, is untested.

## Limitations

- **Relatedness:** related participants (64,087 flagged by Pan-UKB) were not removed, so
  relatives can fall in both the training and the test samples. This can inflate held-out
  accuracy for every method, and not necessarily by the same amount.
- **Covariates:** the traits were adjusted for sex and age only, not for genetic principal
  components. Restricting to European ancestry limits population structure but does not
  remove it.
- **Baselines:** only two polygenic scores were compared, both with an LD reference of
  10,000 training samples. On height, most of LDpred2-auto's sampler chains failed its
  convergence check (4 of 30 kept on fold 0), and it scored well below lassosum2, so it is
  likely not at its best there. Stronger methods (PRS-CS,
  SBayesRC, or a penalized regression on the genotypes such as snpnet) were not run.
- **Statistics:** each model was trained once (one seed). The folds share training samples,
  so they are not independent, and the gains carry no confidence interval or p-value.
- **Model selection:** lambda was chosen on fold 0's test set. Choosing on validation agreed
  for `dec`. Folds 1 to 4 played no part in the choice and show the same gains.
- **Scope:** one biobank and one ancestry, with no external test cohort. The results may not
  transfer to other ancestries or to smaller training sets.
- **Compute:** a penalized model takes 4 to 9 hours per fold on one A100 GPU with 128 GB of
  memory. The free weights add about 129 million parameters at 1.29M SNPs.
- **AlphaGenome:** only two forms were tried (64 unstandardized components alongside `dec`,
  and 200 standardized components alone), both as the auxiliary network's input. We scored
  the HapMap3+ SNPs themselves, which mostly tag causal variants rather than being causal.

## Reproducing

The SLURM jobs and their order are in `experiments/ukbb_traits/README.md`. The training
settings at 1.29M SNPs were:

```bash
dietnet train ... --engine dietnet --bn-recalibrate 20000 --snp-sample 0.5 \
    --chunk-rows 2048 --chunks-per-buffer 4 --snp-blocks 64 --free-snp-weights 0.3
```

with the `b138_d50` configuration (batch size 138, input dropout 0.5) and seed 78.
`experiments/ukbb_traits/benchmark/rescale.sbatch` evaluates trained packages as described
above. The UK Biobank data and every result derived from individual-level data stay on the
cluster.
