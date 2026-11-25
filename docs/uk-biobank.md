# Quantitative traits: height and BMI

Here we describe how we used the Diet Network to predict height and body mass index (BMI) in UK Biobank, and what we learned that can help with other traits.
The data cannot be shared, so this page gives the approach and not a recipe to rerun.
The commands are the same for any cohort (see [Training](training.md#regression)).

## What we did

We predicted each trait in 424,142 participants of European ancestry, from the 1.29 million HapMap3+ SNPs with a minor allele frequency of at least 1%.
The participants were split into five folds.
In each fold, 254,474 of them trained the model, 84,846 were used to choose the best epoch, and 84,822 to test it.

We compared three kinds of model on the same folds: the Diet Network, the Diet Network with [free SNP weights](training.md#free-snp-weights), and two standard polygenic scores, lassosum2 and LDpred2-auto.
Each prediction was calibrated by a linear fit on the validation samples, which does not change r.

| Model | Height r | Height R² | BMI r | BMI R² |
|---|---|---|---|---|
| Diet Network | 0.394 | 0.155 | 0.260 | 0.068 |
| lassosum2 | 0.517 | 0.267 | 0.296 | 0.087 |
| LDpred2-auto | 0.396 | 0.157 | 0.308 | 0.095 |
| Diet Network with free SNP weights | 0.540 | 0.292 | 0.318 | 0.101 |

The values are on the test samples, averaged over the five folds.

## What we learned

**Use free SNP weights.** Without them, the Diet Network was less accurate than the better polygenic score for each trait.
With them, it was more accurate on every fold, by 0.022 to 0.027 in R² for height and by 0.002 to 0.011 for BMI.
We used \(\lambda = 0.3\) for both traits, but the best value can differ for other data and should be chosen on the validation samples.

**Adjust the trait first.** Height depends on sex and age, which a model of autosomal SNPs cannot see.
We regressed each trait on sex and age and predicted what remains:

```bash
dietnet adjust-phenotype --phenotypes phenotypes.tsv --trait height \
                         --covariates sex,age --out height.tsv
```

**Use the trait to make the SNP features.** There are no populations here, so the published features do not apply.
We divided the participants into ten groups by trait value (deciles), and used the genotype frequencies of each SNP in each group:

```yaml
blocks:
  - provider: genotype_freq
    classes: deciles.tsv      # sample id, decile
```

The frequencies come from the training samples of each fold only.
They tell the auxiliary network whether an allele becomes more or less common as the trait increases.

**Lower the input dropout.** The published value of 0.995 keeps about 0.5% of the SNPs of each sample at each step.
This suits ancestry, where a few hundred SNPs are informative, but the effects on height and BMI are spread over many SNPs.
We kept a random half of the SNPs at each step instead (`--snp-sample 0.5`).

**Compare with a polygenic score on the same folds.** An r of 0.39 for height means little on its own.
The comparison is what showed that the Diet Network without free SNP weights was not enough.

## Limits

These results come from one biobank and one ancestry.
Each model was trained once, with one seed, and related participants were not removed, so relatives can be in both the training and the test samples.
This can inflate the accuracy of every method, and not necessarily by the same amount.
The traits were not adjusted for genetic principal components.
Most of LDpred2-auto's chains failed its convergence check on height, so its score there is likely below its best.

We chose \(\lambda\) on the test samples of the first fold, which is also in the averages above.
Choosing it on the validation samples gave the same value for these features, and the other four folds, which were not used to choose it, show the same gains.
