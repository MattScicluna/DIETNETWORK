# Quickstart

Here we describe how to predict the ancestry of the individuals in your cohort with the 15 published models.
You need the cohort's genotypes as PLINK files (`mycohort.bed`, `mycohort.bim` and `mycohort.fam`), with positions on GRCh38.

## 1. Download the models

```bash
dietnet published --dest models/
```

The models are downloaded from [Zenodo](https://doi.org/10.5281/zenodo.16943452), checked against its checksum, and saved as [model packages](formats.md#model-packages) in `models/packages`.
There is one model for each of three seeds and five folds.

## 2. Predict

```bash
dietnet predict --model models/packages --plink mycohort --out predictions.tsv
```

The models were trained on 229,986 SNPs, and your cohort does not need to contain all of them.
`predict` first finds these SNPs in the cohort by chromosome, position and alleles.
If the cohort counts the other allele of a SNP, its genotypes are flipped: without this, the model would read the two homozygous genotypes the wrong way around.
Alleles given on the other strand are recognized, except for A/T and C/G SNPs, where the strand cannot be told from the alleles.
A SNP that is absent, or whose alleles do not match, is treated as missing.
To compensate for the missing SNPs, the standardized genotypes of the SNPs that were found are multiplied by the total number of SNPs divided by the number found.

`predict` prints how many SNPs were found.
The paper advises against relying on predictions from fewer than 1,000, and `predict` warns you when this happens.
A very low number can also mean that the cohort is not on GRCh38, and if no SNP is found, `predict` stops with an error.

`predict` also saves the cohort's genotypes at the models' SNPs, in a file ending in `.matched.h5` next to `predictions.tsv`, so that a second run is faster.
Treat this file as you would the cohort itself.

## 3. Read the predictions

`predictions.tsv` has one row per individual:

| column | meaning |
|---|---|
| `sample` | the individual's id in the `.fam` (IID) |
| `prediction` | the population with the highest probability, averaged over the models |
| `probability` | that average probability |
| `model_agreement` | the fraction of models whose own most likely population is `prediction` |

The models can disagree, and `model_agreement` shows by how much.
It can even be 0, when the population with the highest average is not the first choice of any single model.

To keep the probabilities of every population from every model, add `--scores scores.npz` ([format](formats.md#per-model-scores)).
The [command line reference](cli.md#dietnet-predict) lists the other options.

## Check against known labels

If you know the population of some individuals, you can measure the accuracy of the predictions:

```bash
dietnet check --predictions predictions.tsv --labels labels.tsv
```

`labels.tsv` has a header row and then `sample id<TAB>label`, with the population names used by the models.
The accuracy is printed for each population and overall, over the individuals that are in both files.

## Large cohorts

Finding the models' SNPs is limited by the CPU and the disk, while the predictions are faster on a GPU.
On a cluster, run them as two jobs, so that the GPU does not wait:

```bash
dietnet match   --model models/packages --plink mycohort --work-dir work   # CPU job
dietnet predict --model models/packages --plink mycohort --work-dir work \
                --out predictions.tsv                                      # GPU job
```

`match` saves the genotypes of the models' SNPs in `work`, and `predict` reuses them.
For UK Biobank (486,748 individuals), `match` took 4.5 minutes on 8 CPUs and `predict` about 12 minutes on one A100.
That cohort had only 5,810 of the models' SNPs, so `match` can take longer on yours.
See [scaling](scaling.md) for the SLURM jobs.
