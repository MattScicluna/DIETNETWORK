# Quickstart

Predict the ancestry of your cohort with the 15 published models.

## 1. Fetch the published models

```bash
dietnet published --dest models/
```

This downloads `Dietnet_training.tar.gz` from Zenodo record 21133813
(concept [doi:10.5281/zenodo.16943452](https://doi.org/10.5281/zenodo.16943452)),
checks it against the checksum Zenodo publishes, unpacks it under
`models/download/` and turns it into 15 [model packages](formats.md#model-packages)
under `models/packages/seed_{78,79,80}/fold_{0..4}/`. If the release is already
unpacked somewhere, `--release path/to/Dietnet_training` skips the download.

## 2. Predict

```bash
dietnet predict --model models/packages --plink mycohort --out predictions.tsv \
                --work-dir "$SCRATCH/dietnet"
```

`mycohort.bed/.bim/.fam` can hold any SNP set. `predict`:

1. **matches** the cohort to the models' SNPs by position and allele
   ([how](formats.md#the-matched-cohort)): copied where the alleles agree,
   flipped (2 − dosage) where they are swapped, either on the other strand
   except for A/T and C/G SNPs, and missing otherwise. The matched copy is an
   HDF5 file in `--work-dir` (next to `--out` if not given), named after the
   cohort and a fingerprint of the model SNPs and the cohort's `.bim`/`.fam`,
   so a second run on the same inputs reuses it;
2. **scores** every sample with every model, using upstream's model class and
   its `indep_test_step` (imputation with the training means,
   standardisation, the D/(D−|C|) missing-SNP scale, softmax);
3. **combines** the models into one call per sample.

It reports how many of the models' 229,986 SNPs were found. The paper advises
against relying on predictions from fewer than 1,000, and `predict` warns below
that.

`predictions.tsv` has one row per sample:

| column | |
|---|---|
| `sample` | the `.fam` IID |
| `prediction` | the population with the highest mean probability across models |
| `probability` | that mean probability |
| `model_agreement` | the fraction of models whose own top population it is |

`model_agreement` can be 0: with many classes the mean probability can favour a
population no single model ranked first.

## Options worth knowing

- `--scores scores.npz` also saves every model's probabilities
  ([format](formats.md#per-model-scores)).
- `--seeds 78,79 --folds 0` uses a subset of the models. Values are
  comma-separated, or the flag is repeated (`--seeds 78 --seeds 79`).
- `--device cpu` or `cuda`; the default is CUDA when available.
- `--batch-size` (default 1024) is the number of samples read and scored at a
  time.

## On a cluster

Matching is CPU and I/O; inference wants a GPU. Split them into two jobs with
the same `--work-dir`:

```bash
dietnet match   --model models/packages --plink mycohort --work-dir "$SCRATCH/dietnet"   # CPU job
dietnet predict --model models/packages --plink mycohort --work-dir "$SCRATCH/dietnet" \
                --out predictions.tsv --device cuda                                      # GPU job
```

`predict` finds the matched copy and goes straight to inference. For UK Biobank
(486,748 samples), matching onto the published models took 4.5 minutes on 8
CPUs, and the 15 models about 12 minutes on one A100. See [scaling](scaling.md)
for the SLURM jobs.

## Checking against known labels

```bash
dietnet check --predictions predictions.tsv --labels labels.tsv --min-accuracy 0.9
```

prints accuracy per class and overall, over the samples in both files, and
exits with status 1 below `--min-accuracy`. `labels.tsv` has a header row, then
`sample id<TAB>label`.
