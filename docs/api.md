# Python API

Here we describe the Python functions behind the commands, for use in a notebook or another pipeline.
For example, the published models can be applied to a cohort as follows:

```python
from dietnet import package, predict

packages = package.find("models/packages")             # every package below, in (seed, fold) order
matched = predict.matched_dataset(packages, "mycohort", "work/")
result = predict.predict(packages, matched, device="cpu")

result.table()      # one row per sample: prediction, probability, model_agreement
result.scores       # models x samples x classes
result.alignment    # the matching counts
```

## Commands and functions

Most commands call one function, with the same inputs:

| command | function |
|---|---|
| `dietnet create-dataset` | [`dataset.build`][dietnet.dataset.build] |
| `dietnet pack-genotypes` | [`genotypes.pack`][dietnet.genotypes.pack] |
| `dietnet adjust-phenotype` | [`phenotype.adjust`][dietnet.phenotype.adjust] |
| `dietnet partition` | none |
| `dietnet generate-embedding` | [`features.class_frequencies`][dietnet.features.class_frequencies] |
| `dietnet build-embedding` | [`features.build`][dietnet.features.build] |
| `dietnet alphagenome` | [`alphagenome.score`][dietnet.alphagenome.score] |
| `dietnet compute-input-stats` | [`input_stats.write`][dietnet.input_stats] |
| `dietnet train` | [`training.train`][dietnet.training.train] |
| `dietnet match` | [`predict.matched_dataset`][dietnet.predict.matched_dataset] |
| `dietnet predict` | [`predict.matched_dataset`][dietnet.predict.matched_dataset], then [`predict.predict`][dietnet.predict] |
| `dietnet check` | none |
| `dietnet published` | [`published.fetch`][dietnet.published.fetch], then [`published.import_training_release`][dietnet.published.import_training_release] |
| `dietnet info` | [`upstream.verify`][dietnet.upstream.verify], for the check of the published code |

`partition` runs a script of the published code in a subprocess, which [`upstream.run`][dietnet.upstream.run] does from Python.
`check` computes its scores in the command itself, from the files that `train` and `predict` write.

## Prediction

::: dietnet.predict

## Model packages

::: dietnet.package

## Matching a cohort

::: dietnet.align

## The HDF5 dataset

::: dietnet.dataset

::: dietnet.genotypes.pack

## Phenotypes

::: dietnet.phenotype

## Input statistics

::: dietnet.input_stats

## Training

::: dietnet.training

## SNP features

::: dietnet.features

::: dietnet.alphagenome

## The published models

::: dietnet.published

## The published code

::: dietnet.upstream
