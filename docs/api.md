# Python API

Every command is a thin layer over these modules, and they can be used directly,
from a notebook or another pipeline. For example, the published models on a
cohort, without the command line:

```python
from dietnet import package, predict

packages = package.find("models/packages")             # every package below, in (seed, fold) order
matched = predict.matched_dataset(packages, "mycohort", "work/")
result = predict.predict(packages, matched, device="cpu")

result.table()      # one row per sample: prediction, probability, model_agreement
result.scores       # models x samples x classes
result.alignment    # the matching counts
```

Upstream's own code is reached through `dietnet.upstream`: `run` executes one of
its scripts in a subprocess and `modules` imports its modules in this process.
The modules below call it; they never reimplement what it does.

## Prediction

::: dietnet.predict

## Model packages

::: dietnet.package

## Matching a cohort

::: dietnet.align

## The HDF5 dataset

::: dietnet.dataset

## Training

::: dietnet.training

## SNP features

::: dietnet.features

::: dietnet.alphagenome

## The published models

::: dietnet.published
