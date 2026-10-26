# DietNetwork

Genetic ancestry prediction from SNP genotypes with HussinLab's published Diet
Network, behind a command line, `dietnet`, that takes PLINK files in and gives
predictions out.

[Install](install.md){ .md-button .md-button--primary }
[Quickstart](quickstart.md){ .md-button }

> Rochefort-Boulanger C\*, Scicluna M\*, Poujol R, Grenier J-C, Carrier PL,
> Lemieux S, Hussin JG. *A Transparent and Generalizable Deep Learning Framework
> for Genomic Ancestry Prediction.* AJHG (2026),
> [doi:10.1016/j.ajhg.2026.06.003](https://doi.org/10.1016/j.ajhg.2026.06.003).
> Models: Zenodo [doi:10.5281/zenodo.16943452](https://doi.org/10.5281/zenodo.16943452).

The published models were trained on the 1000 Genomes Project (1KGP, WGS 30x,
GRCh38): 2,322 samples in 24 populations, 229,986 SNPs, five folds times three
seeds (78, 79, 80), so 15 models.

## What it is

A Diet Network is a classifier whose first layer is too large to learn directly
(one weight per SNP per hidden unit), so a second, *auxiliary* network predicts
those weights from per-SNP features. In the published model the features are
each SNP's genotype frequencies per population.

This package is HussinLab's code
([HussinLab/DIETNETWORK](https://github.com/HussinLab/DIETNETWORK)) with a layer
around it:

| | |
|---|---|
| **The model and training are upstream's, unmodified** | The scripts in `Dietnet_train_and_evaluate/` are byte for byte HussinLab's, pinned to one commit and checked by the test suite. `dietnet` runs them, and runs their model class for prediction. |
| **Biobank scale** | Cohorts are read straight from `.bed`, matched to a model once and memory-mapped; every model in an ensemble reads the same copy. UK Biobank (486,748 samples) runs through the 15 published models in about 13 minutes on one A100. |
| **Safer matching** | New cohorts are matched to the model's SNPs by position *and allele*. A SNP whose dosage counts the other allele is flipped, never silently mirrored. Missing SNPs are scaled the published way, by D/(D−\|C\|). |
| **Portable model packages** | Each trained (seed, fold) is a self-contained directory that records its preprocessing and the upstream commit that trained it. |
| **Other SNP features** | `dietnet build-embedding` assembles the auxiliary network's input from several sources, including AlphaGenome variant effects, without changing upstream's model. |

## Correct means identical to upstream

Where this package does something upstream's scripts also do, the definition of
correct is that the two give the same result. `tests/parity/` checks it against
a full upstream checkout at the pinned commit: the HDF5 dataset, the prediction
scores, and the published models' scores through upstream's own inference
script. A change that moves a parity result is a bug, never a tolerance to
loosen. See [parity with upstream](parity.md).

## Where to start

- [Install](install.md): from source with uv or pip, the extras, and how to
  check the upstream code is intact
- [Quickstart](quickstart.md): the published models on your cohort
- [Training](training.md): your own models, from PLINK files and labels
- [Tutorial](tutorial.ipynb): the whole pipeline on the bundled 1KGP fixture,
  in under a minute
- [SNP features](features.md): other inputs for the auxiliary network,
  including AlphaGenome
- [Formats](formats.md): every file the pipeline reads and writes
- [Parity with upstream](parity.md): how exactness is tested
- [Scaling](scaling.md): SLURM jobs for biobank cohorts and the reproduction
- [Command line](cli.md): every command and option
- [Python API](api.md): the modules behind the commands
