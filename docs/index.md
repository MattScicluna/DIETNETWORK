# DietNetwork

`dietnet` is a command line program that predicts genetic ancestry from SNP genotypes with the published Diet Networks, and trains new ones.
It reads PLINK files.

[Install](install.md){ .md-button .md-button--primary }
[Quickstart](quickstart.md){ .md-button }

> Rochefort-Boulanger C\*, Scicluna M\*, Poujol R, Grenier J-C, Carrier PL,
> Lemieux S, Hussin JG. *A Transparent and Generalizable Deep Learning Framework
> for Genomic Ancestry Prediction.* AJHG (2026),
> [doi:10.1016/j.ajhg.2026.06.003](https://doi.org/10.1016/j.ajhg.2026.06.003).
> Models: Zenodo [doi:10.5281/zenodo.16943452](https://doi.org/10.5281/zenodo.16943452).

## What is a Diet Network?

A neural network that reads genotypes has one weight in its first layer for every SNP and hidden unit.
For the published models (229,986 SNPs and 100 hidden units), this is about 23 million weights, while the dataset has 2,322 individuals.
A Diet Network does not learn these weights directly.
A second network, the auxiliary network, computes the weights of each SNP from features of that SNP, so far fewer parameters are learned.
In the published models, the features of a SNP are its genotype frequencies in each population.
The [formula](training.md#free-snp-weights) is in the training guide.

The published models were trained on the 1000 Genomes Project (1KGP, WGS 30x, GRCh38) to classify individuals into 24 populations.
There are 15 of them: one for each of three seeds and five folds.

## What you can do with it

- **Predict the ancestry of a cohort** with the published models, from any set of SNPs on GRCh38. See the [Quickstart](quickstart.md).
- **Train a model** on your own genotypes and labels. See [Training](training.md), or run the [tutorial](tutorial.ipynb), which takes under a minute.
- **Predict a quantitative trait.** We describe what we learned from [height and BMI](uk-biobank.md) in UK Biobank.
- **Change the SNP features** that the auxiliary network reads, for example to AlphaGenome variant effects. See [SNP features](features.md).

## Can I trust the predictions?

A cohort rarely has the same SNPs as the one a model was trained on, or counts the same alleles.
`dietnet` finds the SNPs of a model in the cohort by position and alleles, and flips the genotypes of a SNP when the cohort counts the other allele.
It reports how many SNPs were found, and warns you when there are too few to rely on the predictions.

The network and the code that computes the predictions are those of the published code ([HussinLab/DIETNETWORK](https://github.com/HussinLab/DIETNETWORK)), unmodified.
Tests check that both give the same scores, including for the published models.
Training uses the same network and loss in a faster loop, so a model that you train differs slightly from one trained with the published code.
See [parity with the published code](parity.md).

## Reference

- [Formats](formats.md): the files that the commands read and write
- [Command line](cli.md): every command and option
- [Python API](api.md): the functions behind the commands
- [Scaling](scaling.md): time and memory on a biobank, SLURM jobs, and how to retrain the published models
