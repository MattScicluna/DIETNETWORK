# SNP features for the auxiliary network

Here we describe how to change the features that the auxiliary network uses to compute the weights of each SNP.
For SNP \(j\), these features are a vector \(e_j\), and the network's first layer has weights

\[
W_{jk} = f_k(e_j),
\]

where \(f\) is the auxiliary network and \(k\) is a hidden unit (see [Training](training.md#free-snp-weights)).
In the published models, \(e_j\) holds the genotype frequencies of SNP \(j\) in each population.
Any other numbers that describe a SNP can be used instead, or added to them.

## Building the features

The features are described in a YAML file as a list of blocks, and \(e_j\) is the blocks placed side by side.
For example, this file adds 64 features made from a table of scores to the published ones, and 64 random features as a control:

```yaml
# features.yaml
blocks:
  - provider: class_genotype_freq      # genotype frequencies in each population
  - provider: precomputed              # a table with one row per SNP
    path: scores.tsv
    orient: effect
    pca: 64
  - provider: random
    columns: 64
```

`build-embedding` writes the features to a file, which `train` reads in place of the one from `generate-embedding`:

```bash
dietnet build-embedding --dataset dataset.h5 --partition partition.npz \
                        --spec features.yaml --out embedding.npz
dietnet train --embedding embedding.npz ...
```

It also writes a description of every block to `embedding.npz.json`.
`train` copies this description into each model package, so a package records the features it was trained on.

## Genotype frequencies

`class_genotype_freq` gives the features of the published models: the frequencies of genotypes 0, 1 and 2 in each class, computed from the training samples of each fold.

`genotype_freq` gives the frequencies over all the training samples, without classes.
Use it for a quantitative trait such as height, which has no classes.

`genotype_freq` can also use classes, from a file of sample ids and classes in the format of `--labels`:

```yaml
  - provider: genotype_freq
    classes: labels.tsv
```

The classes need not be those of the dataset: they can be the deciles of a quantitative trait, for example, as long as each class has training samples in every fold.

## A table of your own

`precomputed` reads a table with the columns `chrom`, `pos`, `ref` and `alt`, followed by numeric features.
The table can be a TSV file, possibly gzipped, or a parquet file (which needs `pyarrow`).
Its SNPs are matched to those of the dataset by position and alleles, on either strand.
`path:` is relative to the directory in which `dietnet` runs, and `columns:` selects some of the features.

**Set `orient: effect` if the values describe the effect of the ALT allele relative to REF**, as in a GWAS.
PLINK counts the minor allele, which is REF for a quarter of the 1KGP SNPs, and the sign of these SNPs is then reversed.
The sign cannot be known for A/T and C/G SNPs, so they are treated as missing.
With `orient: none`, the default, the values are used as they are.

SNPs that are not in the table get zeros, and a column named `<name>:missing` marks them, where `name:` defaults to the name of the file.
If `pca:` is used, this column is one of those reduced.
Missing values in the table are an error: fill or remove them first.

**A table for each fold.** Features such as GWAS results must come from the training samples of each fold only, since a model would otherwise be trained on information about its own test samples.
Write one table per fold, and put `{fold}` in the path:

```yaml
  - provider: precomputed
    path: features/gwas_height_fold{fold}.tsv   # gwas_height_fold0.tsv, ...
    name: gwas
    orient: effect
```

The tables must have the same features and cover the same SNPs.

## Controls

A model with more features can improve for two reasons: the features are informative, or there are simply more of them.
Two blocks help to tell these apart:

- `random` adds `columns:` columns of random numbers, which carry no information.
- `shuffled` copies an earlier block and permutes its SNPs, so that each SNP gets the features of another. `of:` is the position of that block in the list, counting from 0.

To train on the shuffled copy without the original, add `include: false` to the original.
It is then built, so that it can be copied, but left out of the features:

```yaml
blocks:
  - provider: genotype_freq
  - provider: precomputed          # built but not used
    path: alphagenome.tsv
    orient: effect
    pca: 64
    include: false
  - provider: shuffled             # its shuffled copy
    of: 1
```

Compare also with the MLP, which uses no SNP features (`dietnet train --model mlp`).

## Reducing a block

The network divides each feature by its norm over the SNPs, so all features enter with the same scale.
A block with many features therefore counts for more than a block with few.
Each block can be reduced before it is added:

- `pca: 64` keeps its first 64 principal components.
- `standardize: true`, with `pca`, first scales each feature to unit variance, so that features with large values do not decide the components.
- `center: true` subtracts the mean of each feature over the SNPs.

If one block differs between folds and another does not, the second is repeated for every fold.
A feature that is zero for every SNP is refused, since dividing it by its norm would turn the model to NaN.

## AlphaGenome variant effects

[AlphaGenome](https://github.com/google-deepmind/alphagenome) predicts thousands of genomic tracks (such as gene expression and chromatin accessibility) from DNA sequence.
It scores a variant by how much its ALT allele changes each track relative to REF.
`dietnet alphagenome` scores every SNP of a dataset and writes a table for `precomputed`.

!!! warning "This sends your variants to Google"
    The chromosome, position, REF and ALT of every SNP are sent to Google DeepMind's hosted AlphaGenome service.
    No genotypes or sample ids are sent, but check that sharing the list of variants is allowed for your data.
    The service needs a personal key, and its terms of use are non-commercial.

```bash
pip install 'dietnetwork[alphagenome]'
export ALPHAGENOME_API_KEY=...            # never in a file or config
dietnet alphagenome --dataset dataset.h5 --out alphagenome.tsv
```

The table can then be used as a block:

```yaml
  - provider: precomputed
    path: alphagenome.tsv
    orient: effect
    pca: 64            # thousands of tracks -> 64 features
```

Each row is a SNP, and each feature is the signed score of one track, ALT minus REF.
Where a track has a score for each nearby gene, the one with the largest magnitude is kept.
A SNP without a score for a track, such as one far from any gene, gets 0.

**SNP ids must be `chr:pos:REF:ALT` on GRCh38**, with REF the reference base.
The alleles in a `.bim` say which one is counted, not which one is the reference, so REF and ALT have to come from the id.
The command stops if the ids do not have this form, but it cannot tell whether REF is correct: some pipelines write `chr:pos:A1:A2`, and the scores would then describe the reverse mutation.
Check some positions against the reference genome before a long run.

**A long run can be interrupted.** The scores are saved to `alphagenome.tsv.parts/` as they arrive, and running the same command again scores only the remaining SNPs.

**If the genotypes cannot leave their machine**, score the list of SNPs on another one, with `--bim cohort.bim` in place of `--dataset`.
Only the SNP ids are read.

The [command line reference](cli.md#dietnet-alphagenome) lists the options, such as which tracks to score.
