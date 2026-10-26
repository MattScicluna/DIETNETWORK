# Diet Network

A deep learning framework for genetic ancestry inference from SNP genotypes.
The published models were trained on the 1000 Genomes Project (1KGP, WGS 30x,
GRCh38) to classify individuals into 24 populations.

> Rochefort-Boulanger C\*, Scicluna M\*, Poujol R, Grenier J-C, Carrier PL,
> Lemieux S, Hussin JG. *A Transparent and Generalizable Deep Learning Framework
> for Genomic Ancestry Prediction.* AJHG (2026),
> [doi:10.1016/j.ajhg.2026.06.003](https://doi.org/10.1016/j.ajhg.2026.06.003).
> Models: Zenodo [doi:10.5281/zenodo.16943452](https://doi.org/10.5281/zenodo.16943452).

This repository is HussinLab's published code
([HussinLab/DIETNETWORK](https://github.com/HussinLab/DIETNETWORK)) behind a
command line, `dietnet`, that takes PLINK files in and gives predictions out:

- **The model and training are upstream's, unmodified.** The training scripts
  in `Dietnet_train_and_evaluate/` are byte for byte HussinLab's, pinned to
  one commit (`src/dietnet/upstream_manifest.json`, checked by the test suite).
  `dietnet` runs them, and runs their model class for prediction.
- **Biobank scale.** Cohorts are read straight from `.bed`. They are matched to
  a model once and memory-mapped, and every model in an ensemble reads the
  same copy.
- **Safer matching.** New cohorts are matched to the model's SNPs by position
  *and allele*. A SNP whose dosage counts the other allele is flipped, never
  silently mirrored. Missing SNPs are scaled the published way, by D/(D−|C|).
- **Portable model packages.** Each one is self-contained, and records its
  preprocessing and the upstream commit that trained it.

## Install

```bash
git clone https://github.com/MattScicluna/DIETNETWORK && cd DIETNETWORK
uv sync --frozen          # or: pip install -e .
uv run dietnet info       # versions; confirms the upstream code is unmodified
```

Tested on Python 3.10–3.12 (Linux and macOS). A GPU is optional; CPU is enough for prediction.

## Predict with the published models

```bash
dietnet published --dest models/                 # the 15 published models (Zenodo, ~250 MB)
dietnet predict --model models/packages --plink mycohort --out predictions.tsv \
                --work-dir "$SCRATCH/dietnet"    # where the matched cohort copy goes
```

`mycohort.bed/.bim/.fam` can hold any SNP set. The output reports how many of
the model's 229,986 SNPs were found; the paper suggests not relying on
predictions with fewer than 1,000. `predictions.tsv` has one row per sample:

| column | |
|---|---|
| `sample` | the .fam IID |
| `prediction` | the population with the highest mean probability across models |
| `probability` | that mean probability |
| `model_agreement` | the fraction of models whose own top population it is |

Add `--scores scores.npz` for every model's probabilities, and `--seeds`/`--folds`
to use a subset of models.

On a cluster, split the two stages: `dietnet match` (CPU and I/O: reads the
cohort and writes the matched copy to `--work-dir`) as one job, then
`dietnet predict` with the same `--work-dir` on a GPU, which finds the matched
copy and only runs the models. For UK Biobank (486,748 × 169,829), matching
takes about 20 minutes on 8 CPUs.

## Train your own

```bash
dietnet create-dataset      --plink panel --labels labels.tsv --out dataset.h5
dietnet partition           --dataset dataset.h5 --out partition.npz          # 5 folds, labels balanced
dietnet generate-embedding  --dataset dataset.h5 --partition partition.npz --out embedding.npz
dietnet compute-input-stats --dataset dataset.h5 --partition partition.npz --out stats.npz
dietnet train --dataset dataset.h5 --partition partition.npz --embedding embedding.npz \
              --input-stats stats.npz --config config.yaml --exp-dir exp \
              --seeds 78,79,80 --folds 0,1,2,3,4
dietnet check --packages exp/packages            # each model's held-out test accuracy
```

`--model mlp` trains upstream's MLP baseline instead: no auxiliary network, no
`--embedding`. It's the comparison point for any change to the SNP features.

- `labels.tsv`: a header row, then `sample id<TAB>label`. Every sample in the
  .fam needs a label.
- `config.yaml`: the published one is the default to start from:

  ```yaml
  batch_size: 138
  epochs: 8000
  input_dropout: 0.995
  dropout_main: 0.1
  lr_aux: 0.0006
  lr_main: 0.0003
  learning_rate_annealing: 0.99
  nb_hidden_u_aux: [100, 100]
  nb_hidden_u_main: [100]
  patience: 2000
  seed: 78          # overridden per --seeds
  uniform_init_limit: 0.02
  ```
- Regression: give `create-dataset` both `--labels` (class labels; the
  embedding is built from them, as upstream does) and `--regression-labels`,
  then `train --task regression`.

Each step after `create-dataset` is the upstream script of the same name; logs
are under `exp/logs/`. Packages land in `exp/packages/seed_*/fold_*/`, with the
fold's held-out predictions in `test_predictions.tsv`.

## Other SNP features for the auxiliary network

The auxiliary network turns per-SNP features into the main network's first
layer. The published model uses genotype frequencies per population.
`dietnet build-embedding` can replace or extend them, without changing
upstream's model (it reads whatever matrix `embedding.npz` holds):

```yaml
# features.yaml
blocks:
  - provider: class_genotype_freq      # the published features (upstream's script)
  - provider: precomputed              # any per-SNP table: chrom, pos, ref, alt, features...
    path: scores.tsv
    orient: effect                     # ALT-vs-REF values: flipped where A1 is REF
    pca: 64                            # keep this block to 64 columns
  - provider: random                   # control: same width, no information
    columns: 64
```

```bash
dietnet build-embedding --dataset dataset.h5 --partition partition.npz \
                        --spec features.yaml --out embedding.npz
```

Then train with `--embedding embedding.npz` as usual. Blocks are matched to
the dataset's SNPs by position and allele, on either strand. SNPs a table
lacks are zeroed and flagged in a `<name>:missing` column. Packages record the
blocks a model was trained on. `shuffled` (an earlier block with SNP rows
permuted) is the other control. With only `class_genotype_freq`, the output
is exactly the published embedding.

### AlphaGenome variant effects as features

```bash
pip install 'dietnetwork[alphagenome]'
export ALPHAGENOME_API_KEY=...            # never in a file or config
dietnet alphagenome --dataset dataset.h5 --out alphagenome.tsv   # resumable: rerun after an interruption
```

This scores every SNP through the hosted AlphaGenome API. Each row holds a
signed ALT − REF effect per track (default scorers: ATAC, DNase, TF and
histone ChIP, CAGE, RNA-seq; `--scorers` to choose). The result is a table
for the `precomputed` provider:

```yaml
  - provider: precomputed
    path: alphagenome.tsv
    orient: effect     # flipped where the dataset counts REF (a quarter of the 1KGP SNPs)
    pca: 64            # thousands of tracks -> 64 columns
```

SNP ids must be `chr:pos:REF:ALT` on GRCh38, as in the 1KGP data, because
REF and ALT are not recorded in a `.bim`. For the experiments, compare it
against the same spec with a `random` block of the same width, and with a
`shuffled` copy of the AlphaGenome block.

## How it is kept exact

- `Dietnet_train_and_evaluate/` holds only the upstream files the CLI runs (the
  four scripts and their imports), unedited. The rest of upstream (deprecated
  code, experiments, its own inference scripts) is not kept; the manifest lists
  it.
- `tests/parity/` checks, against a full upstream checkout at the pinned
  commit, that:
  - `create-dataset` writes the same HDF5 as upstream's `create_dataset.py`;
  - `predict` gives the same scores as upstream's `test_independent_dataset.py`,
    including when upstream does its own SNP matching;
  - with the published models, `predict` gives the same scores as upstream's
    `Dietnet_inference/inference.py`.
- To take newer upstream code: `git merge upstream/main`, delete any removed
  file it brings back, run `python scripts/upstream_manifest.py --commit
  upstream/main`, then the tests.

## Development

```bash
uv sync --frozen --extra dev && uv run pre-commit install
uv run pytest                                   # see tests/README.md for the parity tests
```

See `CLAUDE.md` for conventions and `docs/refactor-plan.md` for the plan.
