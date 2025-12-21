# DIETNETWORK

PyTorch implementation of DietNetwork (https://arxiv.org/abs/1611.09340) for genetic ancestry classification.

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/rochefca/DIETNETWORK.git
cd DIETNETWORK

# Create virtual environment and install package
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Verify installation
dietnet info
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .
```

### Optional dependencies

```bash
# For interpretability features
uv pip install -e ".[interpretability]"

# For experiment tracking with Comet.ML
uv pip install -e ".[tracking]"

# For development
uv pip install -e ".[dev]"
```

## Quick Start - Smoke Test

Validate your installation with the 1000 Genomes smoke test:

```bash
# Run smoke test (downloads model and data automatically)
bash tests/run_smoke_test.sh
```

Expected output:
```
Accuracy: 94.75%
Expected: 85-100%
✓ PASSED
```

This test:
- Downloads a pretrained 1KGP model (~57 MB)
- Downloads 1KGP test data (~1-2 GB)
- Runs inference and validates accuracy
- Caches everything for future runs

## Using Model Presets

Run inference with pretrained models:

```bash
# Using a preset model (downloads automatically)
dietnet predict --model 1kgp_default \
                --plink-prefix /path/to/your/data \
                --output predictions.tsv

# Using a local model package
dietnet predict --model ./pretrained_1000g \
                --plink-prefix /path/to/your/data \
                --output predictions.tsv
```

**Available presets:**
- `1kgp_default`: 1000 Genomes Phase 3 (24 populations, single model)
- `hgdp_ukbb`: HGDP+1KGP for UKBB inference (coming soon)

## Train on PLINK Data (recommended)

Train directly from PLINK files and produce packaged models (`seed_X/fold_Y` with `model.pt`, `metadata.json`, `snps.txt`, `input_stats.npz`, `embedding.npz`, `label_mapping.json`, `allpos.bim`):

```bash
# 1) Partition your PLINK dataset (optionally stratify by population)
dietnet partition \
    --exp-path ./data \
    --dataset train.bed \
    --nb-folds 5 \
    --stratify \
    --label-file labels.tsv   # required for stratified PLINK

# 2) Compute embeddings per fold
dietnet generate-embedding \
    --exp-path ./data \
    --dataset train.bed \
    --label-file labels.tsv \
    --output-name embedding.npz

# 3) Compute input stats (means/stds) per fold
python Dietnet/compute_input_features_mean.py \
    --exp-path ./data \
    --dataset train.bed \
    --partition partitioned_idx.npz \
    --out input_features_means.npz

# 4) Train and package a single model (default seed, all folds)
dietnet train \
    --exp-path ./data \
    --exp-name my_experiment \
    --config config.yaml \
    --plink-prefix ./data/train \
    --label-file labels.tsv \
    --folds 0   # pick one fold if you only want a single model

# 5) Train an ensemble across seeds/folds
dietnet train \
    --exp-path ./data \
    --exp-name my_experiment \
    --config config.yaml \
    --plink-prefix ./data/train \
    --label-file labels.tsv \
    --seeds 42 43 44 \
    --folds 0 1 2 3 4 \
    --output-dir ./my_packages   # optional override
```

Packages land in `<exp-path>/<exp-name>_packages/seed_*/fold_*/` by default and are ready for `dietnet predict`.

## Inference (presets or your own packages)

### Preset models (downloaded automatically)
```bash
dietnet predict --model 1kgp_default \
                --plink-prefix /path/to/test_data \
                --output predictions.tsv
```

### Your own packaged models
Point `--model` to the directory that contains `seed_*` folders (the parent of the packages):
```bash
dietnet predict --model ./my_packages \
                --plink-prefix /path/to/test_data \
                --output predictions.tsv

# Use a subset of seeds/folds from your ensemble
dietnet predict --model ./my_packages \
                --plink-prefix /path/to/test_data \
                --output predictions.tsv \
                --seeds 42 43 --folds 0 1
```

## Label file format

- Tab-delimited (TSV) with a header row and exactly two columns:
  1) `Sample` — must match the PLINK FAM `iid`
  2) `Population` (or your label name)
- Example:
  ```
  Sample	Population
  NA21144	GIH
  NA21143	GIH
  NA21090	CEU
  ```
- No commas, quotes, or extra columns. UTF-8 text. This file is required for PLINK stratified partitioning and for training/predicting with labels.

## Training from Scratch (legacy HDF5 path)

### Complete workflow

```bash
# 1. Create dataset from genotype and label files
dietnet create-dataset \
    --genotypes data/snps.txt \
    --labels data/labels.txt \
    --output-dir ./processed

# 2. Partition into cross-validation folds
dietnet partition \
    --exp-path ./processed \
    --nb-folds 5

# 3. Generate genotype frequency embeddings
dietnet generate-embedding \
    --exp-path ./processed

# 4. Train model on a specific fold
dietnet train \
    --exp-path ./processed \
    --exp-name experiment1 \
    --which-fold 0 \
    --config config.yaml

# 5. Run predictions on external data
dietnet predict \
    --test-dataset test.hdf5 \
    --train-dataset ./processed/dataset.hdf5 \
    --config ./processed/experiment1/config.yaml \
    --embedding ./processed/embedding.npz \
    --input-features-stats ./processed/input_features_means.npz \
    --model-params ./processed/experiment1/best_model.pt \
    --output-dir ./results \
    --which-fold 0
```

### Get help

```bash
# General help
dietnet --help

# Command-specific help
dietnet train --help
dietnet predict --help
```
