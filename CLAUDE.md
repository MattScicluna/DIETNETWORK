# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## What this is

A PyTorch DietNetwork (https://arxiv.org/abs/1611.09340) for genetic ancestry
classification, with a `dietnet` CLI and portable model packages. It is a fork
of HussinLab/DIETNETWORK taken in January 2022 and since developed separately.
**A refactor is under way: read `docs/refactor-plan.md` first.** It holds the
decisions, the phase plan, and the bug ledger (B1, B2, ...) that the tests refer to.

- **Auxiliary net**: per-SNP features (today, genotype frequencies per class,
  per fold) → tanh layers → the weights of the main net's first ("fat") layer.
- **Main net**: fat layer → ReLU → BatchNorm → dropout → hidden layers → output.

## Layout (pre-refactor; Phase 2 moves it under `src/dietnet/`)

- `Dietnet/cli.py`: click CLI, the entry point for everything.
- `Dietnet/train.py`, `helpers/mainloop_utils.py`: training loop, early stopping.
- `Dietnet/helpers/model.py`: `Feat_emb_net` (aux), `Discrim_net2` (main),
  `CombinedModel`, `Mlp`.
- `Dietnet/generate_embedding.py`, `compute_input_features_mean.py`,
  `partition_data.py`: pipeline steps.
- `Dietnet/helpers/dataset_utils.py`: PLINK loading (pyplink), HDF5 datasets,
  imputation, normalization, `InferenceDataset`.
- `Dietnet/helpers/model_package.py`, `predict_with_plink.py`,
  `helpers/snp_alignment.py`, `helpers/plink_utils.py`: packaging and inference.
- `Dietnet/pretrained_models.py`: download registry for the 1KGP model and test data.

## Pipeline

```bash
dietnet partition           --exp-path DIR --dataset DATA.bed --label-file LABELS.tsv --stratify
dietnet generate-embedding  --exp-path DIR --dataset DATA.bed --partition partitioned_idx.npz --label-file LABELS.tsv
dietnet compute-input-stats --exp-path DIR --dataset DATA.bed --partition partitioned_idx.npz
dietnet train   --exp-path DIR --exp-name NAME --config config.yaml --plink-prefix DIR/DATA \
                --label-file LABELS.tsv --seeds 78 --folds 0 --output-dir PACKAGES
dietnet predict --model PACKAGES --plink-prefix TEST --output predictions.tsv
dietnet check   --predictions predictions.tsv --labels LABELS.tsv
```

`--config` is read from `DIR/NAME/`. Missing calls are -1 and are imputed with
the training-fold mean. Dosage counts the `.bim` A1 allele.

## Development

```bash
uv sync --frozen --extra dev
uv run pre-commit install
uv run pytest                                   # parity tests skip without upstream
DIETNET_UPSTREAM_DIR=/path/to/HussinLab/DIETNETWORK uv run pytest   # at the commit in tests/support.py
```

- Tests: `tests/README.md`. Known bugs are **strict** xfails named after the
  ledger. A fix must remove its marker, and must update `docs/refactor-plan.md`
  and `CHANGELOG.md`.
- Behaviour is pinned against upstream by `tests/parity/`. A change that moves a
  parity result is a decision: record it in the plan rather than loosening a
  tolerance.
- black, isort and flake8 at line length 100. pre-commit also runs
  `scripts/check_sensitive_files.py`. Never commit genotypes, HDF5 datasets,
  `.npy` caches, `.npz` embeddings or `.pt` weights; the only exceptions are the
  fixtures listed in `.sensitive-allow`. Never commit absolute cluster paths.
- Work on a branch with a PR into `master`. Don't open PRs against HussinLab/DIETNETWORK.
