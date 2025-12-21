"""
Command-line interface for DietNetwork.

This module provides a unified CLI for training and inference with DietNetwork.
"""
import os
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.2.0")
def main():
    """
    DietNetwork: Deep learning for genetic ancestry classification.

    A PyTorch implementation of DietNetwork (https://arxiv.org/abs/1611.09340)
    for population genetics and ancestry inference.
    """
    pass


@main.command()
@click.option(
    '--exp-path',
    type=click.Path(exists=True),
    required=True,
    help='Path to directory containing dataset, partitions, and embeddings.'
)
@click.option(
    '--exp-name',
    type=str,
    required=True,
    help='Name of experiment directory where results will be saved.'
)
@click.option(
    '--config',
    type=str,
    default='config.yaml',
    help='YAML config file with hyperparameters (default: config.yaml).'
)
# NEW PLINK PACKAGE APPROACH (Recommended)
@click.option(
    '--plink-prefix',
    type=str,
    default=None,
    help='PLINK file prefix for training data (without .bed/.bim/.fam).'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default=None,
    help='Directory to save trained model package(s). Defaults to <exp-path>/<exp-name>_packages.'
)
@click.option(
    '--seeds',
    type=int,
    nargs=-1,
    default=None,
    help='Seeds to train (space-separated list; default: seed from config).'
)
@click.option(
    '--folds',
    type=int,
    nargs=-1,
    default=None,
    help='Folds to train (space-separated list; default: all folds from partition).'
)
@click.option(
    '--task',
    type=click.Choice(['classification', 'regression']),
    default='classification',
    help='Task type (default: classification).'
)
@click.option(
    '--normalize/--no-normalize',
    default=True,
    help='Apply normalization to input features (default: normalize).'
)
@click.option(
    '--param-init',
    type=click.Path(exists=True),
    default=None,
    help='Path to pre-trained parameter initialization (.npz from Theano).'
)
@click.option(
    '--comet-ml/--no-comet-ml',
    default=False,
    help='Enable Comet.ML experiment tracking (default: disabled).'
)
@click.option(
    '--comet-ml-project-name',
    type=str,
    default=None,
    help='Comet.ML project name.'
)
@click.option(
    '--optimization/--no-optimization',
    default=False,
    help='Run in hyperparameter optimization mode (default: disabled).'
)
@click.option(
    '--dataset',
    type=str,
    default='dataset.hdf5',
    help='[DEPRECATED] HDF5 dataset file (default: dataset.hdf5).'
)
@click.option(
    '--partition',
    type=str,
    default='partitioned_idx.npz',
    help='Partition indexes file (default: partitioned_idx.npz).'
)
@click.option(
    '--embedding',
    type=str,
    default='embedding.npz',
    help='Genotype frequency embedding file (default: embedding.npz).'
)
@click.option(
    '--input-features-means',
    type=str,
    default='input_features_means.npz',
    help='Input feature means for missing value imputation (default: input_features_means.npz).'
)
@click.option(
    '--label-file',
    type=str,
    default=None,
    help='Label file (TSV format, required for PLINK datasets).'
)
@click.option(
    '--which-fold',
    type=int,
    default=None,
    help='[DEPRECATED] Which fold to train on (0-indexed).'
)
def train(exp_path, exp_name, config, plink_prefix, output_dir, seeds, folds,
          task, normalize, param_init, comet_ml, comet_ml_project_name,
          optimization, dataset, partition, embedding, input_features_means,
          label_file, which_fold):
    """
    Train DietNetwork models and save them as model packages.

    Recommended: use --plink-prefix to train directly from PLINK files and
    produce cache-style packages (seed_X/fold_Y). The legacy HDF5 path is still
    available but deprecated.
    """
    import warnings

    # Detect which training path to use
    dataset_is_plink = dataset is not None and dataset.endswith(('.bed', '.bim', '.fam'))
    using_plink = plink_prefix is not None or dataset_is_plink

    if plink_prefix and dataset_is_plink:
        click.echo("Note: --plink-prefix provided; ignoring PLINK-style --dataset.", err=True)

    if using_plink:
        _train_with_plink_packages(
            exp_path=exp_path,
            exp_name=exp_name,
            config=config,
            plink_prefix=plink_prefix or dataset,
            output_dir=output_dir,
            seeds=seeds,
            folds=folds,
            task=task,
            normalize=normalize,
            param_init=param_init,
            comet_ml=comet_ml,
            comet_ml_project_name=comet_ml_project_name,
            optimization=optimization,
            partition=partition,
            embedding=embedding,
            input_features_means=input_features_means,
            label_file=label_file
        )
        return

    # ====================
    # LEGACY HDF5 APPROACH (DEPRECATED)
    # ====================
    warnings.warn(
        "\n"
        "═══════════════════════════════════════════════════════════════\n"
        "DEPRECATION WARNING: HDF5-based training is deprecated!\n"
        "═══════════════════════════════════════════════════════════════\n"
        "\n"
        "Please migrate to the PLINK-based training path (--plink-prefix)\n"
        "which automatically packages models for inference.\n"
        "\n"
        "═══════════════════════════════════════════════════════════════\n",
        DeprecationWarning,
        stacklevel=2
    )

    if which_fold is None:
        click.echo("ERROR: --which-fold is required for legacy HDF5 training.", err=True)
        sys.exit(1)

    from Dietnet import train as train_module

    # Build arguments object to pass to original train script
    class Args:
        pass

    args = Args()
    args.exp_path = exp_path
    args.exp_name = exp_name
    args.config = config
    args.dataset = dataset
    args.partition = partition
    args.embedding = embedding
    args.input_features_means = input_features_means
    args.label_file = label_file
    args.which_fold = which_fold
    args.task = task
    args.normalize = normalize
    args.param_init = param_init
    args.comet_ml = comet_ml
    args.comet_ml_project_name = comet_ml_project_name
    args.optimization = optimization

    # Call the original train logic
    train_module.main_with_args(args)


def _train_with_plink_packages(exp_path, exp_name, config, plink_prefix,
                               output_dir, seeds, folds, task, normalize,
                               param_init, comet_ml, comet_ml_project_name,
                               optimization, partition, embedding,
                               input_features_means, label_file):
    """
    Train one or more models from PLINK data and package them for inference.
    """
    import copy
    import numpy as np
    import torch
    import yaml

    from Dietnet import train as train_module
    from Dietnet.helpers import dataset_utils as du
    from Dietnet.helpers.model_package import ModelPackage
    from Dietnet.helpers.snp_alignment import parse_bim_file

    exp_path = Path(exp_path)
    package_root = Path(output_dir) if output_dir else exp_path / f"{exp_name}_packages"

    if label_file is None:
        click.echo("ERROR: --label-file is required for PLINK training.", err=True)
        sys.exit(1)

    # Resolve PLINK paths
    plink_prefix_path = _normalize_plink_prefix(plink_prefix)
    dataset_arg = str(plink_prefix_path.with_suffix('.bed'))
    dataset_file = _resolve_path(exp_path, dataset_arg)
    bim_file = dataset_file.with_suffix('.bim')

    if not dataset_file.exists():
        click.echo(f"ERROR: Training PLINK file not found: {dataset_file}", err=True)
        sys.exit(1)

    if not bim_file.exists():
        click.echo(f"ERROR: BIM file not found for training data: {bim_file}", err=True)
        sys.exit(1)

    # Load config for default seed
    config_path = exp_path / exp_name / config
    if not config_path.exists():
        click.echo(f"ERROR: Config file not found: {config_path}", err=True)
        sys.exit(1)

    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f) or {}

    default_seed = base_config.get('seed')
    seed_list = list(seeds) if seeds else ([default_seed] if default_seed is not None else [])
    if not seed_list:
        click.echo("ERROR: No seed provided and config file has no 'seed' value.", err=True)
        sys.exit(1)

    # Load folds to determine available folds
    partition_path = _resolve_path(exp_path, partition)
    if not partition_path.exists():
        click.echo(f"ERROR: Partition file not found: {partition_path}", err=True)
        sys.exit(1)

    folds_data = np.load(partition_path, allow_pickle=True)
    folds_indexes = folds_data['folds_indexes']
    available_folds = list(range(len(folds_indexes)))
    fold_list = list(folds) if folds else available_folds

    invalid_folds = [f for f in fold_list if f not in available_folds]
    if invalid_folds:
        click.echo(f"ERROR: Requested folds {invalid_folds} not in available folds {available_folds}.", err=True)
        sys.exit(1)

    # Label mapping (classification only)
    label_path = _resolve_path(exp_path, label_file)
    if not label_path.exists():
        click.echo(f"ERROR: Label file not found: {label_path}", err=True)
        sys.exit(1)

    label_mapping = _load_label_mapping(label_path, task, du)

    # SNP list from training BIM
    snp_df = parse_bim_file(bim_file)
    snps = snp_df['chr_pos'].tolist()

    # Load embedding and input stats once
    embedding_path = _resolve_path(exp_path, embedding)
    stats_path = _resolve_path(exp_path, input_features_means)

    if not embedding_path.exists():
        click.echo(f"ERROR: Embedding file not found: {embedding_path}", err=True)
        sys.exit(1)
    if not stats_path.exists():
        click.echo(f"ERROR: Input feature stats not found: {stats_path}", err=True)
        sys.exit(1)

    embedding_data = np.load(embedding_path, allow_pickle=True)
    stats_data = np.load(stats_path, allow_pickle=True)

    # Train models
    for seed_value in seed_list:
        for fold in fold_list:
            click.echo(f"\n=== Training seed {seed_value}, fold {fold} ===", err=True)

            class Args:
                pass

            args = Args()
            args.exp_path = str(exp_path)
            args.exp_name = exp_name
            args.config = config
            args.dataset = dataset_arg
            args.partition = partition
            args.embedding = embedding
            args.input_features_means = input_features_means
            args.label_file = label_file
            args.which_fold = fold
            args.task = task
            args.normalize = normalize
            args.param_init = param_init
            args.comet_ml = comet_ml
            args.comet_ml_project_name = comet_ml_project_name
            args.optimization = optimization
            args.seed_override = seed_value

            # Run training for this seed/fold
            train_module.main_with_args(args)

            out_dir = exp_path / exp_name / f"{exp_name}_fold{fold}"
            checkpoint = _find_model_checkpoint(out_dir)
            if checkpoint is None:
                click.echo(f"ERROR: No checkpoint found in {out_dir}; skipping packaging.", err=True)
                continue

            model_state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            embedding_for_fold = _select_embedding_for_fold(embedding_data, fold)
            input_stats = _extract_input_stats_for_fold(stats_data, fold)

            config_for_meta = copy.deepcopy(base_config)
            config_for_meta['seed'] = seed_value

            package_dir = package_root / f"seed_{seed_value}" / f"fold_{fold}"
            ModelPackage(package_dir).save(
                model_state=model_state,
                snps=snps,
                input_stats=input_stats,
                embedding=embedding_for_fold,
                label_mapping=label_mapping,
                config=config_for_meta,
                seed=seed_value,
                fold=fold,
                training_info={
                    'exp_path': str(exp_path),
                    'exp_name': exp_name,
                    'dataset': str(dataset_file)
                },
                bim_file=bim_file
            )

            click.echo(f"✓ Saved model package to {package_dir}", err=True)


def _normalize_plink_prefix(plink_prefix: str) -> Path:
    path = Path(plink_prefix)
    if path.suffix in {'.bed', '.bim', '.fam'}:
        path = path.with_suffix('')
    return path


def _resolve_path(base: Path, target: str) -> Path:
    target_path = Path(target)
    if target_path.is_absolute():
        return target_path
    return base / target_path


def _find_model_checkpoint(out_dir: Path):
    if not out_dir.exists():
        return None
    candidates = list(out_dir.glob('*.pt'))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _select_embedding_for_fold(embedding_data, fold: int):
    import numpy as np

    if 'emb' in embedding_data:
        emb_arr = embedding_data['emb']
    elif 'embedding' in embedding_data:
        emb_arr = embedding_data['embedding']
    else:
        emb_arr = embedding_data[list(embedding_data.files)[0]]

    emb_arr = np.array(emb_arr)
    if emb_arr.ndim > 1 and emb_arr.shape[0] > fold:
        return emb_arr[fold]
    return emb_arr


def _extract_input_stats_for_fold(stats_data, fold: int):
    import numpy as np

    if 'means_by_fold' in stats_data:
        fold_stats = stats_data['means_by_fold'][fold]
        if isinstance(fold_stats, np.ndarray) and fold_stats.dtype == object:
            fold_stats = fold_stats.tolist()
        mean = np.array(fold_stats[0])
        std = np.array(fold_stats[1]) if len(fold_stats) > 1 else None
    else:
        mean = stats_data['mean']
        std = stats_data['std'] if 'std' in stats_data else None
        mean = np.array(mean)
        if mean.ndim > 1 and mean.shape[0] > fold:
            mean = mean[fold]
        if std is not None:
            std = np.array(std)
            if std.ndim > 1 and std.shape[0] > fold:
                std = std[fold]

    input_stats = {'mean': np.asarray(mean)}
    if std is not None:
        input_stats['std'] = np.asarray(std)
    return input_stats


def _load_label_mapping(label_file: Path, task: str, du) -> dict:
    import numpy as np

    if task == 'regression':
        return {}

    samples, labels = du.load_labels(label_file)
    label_names = np.unique(labels)
    return {label: idx for idx, label in enumerate(label_names)}


@main.command()
# NEW MODEL PACKAGE APPROACH (Recommended)
@click.option(
    '--model',
    type=str,
    default=None,
    help='Model preset (e.g., 1kgp_default) or path to model package directory.'
)
@click.option(
    '--plink-prefix',
    type=str,
    default=None,
    help='PLINK file prefix for test data (without .bed/.bim/.fam).'
)
@click.option(
    '--output',
    type=str,
    default=None,
    help='Output file for predictions (text).'
)
# LEGACY HDF5 APPROACH (Deprecated)
@click.option(
    '--test-dataset',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] HDF5 file containing test data.'
)
@click.option(
    '--train-dataset',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] HDF5 file containing training data (for SNP alignment).'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] YAML config file used for training.'
)
@click.option(
    '--embedding',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] Genotype frequency embedding file from training.'
)
@click.option(
    '--input-features-stats',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] Input feature statistics from training (for normalization).'
)
@click.option(
    '--model-params',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] Trained model checkpoint (.pt file).'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default=None,
    help='[DEPRECATED] Directory where predictions will be saved.'
)
@click.option(
    '--output-name',
    type=str,
    default='predictions',
    help='[DEPRECATED] Base name for output files (default: predictions).'
)
@click.option(
    '--which-fold',
    type=int,
    default=None,
    help='[DEPRECATED] Which fold was used for training (0-indexed).'
)
# COMMON OPTIONS
@click.option(
    '--seeds',
    type=int,
    nargs=-1,
    default=None,
    help='Seeds to use (space-separated list; default: all in model package).'
)
@click.option(
    '--folds',
    type=int,
    nargs=-1,
    default=None,
    help='Folds to use (space-separated list; default: all in model package).'
)
@click.option(
    '--batch-size',
    type=int,
    default=128,
    help='Batch size for inference (default: 128).'
)
@click.option(
    '--device',
    type=str,
    default=None,
    help='Device: cpu or cuda (default: auto-detect).'
)
@click.option(
    '--num-workers',
    type=int,
    default=0,
    help='DataLoader workers (default: 0).'
)
@click.option(
    '--save-logits',
    type=str,
    default=None,
    help='Optional path to save raw logits/probabilities (.npz).'
)
@click.option(
    '--save-hidden',
    type=str,
    default=None,
    help='Optional path to save final hidden representations (.npz).'
)
@click.option(
    '--force-download',
    is_flag=True,
    help='Force re-download of model preset (only for --model presets).'
)
@click.option(
    '--skip-preprocess',
    is_flag=True,
    help='Skip PLINK preprocessing (use if plink-prefix already preprocessed with preprocess-plink).'
)
@click.option(
    '--task',
    type=click.Choice(['classification', 'regression']),
    default='classification',
    help='Task type (default: classification).'
)
@click.option(
    '--normalize/--no-normalize',
    default=True,
    help='Apply normalization (default: True).'
)
def predict(model, plink_prefix, output, test_dataset, train_dataset, config,
            embedding, input_features_stats, model_params, output_dir, output_name,
            which_fold, seeds, folds, batch_size, device, num_workers, force_download,
            skip_preprocess, task, normalize, save_logits, save_hidden):
    """
    Run inference on a test dataset using a trained model.

    NEW APPROACH (Recommended):
        Use model packages with PLINK data:

        \b
        # Using a preset model
        dietnet predict --model 1kgp_default \\
                        --plink-prefix /path/to/test_data \\
                        --output predictions.tsv

        \b
        # Using a local model package
        dietnet predict --model ./pretrained_1000g \\
                        --plink-prefix /path/to/test_data \\
                        --output predictions.tsv

    LEGACY APPROACH (Deprecated):
        The HDF5-based approach is deprecated. Please use model packages instead.
    """
    import warnings
    import sys
    import os
    import subprocess
    from pathlib import Path

    # Detect which approach is being used
    using_model_package = model is not None
    using_legacy_hdf5 = model_params is not None

    if using_model_package and using_legacy_hdf5:
        click.echo("ERROR: Cannot use both --model and --model-params. Choose one approach.", err=True)
        sys.exit(1)

    if not using_model_package and not using_legacy_hdf5:
        click.echo("ERROR: Must provide either --model (new approach) or --model-params (legacy).", err=True)
        click.echo("See 'dietnet predict --help' for usage examples.", err=True)
        sys.exit(1)

    # ====================
    # NEW MODEL PACKAGE APPROACH
    # ====================
    if using_model_package:
        # Validate required arguments
        if not plink_prefix:
            click.echo("ERROR: --plink-prefix is required when using --model", err=True)
            sys.exit(1)
        if not output:
            click.echo("ERROR: --output is required when using --model", err=True)
            sys.exit(1)

        from Dietnet.model_manager import get_model_path
        from Dietnet.pretrained_models import PRETRAINED_MODELS

        # Resolve model path
        if model in PRETRAINED_MODELS:
            # Model preset - download if needed
            click.echo(f"Using model preset: {model}")
            model_dir = get_model_path(model, force_download=force_download)
        else:
            # Local path
            model_dir = Path(model)
            if not model_dir.exists():
                click.echo(f"ERROR: Model directory not found: {model_dir}", err=True)
                sys.exit(1)
            click.echo(f"Using local model: {model_dir}")

        # Determine device
        if device is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Build command for predict_with_plink.py
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'predict_with_plink.py'),
            '--model-dir', str(model_dir),
            '--plink-prefix', plink_prefix,
            '--output', output,
            '--batch-size', str(batch_size),
            '--device', device,
            '--num-workers', str(num_workers)
        ]

        if seeds:
            cmd.extend(['--seeds'] + [str(s) for s in seeds])
        if folds:
            cmd.extend(['--folds'] + [str(f) for f in folds])
        if skip_preprocess:
            cmd.append('--skip-preprocess')
        if save_logits:
            cmd.extend(['--save-logits', save_logits])
        if save_hidden:
            cmd.extend(['--save-hidden', save_hidden])

        # Run inference
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    # ====================
    # LEGACY HDF5 APPROACH (DEPRECATED)
    # ====================
    else:
        # Show deprecation warning
        warnings.warn(
            "\n"
            "═══════════════════════════════════════════════════════════════\n"
            "DEPRECATION WARNING: HDF5-based inference is deprecated!\n"
            "═══════════════════════════════════════════════════════════════\n"
            "\n"
            "The HDF5-based inference approach (--model-params, --test-dataset)\n"
            "is deprecated and will be removed in a future version.\n"
            "\n"
            "Please use the new model package approach instead:\n"
            "\n"
            "  dietnet predict --model 1kgp_default \\\n"
            "                  --plink-prefix /path/to/data \\\n"
            "                  --output predictions.tsv\n"
            "\n"
            "Benefits:\n"
            "  • Simpler: One --model flag instead of 6 separate files\n"
            "  • Faster: Works directly with PLINK files (no HDF5 conversion)\n"
            "  • Scalable: Handles large datasets with memory mapping\n"
            "  • Portable: Download pretrained models automatically\n"
            "\n"
            "═══════════════════════════════════════════════════════════════\n",
            DeprecationWarning,
            stacklevel=2
        )

        # Validate required legacy arguments
        required_legacy = {
            'test-dataset': test_dataset,
            'train-dataset': train_dataset,
            'config': config,
            'embedding': embedding,
            'input-features-stats': input_features_stats,
            'model-params': model_params,
            'output-dir': output_dir,
            'which-fold': which_fold
        }

        missing = [name for name, value in required_legacy.items() if value is None]
        if missing:
            click.echo(f"ERROR: Missing required legacy arguments: {', '.join('--' + m for m in missing)}", err=True)
            sys.exit(1)

        click.echo(f"Loading model from {model_params}...")
        click.echo(f"Processing test data from {test_dataset}...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Build arguments for test script
        class Args:
            pass

        args = Args()
        args.test_dataset = test_dataset
        args.train_dataset = train_dataset
        args.config = config
        args.embedding = embedding
        args.input_features_stats = input_features_stats
        args.model_params = model_params
        args.test_path = output_dir
        args.test_name = output_name
        args.which_fold = which_fold
        args.task = task
        args.normalize = normalize

        # Import and run test logic
        from Dietnet import test_external_dataset as test_module
        test_module.test_with_args(args)

        click.echo(f"✓ Predictions saved to {output_dir}/{output_name}_results.npz")


@main.command()
@click.option(
    '--genotypes',
    type=click.Path(exists=True),
    required=True,
    help='Tab-separated genotype file or PLINK prefix.'
)
@click.option(
    '--labels',
    type=click.Path(exists=True),
    required=True,
    help='Tab-separated label file (sample_id<tab>label).'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    required=True,
    help='Output directory for dataset.'
)
@click.option(
    '--output-name',
    type=str,
    default='dataset.hdf5',
    help='Output filename (default: dataset.hdf5).'
)
@click.option(
    '--task',
    type=click.Choice(['classification', 'regression']),
    default='classification',
    help='Task type (default: classification).'
)
@click.option(
    '--class-labels',
    type=click.Path(exists=True),
    default=None,
    help='Class labels file (required for regression task).'
)
@click.option(
    '--parallel-loading/--no-parallel-loading',
    default=False,
    help='Use parallel processing for loading (default: disabled).'
)
@click.option(
    '--ncpus',
    type=int,
    default=4,
    help='Number of CPUs for parallel loading (default: 4).'
)
def create_dataset(genotypes, labels, output_dir, output_name, task,
                   class_labels, parallel_loading, ncpus):
    """
    Create HDF5 dataset from genotype and label files.

    Parses genotype data (text or PLINK format) and creates an HDF5 dataset
    suitable for training.

    Example:
        dietnet create-dataset --genotypes data/snps.txt --labels data/labels.txt \\
                               --output-dir ./processed
    """
    os.makedirs(output_dir, exist_ok=True)

    class Args:
        pass

    args = Args()
    args.genotypes = genotypes
    args.labels = labels
    args.exp_path = output_dir
    args.out = output_name
    args.task = task
    args.class_labels = class_labels
    args.parallel_loading = parallel_loading
    args.ncpus = ncpus

    from Dietnet import create_dataset as create_module
    create_module.create_dataset_with_args(args)

    click.echo(f"✓ Dataset created: {output_dir}/{output_name}")


@main.command()
@click.option(
    '--exp-path',
    type=click.Path(exists=True),
    required=True,
    help='Path to directory containing dataset.'
)
@click.option(
    '--dataset',
    type=str,
    default='dataset.hdf5',
    help='Dataset filename (default: dataset.hdf5).'
)
@click.option(
    '--label-file',
    type=str,
    default=None,
    help='Label TSV (required for stratified PLINK partitioning).'
)
@click.option(
    '--output-name',
    type=str,
    default='partitioned_idx.npz',
    help='Output filename (default: partitioned_idx.npz).'
)
@click.option(
    '--nb-folds',
    type=int,
    default=5,
    help='Number of cross-validation folds (default: 5).'
)
@click.option(
    '--train-valid-ratio',
    type=float,
    default=0.8,
    help='Train/validation split ratio (default: 0.8).'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducibility (default: 42).'
)
@click.option(
    '--stratify/--no-stratify',
    default=False,
    help='Stratify folds by label/population (default: no).'
)
def partition(exp_path, dataset, output_name, nb_folds, train_valid_ratio, seed, label_file, stratify):
    """
    Partition dataset into cross-validation folds.

    Creates train/valid/test splits for k-fold cross-validation.

    Example:
        dietnet partition --exp-path ./processed --nb-folds 5
    """
    class Args:
        pass

    args = Args()
    args.exp_path = exp_path
    args.dataset = dataset
    args.out = output_name
    args.nb_folds = nb_folds
    args.train_valid_ratio = train_valid_ratio
    args.seed = seed
    args.label_file = label_file
    args.stratify = stratify

    from Dietnet import partition_data as partition_module
    partition_module.partition_data_with_args(args)

    click.echo(f"✓ Partitions created: {exp_path}/{output_name}")


@main.command()
@click.option(
    '--exp-path',
    type=click.Path(exists=True),
    required=True,
    help='Path to directory containing dataset and partitions.'
)
@click.option(
    '--dataset',
    type=str,
    default='dataset.hdf5',
    help='Dataset filename (default: dataset.hdf5).'
)
@click.option(
    '--partition',
    type=str,
    default='partitioned_idx.npz',
    help='Partition file (default: partitioned_idx.npz).'
)
@click.option(
    '--output-name',
    type=str,
    default='embedding.npz',
    help='Output filename (default: embedding.npz).'
)
@click.option(
    '--task',
    type=click.Choice(['classification', 'regression']),
    default='classification',
    help='Task type (default: classification).'
)
@click.option(
    '--label-file',
    type=str,
    default=None,
    help='Label file (TSV format, required for PLINK datasets).'
)
def generate_embedding(exp_path, dataset, partition, output_name, task, label_file):
    """
    Generate genotype frequency embeddings for each fold.

    Computes embeddings from training data, excluding missing values.

    Example:
        dietnet generate-embedding --exp-path ./processed
    """
    class Args:
        pass

    args = Args()
    args.exp_path = exp_path
    args.dataset = dataset
    args.partition = partition
    args.out = output_name
    args.task = task
    args.label_file = label_file
    args.include_valid = False
    args.only_valid = False
    args.emb_class_label = 'labels'

    from Dietnet import generate_embedding as embedding_module
    embedding_module.generate_embedding_with_args(args)

    click.echo(f"✓ Embeddings generated: {exp_path}/{output_name}")


@main.command()
@click.option(
    '--model',
    type=str,
    required=True,
    help='Model preset (e.g., 1kgp_default) or path to model package directory.'
)
@click.option(
    '--plink-prefix',
    type=str,
    required=True,
    help='Input PLINK file prefix (without .bed/.bim/.fam).'
)
@click.option(
    '--output-prefix',
    type=str,
    required=True,
    help='Output preprocessed PLINK file prefix.'
)
@click.option(
    '--plink-bin',
    type=str,
    default=None,
    help='Path to PLINK binary (default: auto-detect).'
)
@click.option(
    '--force',
    is_flag=True,
    help='Force re-preprocessing even if output exists.'
)
def preprocess_plink(model, plink_prefix, output_prefix, plink_bin, force):
    """
    Preprocess PLINK files to match model's SNP set and allele coding.

    This command aligns test data to the model's reference SNPs and ensures
    consistent allele coding using PLINK2's --alt1-allele force option.

    Example:
        dietnet preprocess-plink --model 1kgp_default \\
                                 --plink-prefix /path/to/test_data \\
                                 --output-prefix ./preprocessed/test
    """
    from pathlib import Path
    from Dietnet.model_manager import get_model_path
    from Dietnet.pretrained_models import PRETRAINED_MODELS
    from Dietnet.helpers.plink_utils import preprocess_test_data_with_plink, find_plink_binary

    # Resolve model path
    if model in PRETRAINED_MODELS:
        click.echo(f"Using model preset: {model}", err=True)
        model_dir = Path(get_model_path(model))
    else:
        model_dir = Path(model)
        if not model_dir.exists():
            click.echo(f"ERROR: Model directory not found: {model_dir}", err=True)
            sys.exit(1)
        click.echo(f"Using local model: {model_dir}", err=True)

    # Find first model package to get BIM file
    seed_dirs = sorted(model_dir.glob('seed_*'))
    if not seed_dirs:
        click.echo(f"ERROR: No seed directories found in {model_dir}", err=True)
        sys.exit(1)

    fold_dir = seed_dirs[0] / 'fold_0'
    bim_file = fold_dir / 'allpos.bim'

    if not bim_file.exists():
        click.echo(f"ERROR: Model BIM file not found: {bim_file}", err=True)
        sys.exit(1)

    # Find PLINK binary
    if plink_bin is None:
        plink_bin = find_plink_binary()
    click.echo(f"Using PLINK: {plink_bin}", err=True)

    # Preprocess
    click.echo(f"Preprocessing PLINK data...", err=True)
    click.echo(f"  Input: {plink_prefix}", err=True)
    click.echo(f"  Output: {output_prefix}", err=True)

    result = preprocess_test_data_with_plink(
        test_plink_prefix=plink_prefix,
        model_bim_file=str(bim_file),
        output_prefix=output_prefix,
        plink_bin=plink_bin,
        force=force
    )

    click.echo(f"✓ Preprocessing complete: {result}", err=True)


@main.command()
@click.option(
    '--predictions',
    type=click.Path(exists=True),
    required=True,
    help='Prediction text file (from dietnet predict).'
)
@click.option(
    '--labels',
    type=click.Path(exists=True),
    required=True,
    help='TSV file with true labels (sample_id<tab>label).'
)
@click.option(
    '--min-accuracy',
    type=float,
    default=None,
    help='Minimum expected accuracy (exits with error if below).'
)
@click.option(
    '--max-accuracy',
    type=float,
    default=None,
    help='Maximum expected accuracy (exits with error if above).'
)
def check(predictions, labels, min_accuracy, max_accuracy):
    """
    Validate predictions against true labels.

    Computes overall accuracy and per-class accuracy, optionally checking
    against expected accuracy thresholds.

    Example:
        dietnet check --predictions predictions.txt \\
                      --labels labels.tsv \\
                      --min-accuracy 0.85
    """
    import pandas as pd
    from pathlib import Path

    # Parse compact prediction format: "<sample_id> <label>" or "<sample_id> <labelA>(count) ..."
    lines = []
    for raw in Path(predictions).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sample_id = parts[0]
        label_token = parts[1]
        label = label_token.split('(')[0].rstrip('.,;')
        lines.append((sample_id, label))

    if not lines:
        click.echo(
            "ERROR: Could not parse predictions file. Expected lines like '<sample_id> CEUGBR(15)'.",
            err=True
        )
        sys.exit(1)

    pred_df = pd.DataFrame(lines, columns=['sample_id', 'predicted_class'])
    pred_df['sample_id'] = pred_df['sample_id'].astype(str)
    pred_df['predicted_class'] = pred_df['predicted_class'].astype(str)

    # Load labels and align types
    labels_df = pd.read_csv(labels, sep='\t')
    label_id_col = labels_df.columns[0]  # First column is always sample ID
    label_class_col = labels_df.columns[1]  # Second column is always the label
    labels_df[label_id_col] = labels_df[label_id_col].astype(str)
    labels_df[label_class_col] = labels_df[label_class_col].astype(str)

    # Merge on sample IDs
    merged = pred_df.merge(labels_df, left_on='sample_id', right_on=label_id_col)

    if len(merged) == 0:
        click.echo("ERROR: No matching samples found between predictions and labels", err=True)
        sys.exit(1)

    # Get true labels
    true_labels = merged[label_class_col]
    predicted_labels = merged['predicted_class']

    # Overall accuracy
    overall_accuracy = (predicted_labels == true_labels).mean()

    click.echo("=" * 60)
    click.echo("Prediction Validation Results")
    click.echo("=" * 60)
    click.echo(f"Total samples: {len(merged)}")
    click.echo(f"Overall accuracy: {overall_accuracy:.2%}")
    click.echo("")

    # Per-class accuracy
    click.echo("Per-class accuracy:")
    click.echo("-" * 60)
    for label in sorted(true_labels.unique()):
        mask = true_labels == label
        class_acc = (predicted_labels[mask] == true_labels[mask]).mean()
        class_count = mask.sum()
        click.echo(f"  {label:15s}: {class_acc:6.2%}  ({class_count:4d} samples)")
    click.echo("=" * 60)

    # Check thresholds
    if min_accuracy is not None:
        if overall_accuracy < min_accuracy:
            click.echo(f"❌ FAILED: Accuracy {overall_accuracy:.2%} below minimum {min_accuracy:.2%}", err=True)
            sys.exit(1)

    if max_accuracy is not None:
        if overall_accuracy > max_accuracy:
            click.echo(f"❌ FAILED: Accuracy {overall_accuracy:.2%} above maximum {max_accuracy:.2%}", err=True)
            sys.exit(1)

    click.echo("✓ PASSED")


@main.command()
@click.option(
    '--predictions',
    type=click.Path(exists=True),
    required=True,
    help='Prediction text file (compact format from dietnet predict).'
)
@click.option(
    '--labels',
    type=click.Path(exists=True),
    required=True,
    help='TSV file with true labels (sample_id<tab>label).'
)
@click.option(
    '--population',
    type=str,
    required=True,
    help='Target population label to analyze (must exist in labels file).'
)
@click.option(
    '--cmap',
    type=click.Path(exists=True),
    required=True,
    help='JSON file mapping population labels to hex colors.'
)
@click.option(
    '--output',
    type=click.Path(),
    required=True,
    help='Path to save stacked bar plot (e.g., figures/mxl_stack.png).'
)
@click.option(
    '--title',
    type=str,
    default=None,
    help='Plot title (default: auto-generated).'
)
def analyze_population(predictions, labels, population, cmap, output, title):
    """
    Generate a per-sample stacked bar plot of ensemble votes for a target population.

    Uses the compact text predictions (with vote counts) and true labels TSV.
    """
    from Dietnet.analysis.prediction_plots import plot_population_stack
    try:
        plot_population_stack(
            predictions_path=predictions,
            labels_path=labels,
            target_population=population,
            cmap_path=cmap,
            output_path=output,
            title=title,
        )
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)
    click.echo(f"✓ Saved stacked bar plot to {output}")


@main.command()
def info():
    """Display information about the DietNetwork installation."""
    import torch
    import numpy as np
    import pandas as pd
    import h5py

    click.echo("DietNetwork Installation Info")
    click.echo("=" * 50)
    click.echo(f"DietNetwork version: 0.2.0")
    click.echo(f"Python version: {sys.version.split()[0]}")
    click.echo(f"PyTorch version: {torch.__version__}")
    click.echo(f"NumPy version: {np.__version__}")
    click.echo(f"Pandas version: {pd.__version__}")
    click.echo(f"h5py version: {h5py.__version__}")
    click.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA version: {torch.version.cuda}")
        click.echo(f"GPU count: {torch.cuda.device_count()}")


if __name__ == '__main__':
    main()
