"""
Inference script for DietNetwork with PLINK preprocessing.

Uses PLINK --real-ref-alleles to ensure consistent allele coding between
training and test datasets.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Dietnet.helpers import model as model_module
from Dietnet.helpers.dataset_utils import InferenceDataset
from Dietnet.helpers.model_package import ModelPackage, find_model_packages
from Dietnet.helpers.plink_utils import find_plink_binary, preprocess_test_data_with_plink


def _cache_base_prefix(plink_prefix: str, model_root: Path, temp_dir: Path) -> Path:
    """
    Build a shared prefix for preprocessed PLINK files and genotype cache.
    The key hashes the absolute plink path and model root to avoid collisions
    while reusing across ensemble models for the same dataset/model.
    """
    plink_abs = os.path.abspath(plink_prefix)
    model_root_abs = os.path.abspath(model_root)
    payload = json.dumps(
        {"plink": plink_abs, "model_root": model_root_abs}, sort_keys=True
    ).encode()
    digest = hashlib.sha1(payload).hexdigest()[:8]
    return Path(temp_dir) / f"{Path(plink_prefix).name}_{digest}"


def load_model_from_package(model_package, device="cpu"):
    """Load a DietNetwork model from a model package."""
    # Get model configuration
    config = model_package.metadata["config"]
    n_classes = model_package.metadata["n_classes"]

    # Get embedding to determine input size
    embedding = model_package.embedding
    if len(embedding.shape) == 1:
        n_feats_emb = 1
    else:
        n_feats_emb = embedding.shape[1]

    n_feats = embedding.shape[0]

    # Create model
    combined_model = model_module.CombinedModel(
        n_feats=n_feats_emb,
        n_hidden_u_aux=config["nb_hidden_u_aux"],
        n_hidden_u_main=config["nb_hidden_u_aux"][-1:] + config["nb_hidden_u_main"],
        n_targets=n_classes,
        param_init=None,
        aux_uniform_init_limit=config.get("uniform_init_limit", 0.02),
        input_dropout=config.get("input_dropout", 0.995),
    )

    # Handle legacy checkpoints (aux_net/main_net) by remapping keys
    def _remap_state_dict(state_dict):
        remapped = {}
        for key, val in state_dict.items():
            if key.startswith("aux_net."):
                new_key = "feat_emb." + key[len("aux_net.") :]
            elif key.startswith("main_net."):
                new_key = "disc_net." + key[len("main_net.") :]
            else:
                new_key = key
            remapped[new_key] = val
        return remapped

    # Load state dict
    checkpoint = model_package.load_model_state(device=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = _remap_state_dict(state_dict)

    combined_model.load_state_dict(state_dict, strict=False)
    combined_model.to(device)
    combined_model.eval()

    return combined_model


def predict_single_model(
    model_package,
    plink_prefix,
    device="cpu",
    batch_size=128,
    num_workers=0,
    plink_bin=None,
    temp_dir=None,
    skip_preprocess=False,
    force_preprocess=False,
    cache_prefix=None,
    force_cache_rebuild=False,
    verbose=True,
    return_logits=False,
    return_hidden=False,
):
    """
    Run inference with a single model using PLINK preprocessing.

    Args:
        model_package: ModelPackage instance
        plink_prefix: Path to test PLINK files (or preprocessed files if skip_preprocess=True)
        device: Device for inference
        batch_size: Batch size
        num_workers: Number of data loader workers
        plink_bin: Path to PLINK binary (auto-detected if None)
        temp_dir: Directory for preprocessed files
        skip_preprocess: If True, assume plink_prefix is already preprocessed
        force_preprocess: If True, rerun preprocessing even if exists
        cache_prefix: Base path for shared cache artifacts
        force_cache_rebuild: If True, rebuild genotype cache even if present

    Returns:
        Tuple of (sample_ids, predictions, probabilities, raw_logits, hidden_reps)
    """
    if verbose:
        print(f"\nRunning inference: seed {model_package.seed}, fold {model_package.fold}")

    if temp_dir is None:
        # Default alongside the provided PLINK prefix
        temp_dir = Path(plink_prefix).parent
    else:
        temp_dir = Path(temp_dir)

    if cache_prefix is None:
        cache_prefix = temp_dir / Path(plink_prefix).name

    preprocess_prefix = cache_prefix.with_name(f"{cache_prefix.name}_preprocessed")
    cache_file = cache_prefix.with_name(f"{cache_prefix.name}_genotypes.npy")

    # Determine which PLINK file to use
    if skip_preprocess:
        # User provided already-preprocessed PLINK file
        preprocessed_plink_prefix = plink_prefix
        if verbose:
            print(f"Using preprocessed PLINK: {preprocessed_plink_prefix}")
    else:
        # Need to preprocess
        # Check if model has BIM file
        bim_file = model_package.package_dir / "allpos.bim"
        if not bim_file.exists():
            raise FileNotFoundError(
                f"Model package missing BIM file: {bim_file}\n"
                "Please re-convert models with updated convert_legacy_models.py"
            )

        # Find PLINK binary
        if plink_bin is None:
            plink_bin = find_plink_binary()
        if verbose:
            print(f"Using PLINK: {plink_bin}")

        # Preprocess test data with PLINK
        temp_dir.mkdir(parents=True, exist_ok=True)

        preprocessed_plink_prefix = preprocess_test_data_with_plink(
            test_plink_prefix=plink_prefix,
            model_bim_file=str(bim_file),
            output_prefix=str(preprocess_prefix),
            plink_bin=plink_bin,
            force=force_preprocess,
        )

    # Load model
    if verbose:
        print("Loading model...")
    model = load_model_from_package(model_package, device=device)

    # Load and normalize embedding
    embedding = torch.from_numpy(model_package.embedding).float().to(device)
    emb_norm = (embedding**2).sum(0) ** 0.5
    embedding = embedding / emb_norm
    if len(embedding.size()) == 1:
        embedding = torch.unsqueeze(embedding, dim=1)

    # Create dataset (loads preprocessed PLINK with memory mapping)
    if verbose:
        print("\nCreating inference dataset (shared cache)...")
    dataset = InferenceDataset(
        plink_prefix=preprocessed_plink_prefix,
        model_package=model_package,
        use_memmap=True,
        cache_file=str(cache_file),
        force_cache_rebuild=force_cache_rebuild,
        verbose=verbose,
    )

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    # Run inference
    if verbose:
        print(f"\nRunning inference on {len(dataset)} samples...")
    import warnings

    warnings.filterwarnings(
        "ignore", message="This DataLoader will create .* worker processes", category=UserWarning
    )
    all_sample_ids = []
    all_logits = []
    all_hidden = []

    with torch.no_grad():
        for batch_geno, batch_sample_ids in tqdm(
            loader, desc="Inference", disable=not verbose, leave=False
        ):
            batch_geno = batch_geno.to(device)
            if return_hidden or return_logits:
                hidden_batch, logits = model(embedding, batch_geno, save_layers=True)
                if return_hidden:
                    all_hidden.append(hidden_batch.cpu())
            else:
                logits = model(embedding, batch_geno)
            all_logits.append(logits.cpu())
            all_sample_ids.extend(batch_sample_ids)

    # Concatenate results
    all_logits = torch.cat(all_logits, dim=0)
    probabilities = F.softmax(all_logits, dim=1).numpy()
    predictions = torch.argmax(all_logits, dim=1).numpy()
    raw_logits = all_logits.numpy() if return_logits else None
    hidden_reps = torch.cat(all_hidden, dim=0).numpy() if all_hidden else None

    if verbose:
        print("✓ Inference complete")

    return all_sample_ids, predictions, probabilities, raw_logits, hidden_reps


def predict_ensemble(
    model_packages,
    plink_prefix,
    device="cpu",
    batch_size=128,
    num_workers=0,
    plink_bin=None,
    temp_dir=None,
    skip_preprocess=False,
    force_preprocess=False,
    cache_prefix=None,
    label_mapping=None,
    save_logits=False,
    save_hidden=False,
):
    """
    Run ensemble inference with multiple models.

    Args:
        model_packages: List of ModelPackage instances
        plink_prefix: Path to test PLINK files
        device: Device for inference
        batch_size: Batch size
        num_workers: Number of data loader workers
        plink_bin: Path to PLINK binary
        temp_dir: Directory for preprocessed files and genotype cache
        skip_preprocess: Skip preprocessing if True
        force_preprocess: Force re-preprocessing if True
        cache_prefix: Shared prefix for preprocessed PLINK and genotype cache
        label_mapping: Label mapping dict

    Returns:
        Tuple of (sample_ids, ensemble_predictions, ensemble_probabilities,
                  agreement_stats, individual_predictions, agg_logits,
                  per_model_logits, per_model_hidden)
    """
    all_predictions = []
    all_probabilities = []
    all_logits = []
    all_hidden = []
    sample_ids = None

    # Determine preprocessing strategy and shared prefixes
    if temp_dir is None:
        temp_dir = Path(plink_prefix).parent
    else:
        temp_dir = Path(temp_dir)

    model_root = model_packages[0].package_root if model_packages else Path(".")
    cache_prefix = cache_prefix or _cache_base_prefix(plink_prefix, model_root, temp_dir)
    preprocess_prefix = cache_prefix.with_name(f"{cache_prefix.name}_preprocessed")
    preprocessed_plink = plink_prefix if skip_preprocess else str(preprocess_prefix)

    # Run inference with each model
    model_bar = tqdm(model_packages, desc="Models", unit="model")
    for i, pkg in enumerate(model_bar, 1):
        model_bar.set_postfix_str(f"seed {pkg.seed} fold {pkg.fold}")

        # First model builds preprocessing/cache unless user provided preprocessed
        if i == 1:
            current_plink = plink_prefix if not skip_preprocess else preprocessed_plink
            skip_prep = skip_preprocess
            force_prep = force_preprocess
            current_temp_dir = temp_dir if not skip_preprocess else None
            force_cache_rebuild = force_preprocess
        else:
            current_plink = preprocessed_plink if not skip_preprocess else plink_prefix
            skip_prep = True
            force_prep = False
            current_temp_dir = temp_dir if not skip_preprocess else None
            force_cache_rebuild = False

        ids, preds, probs, logits, hidden = predict_single_model(
            model_package=pkg,
            plink_prefix=current_plink,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            plink_bin=plink_bin,
            temp_dir=current_temp_dir,
            skip_preprocess=skip_prep,
            force_preprocess=force_prep,
            cache_prefix=cache_prefix,
            force_cache_rebuild=force_cache_rebuild,
            verbose=(i == 1),
            return_logits=save_logits,
            return_hidden=save_hidden,
        )

        if sample_ids is None:
            sample_ids = ids
        else:
            # Verify sample IDs match across models
            if ids != sample_ids:
                raise ValueError(f"Sample ID mismatch in model {i}")

        all_predictions.append(preds)
        all_probabilities.append(probs)
        if save_logits and logits is not None:
            all_logits.append(logits)
        if save_hidden and hidden is not None:
            all_hidden.append(hidden)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples)
    all_probabilities = np.array(all_probabilities)  # Shape: (n_models, n_samples, n_classes)
    per_model_logits = np.array(all_logits) if all_logits else None
    per_model_hidden = np.array(all_hidden) if all_hidden else None

    # Compute ensemble predictions (majority vote)
    ensemble_predictions = []
    agreement_fractions = []

    for i in range(all_predictions.shape[1]):  # For each sample
        sample_preds = all_predictions[:, i]
        unique, counts = np.unique(sample_preds, return_counts=True)
        # Most common prediction
        majority_pred = unique[np.argmax(counts)]
        ensemble_predictions.append(majority_pred)
        # Agreement fraction
        agreement = np.max(counts) / len(sample_preds)
        agreement_fractions.append(agreement)

    ensemble_predictions = np.array(ensemble_predictions)
    agreement_fractions = np.array(agreement_fractions)

    # Average probabilities across models
    ensemble_probabilities = np.mean(all_probabilities, axis=0)
    ensemble_logits = np.mean(per_model_logits, axis=0) if per_model_logits is not None else None

    # Compute agreement statistics
    agreement_stats = {
        "mean_agreement": np.mean(agreement_fractions),
        "min_agreement": np.min(agreement_fractions),
        "max_agreement": np.max(agreement_fractions),
        "agreement_fractions": agreement_fractions,
        "n_models": len(model_packages),
        "seeds": sorted(set(pkg.seed for pkg in model_packages)),
        "folds": sorted(set(pkg.fold for pkg in model_packages)),
    }

    print(f"\n✓ Ensemble prediction complete")
    print(f"  Mean agreement: {agreement_stats['mean_agreement']:.2%}")
    print(
        f"  Agreement range: {agreement_stats['min_agreement']:.2%} - {agreement_stats['max_agreement']:.2%}"
    )

    return (
        sample_ids,
        ensemble_predictions,
        ensemble_probabilities,
        agreement_stats,
        all_predictions,
        ensemble_logits,
        per_model_logits,
        per_model_hidden,
    )


def save_predictions(
    output_file,
    sample_ids,
    predictions,
    probabilities,
    label_mapping,
    agreement_stats=None,
    individual_predictions=None,
):
    """
    Save predictions in a compact, human-readable format.

    For single-model predictions:
        <sample_id> <prediction>

    For ensemble predictions:
        <sample_id> <labelA>(count) <labelB>(count) ...
    """
    idx_to_label = {idx: label for label, idx in label_mapping.items()}
    sample_strs = [str(sid) for sid in sample_ids]
    id_width = max(len(sid) for sid in sample_strs)
    lines = []

    if individual_predictions is not None and len(np.atleast_2d(individual_predictions)) > 0:
        votes = np.atleast_2d(individual_predictions)
        for i, sid in enumerate(sample_strs):
            counts = Counter(votes[:, i])
            ordered = sorted(counts.items(), key=lambda item: (-item[1], idx_to_label[item[0]]))
            label_tokens = [f"{idx_to_label[idx]}({count})" for idx, count in ordered]
            lines.append(f"{sid.rjust(id_width)} " + " ".join(label_tokens))
    else:
        for sid, pred in zip(sample_strs, predictions):
            lines.append(f"{sid.rjust(id_width)} {idx_to_label[pred]}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")

    print(f"✓ Saved predictions to {output_path}")
    print(f"  {len(sample_ids)} samples, {len(label_mapping)} classes")
    if agreement_stats is not None:
        print(f"  Ensemble of {agreement_stats['n_models']} models")
        print(f"  Seeds: {agreement_stats['seeds']}")
        print(f"  Folds: {agreement_stats['folds']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DietNetwork inference with PLINK preprocessing"
    )

    parser.add_argument(
        "--model-dir", type=str, required=True, help="Directory containing model package(s)"
    )

    parser.add_argument(
        "--plink-prefix", type=str, required=True, help="PLINK file prefix (without .bed/.bim/.fam)"
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Output file for predictions (text)"
    )

    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, help="Seeds to use (default: all)"
    )

    parser.add_argument(
        "--folds", type=int, nargs="+", default=None, help="Folds to use (default: all)"
    )

    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available, else cpu)",
    )

    parser.add_argument(
        "--plink-bin", type=str, default=None, help="Path to PLINK binary (default: auto-detect)"
    )

    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of data loader workers (default: 0)"
    )

    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Directory for preprocessed PLINK files and genotype cache (default: alongside --plink-prefix)",
    )

    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip PLINK preprocessing (use if plink-prefix is already preprocessed)",
    )

    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Force re-preprocessing even if preprocessed file exists",
    )
    parser.add_argument(
        "--save-logits",
        type=str,
        default=None,
        help="Optional path to save raw logits/probabilities (.npz)",
    )
    parser.add_argument(
        "--save-hidden",
        type=str,
        default=None,
        help="Optional path to save final hidden representations (.npz)",
    )

    args = parser.parse_args()

    # Find model packages
    model_packages = find_model_packages(
        root_dir=args.model_dir, seeds=args.seeds, folds=args.folds
    )

    if not model_packages:
        print(f"ERROR: No model packages found in {args.model_dir}")
        sys.exit(1)

    seeds = sorted(set(pkg.seed for pkg in model_packages))
    folds = sorted(set(pkg.fold for pkg in model_packages))
    print(f"Found {len(model_packages)} model(s): seeds {seeds}, folds {folds}")

    # Get label mapping
    label_mapping = model_packages[0].label_mapping
    individual_preds = None
    agg_logits = None
    per_model_logits = None
    per_model_hidden = None

    model_root = model_packages[0].package_root
    default_temp = Path(args.plink_prefix).parent
    temp_dir_path = Path(args.temp_dir) if args.temp_dir else default_temp
    cache_prefix = _cache_base_prefix(args.plink_prefix, model_root, temp_dir_path)

    # Run inference with all models (ensemble)
    if len(model_packages) == 1:
        print("\nRunning inference with single model...")
        sample_ids, predictions, probabilities, raw_logits, hidden_reps = predict_single_model(
            model_package=model_packages[0],
            plink_prefix=args.plink_prefix,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            plink_bin=args.plink_bin,
            temp_dir=args.temp_dir,
            skip_preprocess=args.skip_preprocess,
            force_preprocess=args.force_preprocess,
            cache_prefix=cache_prefix,
            force_cache_rebuild=args.force_preprocess,
            return_logits=bool(args.save_logits),
            return_hidden=bool(args.save_hidden),
        )
        agreement_stats = None
        agg_logits = raw_logits
        per_model_hidden = hidden_reps
    else:
        print(f"\nRunning ensemble inference with {len(model_packages)} models...")
        (
            sample_ids,
            predictions,
            probabilities,
            agreement_stats,
            individual_preds,
            agg_logits,
            per_model_logits,
            per_model_hidden,
        ) = predict_ensemble(
            model_packages=model_packages,
            plink_prefix=args.plink_prefix,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            plink_bin=args.plink_bin,
            temp_dir=args.temp_dir,
            skip_preprocess=args.skip_preprocess,
            force_preprocess=args.force_preprocess,
            cache_prefix=cache_prefix,
            label_mapping=label_mapping,
            save_logits=bool(args.save_logits),
            save_hidden=bool(args.save_hidden),
        )

    # Save predictions
    save_predictions(
        output_file=args.output,
        sample_ids=sample_ids,
        predictions=predictions,
        probabilities=probabilities,
        label_mapping=label_mapping,
        agreement_stats=agreement_stats,
        individual_predictions=individual_preds if len(model_packages) > 1 else None,
    )

    def save_extra_outputs(path, sample_ids, label_mapping, **arrays):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        idx_to_label = {idx: label for label, idx in label_mapping.items()}
        payload = {
            "sample_id": np.array(sample_ids),
            "label_names": np.array([idx_to_label[i] for i in range(len(idx_to_label))]),
        }
        for key, value in arrays.items():
            if value is not None:
                payload[key] = value
        np.savez(output_path, **payload)
        print(f"✓ Saved extras to {output_path}")

    if args.save_logits:
        save_extra_outputs(
            path=args.save_logits,
            sample_ids=sample_ids,
            label_mapping=label_mapping,
            logits=agg_logits,
            probabilities=probabilities,
            per_model_logits=per_model_logits,
            seeds=seeds,
            folds=folds,
        )

    if args.save_hidden:
        save_extra_outputs(
            path=args.save_hidden,
            sample_ids=sample_ids,
            label_mapping=label_mapping,
            hidden=per_model_hidden if len(model_packages) == 1 else None,
            per_model_hidden=per_model_hidden if len(model_packages) > 1 else None,
            seeds=seeds,
            folds=folds,
        )

    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
