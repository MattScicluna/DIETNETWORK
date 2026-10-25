"""
Model packaging utilities for DietNetwork.

Provides a standardized format for saving and loading models with all
necessary metadata for inference, including SNP lists, normalization
statistics, embeddings, and configuration.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


class ModelPackage:
    """
    A standardized package format for DietNetwork models that includes
    all metadata needed for inference on new datasets.

    Package Structure:
        model_package/
        ├── model.pt                # PyTorch state dict
        ├── metadata.json           # Model configuration and info
        ├── snps.txt                # SNP list (chr:pos format)
        ├── input_stats.npz         # Mean/std for normalization
        ├── embedding.npz           # Auxiliary network embedding
        └── label_mapping.json      # Class names and indices
    """

    def __init__(self, package_dir: Union[str, Path]):
        """
        Initialize a ModelPackage.

        Args:
            package_dir: Path to the model package directory
        """
        self.package_dir = Path(package_dir)
        self._metadata = None
        self._label_mapping = None
        self._snps = None
        self._input_stats = None
        self._embedding = None
        self._model_state = None

    @property
    def metadata(self) -> Dict:
        """Load and cache metadata"""
        if self._metadata is None:
            metadata_path = self.package_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self._metadata = json.load(f)
        return self._metadata

    @property
    def label_mapping(self) -> Dict:
        """Load and cache label mapping"""
        if self._label_mapping is None:
            label_path = self.package_dir / "label_mapping.json"
            if label_path.exists():
                with open(label_path, "r") as f:
                    self._label_mapping = json.load(f)
        return self._label_mapping

    @property
    def snps(self) -> List[str]:
        """Load and cache SNP list"""
        if self._snps is None:
            snps_path = self.package_dir / "snps.txt"
            if snps_path.exists():
                with open(snps_path, "r") as f:
                    self._snps = [line.strip() for line in f]
        return self._snps

    @property
    def input_stats(self) -> Dict[str, np.ndarray]:
        """Load and cache input statistics"""
        if self._input_stats is None:
            stats_path = self.package_dir / "input_stats.npz"
            if stats_path.exists():
                self._input_stats = dict(np.load(stats_path))
        return self._input_stats

    @property
    def embedding(self) -> np.ndarray:
        """Load and cache embedding"""
        if self._embedding is None:
            emb_path = self.package_dir / "embedding.npz"
            if emb_path.exists():
                data = np.load(emb_path)
                # Handle both single embedding and fold-based embeddings
                if "embedding" in data:
                    self._embedding = data["embedding"]
                elif "emb" in data:
                    self._embedding = data["emb"]
                else:
                    # Take the first array found
                    self._embedding = list(data.values())[0]
        return self._embedding

    @property
    def seed(self) -> int:
        """Get seed from metadata"""
        return self.metadata.get("seed") if self.metadata else None

    @property
    def fold(self) -> int:
        """Get fold from metadata"""
        return self.metadata.get("fold") if self.metadata else None

    @property
    def package_root(self) -> Path:
        """
        Root directory that contains seed_* folders.
        Falls back to the package's parent if structure is non-standard.
        """
        parent = self.package_dir.parent
        if parent.name.startswith("seed_") and parent.parent.exists():
            return parent.parent
        return parent

    def load_model_state(self, device: str = "cpu") -> Dict:
        """
        Load PyTorch model state dict (device agnostic).

        Args:
            device: Device to map tensors to ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            Model state dict
        """
        if self._model_state is None:
            model_path = self.package_dir / "model.pt"
            if model_path.exists():
                self._model_state = torch.load(
                    model_path,
                    map_location=device,
                    weights_only=False,  # We trust our own model files
                )
        return self._model_state

    def save(
        self,
        model_state: Dict,
        snps: Union[List[str], np.ndarray, pd.Index],
        input_stats: Dict[str, np.ndarray],
        embedding: np.ndarray,
        label_mapping: Dict,
        config: Dict,
        seed: Optional[int] = None,
        fold: Optional[int] = None,
        training_info: Optional[Dict] = None,
        bim_file: Optional[Union[str, Path]] = None,
    ):
        """
        Save a complete model package.

        Args:
            model_state: PyTorch model state dict from model.state_dict()
            snps: List of SNP identifiers (chr:pos format or rsIDs)
            input_stats: Dict with 'mean' and optionally 'std' arrays for normalization
            embedding: Genotype frequency embedding array
            label_mapping: Dict mapping label names to indices
            config: Model hyperparameters and configuration
            seed: Random seed used for training
            fold: Fold number (0-indexed)
            training_info: Optional dict with training metrics, dates, etc.
            bim_file: Optional path to BIM file to copy (for PLINK preprocessing)
        """
        import shutil

        # Create package directory
        self.package_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(model_state, self.package_dir / "model.pt")
        print(f'Saved model state to {self.package_dir / "model.pt"}')

        # Save SNP list
        with open(self.package_dir / "snps.txt", "w") as f:
            if isinstance(snps, (pd.Index, pd.Series)):
                snps = snps.tolist()
            for snp in snps:
                f.write(f"{snp}\n")
        print(f'Saved {len(snps)} SNPs to {self.package_dir / "snps.txt"}')

        # Copy BIM file if provided (for PLINK preprocessing)
        if bim_file is not None:
            bim_source = Path(bim_file)
            if bim_source.exists():
                bim_dest = self.package_dir / "allpos.bim"
                shutil.copy2(bim_source, bim_dest)
                print(f"Saved BIM file to {bim_dest}")
            else:
                print(f"Warning: BIM file not found: {bim_file}")

        # Save input statistics
        np.savez(self.package_dir / "input_stats.npz", **input_stats)
        print(f'Saved input stats to {self.package_dir / "input_stats.npz"}')

        # Save embedding
        np.savez(self.package_dir / "embedding.npz", embedding=embedding)
        print(f'Saved embedding to {self.package_dir / "embedding.npz"}')

        # Save label mapping
        with open(self.package_dir / "label_mapping.json", "w") as f:
            json.dump(label_mapping, f, indent=2)
        print(f'Saved label mapping to {self.package_dir / "label_mapping.json"}')

        # Build metadata
        metadata = {
            "seed": seed,
            "fold": fold,
            "n_snps": len(snps),
            "n_classes": len(label_mapping) if isinstance(label_mapping, dict) else None,
            "config": config,
        }

        if training_info:
            metadata["training_info"] = training_info

        # Save metadata
        with open(self.package_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f'Saved metadata to {self.package_dir / "metadata.json"}')

        print(f"✓ Model package saved successfully to {self.package_dir}")

    @classmethod
    def load(cls, package_dir: Union[str, Path]) -> "ModelPackage":
        """
        Load a model package.

        Args:
            package_dir: Path to the package directory

        Returns:
            ModelPackage instance
        """
        package_dir = Path(package_dir)
        if not package_dir.exists():
            raise FileNotFoundError(f"Model package not found: {package_dir}")

        # Verify required files exist
        required_files = [
            "model.pt",
            "metadata.json",
            "snps.txt",
            "input_stats.npz",
            "embedding.npz",
            "label_mapping.json",
        ]

        missing_files = []
        for fname in required_files:
            if not (package_dir / fname).exists():
                missing_files.append(fname)

        if missing_files:
            raise ValueError(f"Incomplete model package. Missing files: {', '.join(missing_files)}")

        package = cls(package_dir)
        # Quietly load package (verbose output handled by caller if needed)
        return package

    def __repr__(self):
        return (
            f"ModelPackage(dir={self.package_dir}, "
            f"seed={self.seed}, fold={self.fold}, "
            f"n_snps={len(self.snps) if self.snps else 0})"
        )


def find_model_packages(
    root_dir: Union[str, Path], seeds: Optional[List[int]] = None, folds: Optional[List[int]] = None
) -> List[ModelPackage]:
    """
    Find and load all model packages in a directory tree.

    Searches for directories containing 'metadata.json' and loads them
    as ModelPackage instances.

    Args:
        root_dir: Root directory to search
        seeds: Optional list of seeds to filter by
        folds: Optional list of folds to filter by

    Returns:
        List of ModelPackage instances
    """
    root_dir = Path(root_dir)
    packages = []

    # Find all metadata.json files
    for metadata_file in root_dir.rglob("metadata.json"):
        package_dir = metadata_file.parent
        try:
            pkg = ModelPackage.load(package_dir)

            # Apply filters
            if seeds is not None and pkg.seed not in seeds:
                continue
            if folds is not None and pkg.fold not in folds:
                continue

            packages.append(pkg)
        except Exception as e:
            print(f"Warning: Failed to load package from {package_dir}: {e}")
            continue

    # Sort by seed, then fold
    packages.sort(key=lambda p: (p.seed or 0, p.fold or 0))

    return packages


def convert_legacy_model(
    model_path: Union[str, Path],
    dataset_file: Union[str, Path],
    embedding_file: Union[str, Path],
    input_stats_file: Union[str, Path],
    config_file: Union[str, Path],
    output_dir: Union[str, Path],
    seed: Optional[int] = None,
    fold: Optional[int] = None,
) -> ModelPackage:
    """
    Convert a legacy model (just .pt file) to the new package format.

    This extracts metadata from the HDF5 dataset, embedding, stats, and
    config files that were used during training.

    Args:
        model_path: Path to .pt model file
        dataset_file: Path to HDF5 dataset file (for SNP names, labels)
        embedding_file: Path to embedding.npz file
        input_stats_file: Path to input_features_means.npz or input_stats.npz
        config_file: Path to config.yaml file
        output_dir: Where to save the model package
        seed: Random seed used (extract from path if None)
        fold: Fold number (extract from path if None)

    Returns:
        ModelPackage instance
    """
    import h5py
    import yaml

    # Load model state (device agnostic)
    model_state = torch.load(model_path, map_location="cpu", weights_only=False)

    # Load config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Load SNP names and labels from HDF5
    with h5py.File(dataset_file, "r") as f:
        snps = f["snp_names"][:].astype(str).tolist()

        # Get label names
        if "class_label_names" in f:
            label_names = f["class_label_names"][:].astype(str).tolist()
        elif "label_names" in f:
            label_names = f["label_names"][:].astype(str).tolist()
        else:
            raise ValueError("No label names found in dataset")

    # Create label mapping
    label_mapping = {name: idx for idx, name in enumerate(label_names)}

    # Load embedding for this fold
    emb_data = np.load(embedding_file)
    if fold is not None and "emb_by_fold" in emb_data:
        embedding = emb_data["emb_by_fold"][fold]
    elif "embedding" in emb_data:
        embedding = emb_data["embedding"]
    else:
        # Take first array
        embedding = list(emb_data.values())[0]

    # Load input stats for this fold
    stats_data = np.load(input_stats_file)
    if fold is not None and "means_by_fold" in stats_data:
        means = stats_data["means_by_fold"][fold]
    elif "mean" in stats_data:
        means = stats_data["mean"]
    else:
        means = list(stats_data.values())[0]

    input_stats = {"mean": means}
    if "std" in stats_data or "stds_by_fold" in stats_data:
        if fold is not None and "stds_by_fold" in stats_data:
            input_stats["std"] = stats_data["stds_by_fold"][fold]
        elif "std" in stats_data:
            input_stats["std"] = stats_data["std"]

    # Create package
    package = ModelPackage(output_dir)
    package.save(
        model_state=model_state,
        snps=snps,
        input_stats=input_stats,
        embedding=embedding,
        label_mapping=label_mapping,
        config=config,
        seed=seed,
        fold=fold,
    )

    return package
