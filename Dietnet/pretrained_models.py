"""
Registry of pretrained DietNetwork models and test datasets.

This module defines available model presets that can be downloaded and cached
automatically, as well as test datasets for validation.
"""

from pathlib import Path
from typing import Any, Dict, Optional

PRETRAINED_MODELS: Dict[str, Dict[str, Any]] = {
    "1kgp_default": {
        "name": "1000 Genomes Phase 3 (Ensemble)",
        "version": "v1",
        "description": "Ensemble of 15 models (3 seeds × 5 folds) trained on 1KGP 2504 samples, 24 populations",
        "n_models": 15,
        "seeds": [78, 79, 80],
        "folds": [0, 1, 2, 3, 4],
        "n_classes": 24,
        "n_snps": 229986,
        "training_data": "1000 Genomes Phase 3 (WGS30X)",
        "model_url": "https://www.dropbox.com/scl/fi/eqkz5hmo9mlmy0mu78mi6/dietnet_1kgp_default_v1.tar.gz?rlkey=tf5knko5byo7ab7c3fz5tg0xw&st=kbb4bych&dl=1",  # User fills after upload
        "model_sha256": "a9c664951064e29404c4aa37a6c2a98bd481ad796b3e966159be216dc650a9cd",
        "size_mb": 1600,  # ~1.6 GB for 15 models
        "cache_dir": "1kgp_default_v1",
        "populations": [
            "ACB",
            "ASW",
            "BEB",
            "CDX",
            "CEUGBR",
            "CHB",
            "CHS",
            "CLM",
            "ESN",
            "FIN",
            "GIH",
            "GWD",
            "IBS",
            "JPT",
            "KHV",
            "LWK",
            "MSL",
            "MXL",
            "PEL",
            "PJL",
            "PUR",
            "STUITU",
            "TSI",
            "YRI",
        ],
    },
    "hgdp_ukbb": {
        "name": "HGDP + 1KGP for UKBB Inference",
        "version": "v1",
        "description": "Trained on HGDP+1KGP for UKBB population inference",
        "model_url": None,  # To be added later
        "model_sha256": None,
        "cache_dir": "hgdp_ukbb_v1",
    },
}

TEST_DATA: Dict[str, Dict[str, Any]] = {
    "1kgp_default": {
        "name": "1000 Genomes Phase 3 (Test Subset)",
        "description": "Test subset for validating 1KGP model accuracy",
        "plink_url": "https://www.dropbox.com/scl/fi/dqumjkv8kxfbt3caog81h/dietnet_1kgp_test_data_v1.tar.gz?rlkey=wl1u2cufinui7k0m9pp4197n5&st=u5cy2q6z&dl=1",  # User fills after upload
        "plink_sha256": "fbf5dab7afca99008870e669a9efe20cbca19c041695a23536a848258c7400e5",
        "expected_accuracy_min": 0.85,  # 85% minimum expected
        "expected_accuracy_max": 1.0,  # 100% maximum expected
        "plink_basename": "1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1",
        "labels_filename": "labels_pop_subsampleV1.tsv",
    }
}


def list_available_models() -> Dict[str, str]:
    """
    Get a dictionary of available model presets.

    Returns:
        Dict mapping preset names to descriptions
    """
    return {
        name: config.get("description", "No description available")
        for name, config in PRETRAINED_MODELS.items()
    }


def get_model_info(preset: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a model preset.

    Args:
        preset: Model preset name

    Returns:
        Model configuration dict, or None if preset doesn't exist
    """
    return PRETRAINED_MODELS.get(preset)


def is_model_available(preset: str) -> bool:
    """
    Check if a model preset has a download URL configured.

    Args:
        preset: Model preset name

    Returns:
        True if model can be downloaded, False otherwise
    """
    if preset not in PRETRAINED_MODELS:
        return False

    config = PRETRAINED_MODELS[preset]
    return config.get("model_url") is not None and config["model_url"] != "PLACEHOLDER_MODEL_URL"
