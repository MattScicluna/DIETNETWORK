"""
Model management utilities for downloading, caching, and validating pretrained models.
"""

import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

from Dietnet.pretrained_models import PRETRAINED_MODELS


def get_cache_dir() -> Path:
    """
    Get cache directory for models.

    Returns:
        Path to cache directory (~/.cache/dietnet/)
    """
    cache_home = Path.home() / ".cache" / "dietnet"
    cache_home.mkdir(parents=True, exist_ok=True)
    return cache_home


def get_model_path(
    preset: str, force_download: bool = False, allow_local_fallback: bool = True
) -> Path:
    """
    Resolve model preset to local path, downloading if needed.

    Args:
        preset: Model preset name (e.g., '1kgp_default')
        force_download: Force re-download even if cached
        allow_local_fallback: If download fails, try local model directory

    Returns:
        Path to model directory (contains seed_X/fold_Y/)

    Raises:
        ValueError: If preset is unknown or has no download URL
    """
    if preset not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model preset: {preset}")

    config = PRETRAINED_MODELS[preset]
    cache_dir = get_cache_dir() / config["cache_dir"]

    # Check if already cached
    if cache_dir.exists() and not force_download:
        if validate_model(cache_dir):
            print(f"✓ Using cached model: {cache_dir}", file=sys.stderr)
            return cache_dir
        else:
            print(f"⚠ Cached model invalid, will try to re-download...", file=sys.stderr)
            shutil.rmtree(cache_dir)

    # Try to download and extract
    try:
        print(f"Downloading model preset '{preset}'...", file=sys.stderr)
        download_model(preset, cache_dir)
        return cache_dir
    except Exception as e:
        # Download failed - try local fallback
        if allow_local_fallback and preset == "1kgp_default":
            print(f"\n⚠ Download failed: {e}", file=sys.stderr)
            print("Checking for local model directory...", file=sys.stderr)

            # Try to find local model directory
            from pathlib import Path

            possible_paths = [
                Path.cwd() / "pretrained_1000g_models",
                Path(__file__).parent.parent / "pretrained_1000g_models",
            ]

            for local_path in possible_paths:
                if local_path.exists() and validate_model(local_path):
                    print(f"✓ Using local model directory: {local_path}", file=sys.stderr)
                    return local_path

            # No local fallback found
            raise RuntimeError(
                f"Download failed and no valid local model found.\n"
                f"Tried: {[str(p) for p in possible_paths]}\n"
                f"Original error: {e}"
            )
        else:
            raise


def download_model(preset: str, dest: Path):
    """
    Download and extract model package.

    Args:
        preset: Model preset name
        dest: Destination directory for extracted model

    Raises:
        ValueError: If model has no download URL or checksum fails
    """
    config = PRETRAINED_MODELS[preset]

    if config.get("model_url") is None or config["model_url"] == "PLACEHOLDER_MODEL_URL":
        raise ValueError(
            f"Model preset '{preset}' has no download URL configured yet.\n"
            f"Please check the documentation or wait for the model to be uploaded."
        )

    # Download to temp file
    temp_file = dest.parent / f"{preset}_temp.tar.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {config['model_url']}...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(config["model_url"], temp_file, reporthook=_download_progress)
        print(file=sys.stderr)  # New line after progress
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to download model: {e}")

    # Validate checksum
    if config.get("model_sha256") and config["model_sha256"] != "PLACEHOLDER_SHA256":
        print("Validating checksum...", file=sys.stderr)
        actual_hash = compute_sha256(temp_file)
        if actual_hash != config["model_sha256"]:
            temp_file.unlink()
            raise ValueError(
                f"Checksum mismatch!\n"
                f"Expected: {config['model_sha256']}\n"
                f"Got:      {actual_hash}\n"
                f"The downloaded file may be corrupted."
            )
        print("✓ Checksum valid", file=sys.stderr)

    # Extract
    print(f"Extracting to {dest}...", file=sys.stderr)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(temp_file, "r:gz") as tar:
            tar.extractall(dest)
    except Exception as e:
        temp_file.unlink()
        if dest.exists():
            shutil.rmtree(dest)
        raise RuntimeError(f"Failed to extract model: {e}")

    # Cleanup
    temp_file.unlink()
    print("✓ Model download complete", file=sys.stderr)


def validate_model(model_dir: Path) -> bool:
    """
    Check if model directory has required files.

    Args:
        model_dir: Path to model directory

    Returns:
        True if valid model package, False otherwise
    """
    # Should have seed_X/fold_Y/ structure
    seed_dirs = list(model_dir.glob("seed_*"))
    if not seed_dirs:
        return False

    # Check first model has required files
    fold_dirs = list(seed_dirs[0].glob("fold_*"))
    if not fold_dirs:
        return False

    fold_dir = fold_dirs[0]
    required_files = [
        "model.pt",
        "metadata.json",
        "snps.txt",
        "input_stats.npz",
        "embedding.npz",
        "label_mapping.json",
        "allpos.bim",
    ]

    return all((fold_dir / f).exists() for f in required_files)


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 checksum of file.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal SHA256 checksum
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def clear_cache(preset: Optional[str] = None):
    """
    Clear cached models.

    Args:
        preset: Optional preset name to clear only that model.
                If None, clears all cached models.
    """
    cache_dir = get_cache_dir()

    if preset:
        if preset not in PRETRAINED_MODELS:
            raise ValueError(f"Unknown model preset: {preset}")

        model_cache = cache_dir / PRETRAINED_MODELS[preset]["cache_dir"]
        if model_cache.exists():
            shutil.rmtree(model_cache)
            print(f"✓ Cleared cache for {preset}", file=sys.stderr)
        else:
            print(f"No cache found for {preset}", file=sys.stderr)
    else:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("✓ Cleared all cached models", file=sys.stderr)
        else:
            print("No cached models found", file=sys.stderr)


def _download_progress(block_num, block_size, total_size):
    """Progress callback for urllib.request.urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        bar_length = 40
        filled = int(bar_length * downloaded / total_size)
        bar = "=" * filled + "-" * (bar_length - filled)
        print(
            f"\r[{bar}] {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)",
            end="",
            flush=True,
            file=sys.stderr,
        )
