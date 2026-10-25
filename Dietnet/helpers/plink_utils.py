"""
PLINK preprocessing utilities for DietNetwork inference.

Handles allele coding consistency between training and test datasets.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def preprocess_test_data_with_plink(
    test_plink_prefix: str,
    model_bim_file: str,
    output_prefix: str,
    plink_bin: str = "plink2",
    force: bool = False,
) -> str:
    """
    Use PLINK2 to preprocess test data to match training allele coding.

    This runs:
        plink2 --bfile test_data \\
               --extract model_snps.bim \\
               --alt1-allele force model_snps.bim 5 2 \\
               --make-bed \\
               --out preprocessed_test

    The --alt1-allele force flag ensures that the A1 allele in the output BIM
    matches the A1 allele (column 5) from the model's BIM file, ensuring
    consistent allele coding between training and test datasets.

    Args:
        test_plink_prefix: Path to test PLINK files (without .bed/.bim/.fam)
        model_bim_file: Path to model's BIM file (defines reference alleles)
        output_prefix: Where to save preprocessed PLINK files
        plink_bin: Path to PLINK2 binary (default: 'plink2' in PATH)
        force: If True, rerun preprocessing even if output exists

    Returns:
        output_prefix: Path to preprocessed PLINK files (without extension)
    """
    output_prefix = str(output_prefix)

    # Check if already preprocessed
    if not force and os.path.exists(f"{output_prefix}.bed"):
        print(f"\n✓ Preprocessed PLINK file already exists: {output_prefix}.bed")
        print("  (Use --force-preprocess to regenerate)")
        return output_prefix

    # Run PLINK2 to extract SNPs and force correct allele coding
    # --alt1-allele force <file> 5 2 means:
    #   - Use column 5 (a1) from model_bim_file
    #   - Match variants by column 2 (variant ID)
    #   - 'force' allows changing already-set alleles
    cmd = [
        plink_bin,
        "--bfile",
        test_plink_prefix,
        "--extract",
        model_bim_file,
        "--alt1-allele",
        "force",
        model_bim_file,
        "5",
        "2",
        "--make-bed",
        "--out",
        output_prefix,
    ]

    print(f"\nConverting PLINK to correct format...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ PLINK2 conversion complete")
    except subprocess.CalledProcessError as e:
        print(f"PLINK2 command failed with exit code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise RuntimeError(f"PLINK2 preprocessing failed: {e}")

    # Verify output files exist
    for ext in [".bed", ".bim", ".fam"]:
        output_file = f"{output_prefix}{ext}"
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"PLINK2 did not create expected output: {output_file}")

    print(f"✓ Created preprocessed PLINK files: {output_prefix}{{.bed,.bim,.fam}}")
    print("  (This file can be reused for future inference)")

    return output_prefix


def find_plink_binary() -> str:
    """
    Find PLINK2 binary in the following order:
    1. bin/plink2 in current directory
    2. plink2 in PATH
    3. Raise error if not found

    Returns:
        Path to PLINK2 binary
    """
    # Check local bin/ directory
    local_plink2 = Path("bin/plink2")
    if local_plink2.exists():
        return str(local_plink2.absolute())

    # Check PATH
    import shutil

    plink2_path = shutil.which("plink2")
    if plink2_path:
        return plink2_path

    raise FileNotFoundError(
        "PLINK2 not found! Please either:\n"
        "  1. Run setup.sh to download PLINK2 to bin/\n"
        "  2. Install PLINK2 and add it to your PATH"
    )
