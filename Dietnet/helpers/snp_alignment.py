"""
SNP alignment utilities for DietNetwork inference.

Handles aligning test dataset SNPs to model training SNPs by matching
chromosome:position information, dealing with missing SNPs, and creating
efficient index mappings for batch processing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyplink import PyPlink


def parse_bim_file(bim_path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse a PLINK .bim file and return SNP information.

    Args:
        bim_path: Path to .bim file

    Returns:
        DataFrame with columns: chr, snp_id, cm, pos, a1, a2
        Index is 'chr:pos' for easy matching
    """
    bim_path = Path(bim_path)

    # Read BIM file (no header)
    # Columns: chr, snp_id, cm (centimorgan), pos, a1, a2
    bim = pd.read_csv(
        bim_path,
        sep="\t",
        header=None,
        names=["chr", "snp_id", "cm", "pos", "a1", "a2"],
        dtype={"chr": str, "snp_id": str, "cm": float, "pos": int, "a1": str, "a2": str},
    )

    # Create chr:pos identifier for matching
    bim["chr_pos"] = bim["chr"].astype(str) + ":" + bim["pos"].astype(str)

    # Set index for fast lookup
    bim = bim.set_index("chr_pos", drop=False)

    return bim


def create_snp_mapping(
    test_bim: Union[str, Path, pd.DataFrame],
    model_snps: Union[List[str], np.ndarray, pd.Index],
    model_bim: Optional[Union[str, Path, pd.DataFrame]] = None,
    fill_value: int = -1,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Create SNP index mapping from test dataset to model training SNPs.

    This handles the case where test and training datasets have different
    SNP sets. For each model SNP, we find the corresponding test SNP index
    or use fill_value if not present.

    Args:
        test_bim: Path to test .bim file or DataFrame
        model_snps: List of model SNP identifiers (chr:pos format or rsIDs)
        model_bim: Optional path to model .bim file or DataFrame (if model_snps are rsIDs)
        fill_value: Value to use for missing SNPs in test data (default: -1)

    Returns:
        Tuple of:
            - snp_indices: Array of shape (n_model_snps,) with test dataset indices
                          (or -1 for missing SNPs)
            - alignment_info: Dict with alignment statistics
    """
    # Parse test BIM file if needed
    if isinstance(test_bim, (str, Path)):
        test_bim_df = parse_bim_file(test_bim)
    else:
        test_bim_df = test_bim

    # Convert model_snps to list if needed
    if isinstance(model_snps, (pd.Index, pd.Series)):
        model_snps = model_snps.tolist()
    elif isinstance(model_snps, np.ndarray):
        model_snps = model_snps.tolist()

    # If model_snps are rsIDs and we have model_bim, convert to chr:pos
    if model_bim is not None:
        if isinstance(model_bim, (str, Path)):
            model_bim_df = parse_bim_file(model_bim)
        else:
            model_bim_df = model_bim

        # Try to match by rsID first
        # Assume model_snps are rsIDs if they don't contain ':'
        if ":" not in str(model_snps[0]):
            # Create mapping from rsID to chr:pos
            rsid_to_chrpos = dict(zip(model_bim_df["snp_id"], model_bim_df["chr_pos"]))
            model_snps_chrpos = [rsid_to_chrpos.get(snp, snp) for snp in model_snps]
        else:
            model_snps_chrpos = model_snps
    else:
        model_snps_chrpos = model_snps

    # Create mapping: for each model SNP, find index in test dataset
    snp_indices = []
    matched_snps = 0
    missing_snps = 0

    # Create reverse mapping from chr:pos to test index
    test_chrpos_to_idx = {chrpos: idx for idx, chrpos in enumerate(test_bim_df["chr_pos"])}

    # Also try matching by rsID if chr:pos fails
    test_rsid_to_idx = {rsid: idx for idx, rsid in enumerate(test_bim_df["snp_id"])}

    for model_snp in model_snps_chrpos:
        # Try matching by chr:pos first
        if model_snp in test_chrpos_to_idx:
            snp_indices.append(test_chrpos_to_idx[model_snp])
            matched_snps += 1
        # Try matching by original SNP ID (if it's an rsID)
        elif model_snp in test_rsid_to_idx:
            snp_indices.append(test_rsid_to_idx[model_snp])
            matched_snps += 1
        else:
            # SNP not found in test dataset
            snp_indices.append(fill_value)
            missing_snps += 1

    snp_indices = np.array(snp_indices, dtype=np.int32)

    # Calculate alignment statistics
    alignment_info = {
        "n_model_snps": len(model_snps),
        "n_test_snps": len(test_bim_df),
        "n_matched": matched_snps,
        "n_missing": missing_snps,
        "overlap_fraction": matched_snps / len(model_snps) if len(model_snps) > 0 else 0,
        "missing_indices": np.where(snp_indices == fill_value)[0].tolist(),
    }

    return snp_indices, alignment_info


def align_genotypes(
    test_genotypes: np.ndarray, snp_mapping: np.ndarray, fill_value: int = -1
) -> np.ndarray:
    """
    Align test genotypes to model SNP order.

    Args:
        test_genotypes: Test genotype array of shape (n_samples, n_test_snps)
        snp_mapping: SNP index mapping from create_snp_mapping()
        fill_value: Value to use for missing SNPs

    Returns:
        Aligned genotypes of shape (n_samples, n_model_snps)
    """
    n_samples = test_genotypes.shape[0]
    n_model_snps = len(snp_mapping)

    # Create output array
    aligned = np.full((n_samples, n_model_snps), fill_value, dtype=test_genotypes.dtype)

    # Fill in matched SNPs
    valid_mask = snp_mapping >= 0
    valid_indices = snp_mapping[valid_mask]

    aligned[:, valid_mask] = test_genotypes[:, valid_indices]

    return aligned


def check_alignment_quality(
    alignment_info: Dict,
    min_overlap: float = 0.5,
    warn_overlap: float = 0.8,
    raise_on_poor: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Check alignment quality and issue warnings/errors as appropriate.

    Args:
        alignment_info: Dict from create_snp_mapping()
        min_overlap: Minimum acceptable overlap fraction (default: 0.5)
        warn_overlap: Overlap fraction below which to warn (default: 0.8)
        raise_on_poor: Whether to raise error on poor overlap (default: False)

    Returns:
        True if alignment is acceptable, False otherwise

    Raises:
        ValueError: If overlap is below min_overlap and raise_on_poor is True
    """
    overlap = alignment_info["overlap_fraction"]
    n_matched = alignment_info["n_matched"]
    n_total = alignment_info["n_model_snps"]

    if verbose:
        print(f"\nSNP Alignment Summary:")
        print(f"  Model SNPs: {n_total}")
        print(f"  Test SNPs: {alignment_info['n_test_snps']}")
        print(f"  Matched: {n_matched} ({overlap:.1%})")
        print(f"  Missing: {alignment_info['n_missing']} ({1-overlap:.1%})")

    if overlap < min_overlap:
        msg = (
            f"Poor SNP overlap: {overlap:.1%} < {min_overlap:.1%}. "
            f"Consider retraining model on dataset with better SNP coverage."
        )

        if raise_on_poor:
            raise ValueError(msg)
        else:
            if verbose:
                print(f"\n⚠️  WARNING: {msg}")
                print("Inference will continue, but predictions may be unreliable.")
            return False

    elif overlap < warn_overlap:
        if verbose:
            print(f"\n⚠️  Note: SNP overlap is {overlap:.1%}, which is below {warn_overlap:.1%}.")
            print("Missing SNPs will be imputed with training means.")

    else:
        if verbose:
            print(f"\n✓ Good SNP overlap: {overlap:.1%}")

    return True


def get_snp_list_from_plink(plink_prefix: Union[str, Path]) -> List[str]:
    """
    Extract SNP list (chr:pos format) from a PLINK .bim file.

    Args:
        plink_prefix: PLINK file prefix (without .bed/.bim/.fam)

    Returns:
        List of SNP identifiers in 'chr:pos' format
    """
    bim_path = Path(str(plink_prefix) + ".bim")
    bim_df = parse_bim_file(bim_path)
    return bim_df["chr_pos"].tolist()


def create_snp_file(
    plink_prefix: Union[str, Path], output_file: Union[str, Path], format: str = "chr:pos"
):
    """
    Create a SNP list file from PLINK .bim file.

    Args:
        plink_prefix: PLINK file prefix
        output_file: Where to save SNP list
        format: 'chr:pos' or 'rsid'
    """
    bim_path = Path(str(plink_prefix) + ".bim")
    bim_df = parse_bim_file(bim_path)

    with open(output_file, "w") as f:
        if format == "chr:pos":
            for chrpos in bim_df["chr_pos"]:
                f.write(f"{chrpos}\n")
        elif format == "rsid":
            for rsid in bim_df["snp_id"]:
                f.write(f"{rsid}\n")
        else:
            raise ValueError(f"Unknown format: {format}. Use 'chr:pos' or 'rsid'")

    print(f"Saved {len(bim_df)} SNPs to {output_file} (format: {format})")
