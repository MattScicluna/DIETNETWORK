import argparse
import os
import time
from pathlib import PurePath

import h5py
import numpy as np
import torch

from Dietnet.helpers import dataset_utils as du


def _resolve_path(base_dir: str, path: str) -> str:
    """Return absolute path, respecting already-absolute inputs."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def get_preprocessing_params(args=None):
    """
    Compute per-fold feature statistics for normalization.

    Args:
        args: argparse.Namespace-like object with exp_path, dataset, partition,
              parallel_loading, ncpus, and out attributes. If None, CLI args
              are parsed.
    """
    if args is None:
        args = parse_args()

    start_time = time.time()

    # Load partition indexes
    partition_path = _resolve_path(args.exp_path, args.partition)
    folds_indexes = du.load_folds_indexes(partition_path)

    # Detect dataset type
    dataset_path = _resolve_path(args.exp_path, args.dataset)

    if dataset_path.endswith(".hdf5") or dataset_path.endswith(".h5"):
        # HDF5 mode
        print("Using HDF5 dataset")
        data = h5py.File(dataset_path)

        means_by_fold = []
        sd_by_fold = []
        for fold in range(len(folds_indexes)):
            print("Computing input features statistics of fold", fold)
            train_indexes = np.sort(folds_indexes[fold][0])  # sort is an h5py requirement
            x_train = torch.from_numpy(data["inputs"][train_indexes].astype(np.float32))
            mean, sd = du.compute_norm_values(x_train)
            means_by_fold.append(mean.numpy())
            sd_by_fold.append(sd.numpy())

        data.close()

    else:
        # PLINK mode
        print("Using PLINK dataset")
        from tqdm import tqdm

        from Dietnet.helpers.dataset_utils import load_plink_genotypes

        plink_prefix = dataset_path.replace(".bed", "")
        cache_file = PurePath(
            args.exp_path, PurePath(dataset_path).name.replace(".bed", "_genotypes.npy")
        )

        # Load all genotypes
        genotypes, fam_data, bim_data = load_plink_genotypes(plink_prefix, cache_file)

        means_by_fold = []
        sd_by_fold = []
        for fold in tqdm(range(len(folds_indexes)), desc="Computing fold statistics", unit="fold"):
            train_indices = folds_indexes[fold][0]
            x_train = torch.from_numpy(genotypes[train_indices].astype(np.float32))
            mean, sd = du.compute_norm_values(x_train)
            means_by_fold.append(mean.numpy())
            sd_by_fold.append(sd.numpy())

    out_path = _resolve_path(args.exp_path, args.out)
    print("Saving input features stats to", out_path)
    np.savez(out_path, means_by_fold=means_by_fold, sd_by_fold=sd_by_fold)
    print("Execution time: {} seconds".format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-fold feature means and standard deviations "
        "for missing value imputation and input normalization."
    )

    parser.add_argument("--exp-path", type=str, required=True, help="Path to experiment directory.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset.hdf5",
        help="Dataset filename (default: %(default)s).",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="partitioned_idx.npz",
        help="Partition filename (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="input_features_means.npz",
        help="Output filename (default: %(default)s).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    get_preprocessing_params()
