import argparse
import os

import h5py
import numpy as np

from Dietnet.helpers import dataset_utils as du


def partition_data():
    args = parse_args()
    partition_data_with_args(args)


def partition_data_with_args(args):
    # Load samples
    dataset_file = os.path.join(args.exp_path, args.dataset)

    labels = None

    # Detect dataset type
    if args.dataset.endswith(".hdf5") or args.dataset.endswith(".h5"):
        # HDF5 mode
        with h5py.File(dataset_file, "r") as f:
            nb_samples = len(f["samples"])
            sample_ids = np.array(f["samples"]).astype(str)
            if args.stratify:
                if "class_labels" in f:
                    labels = np.array(f["class_labels"])
                elif "labels" in f:
                    labels = np.array(f["labels"])
                else:
                    raise ValueError("No labels found in dataset for stratified split.")
    else:
        # PLINK mode - just read FAM file (fast, no genotype loading)
        from pyplink import PyPlink

        plink_prefix = dataset_file.replace(".bed", "")
        plink_reader = PyPlink(plink_prefix)
        nb_samples = plink_reader.get_nb_samples()
        fam_data = plink_reader.get_fam()
        fam_samples = fam_data["iid"].values
        sample_ids = fam_samples
        if args.stratify:
            if args.label_file is None:
                raise ValueError("--label-file is required for stratified PLINK partitioning.")
            label_path = os.path.join(args.exp_path, args.label_file)
            label_samples, label_values = du.load_labels(label_path)
            labels = du.order_labels(fam_samples, label_samples, label_values)

    indices = np.arange(nb_samples)

    print("Partitioning indices of", len(indices), "samples")

    # Partition
    partition = du.partition(
        indices, args.nb_folds, args.train_valid_ratio, args.seed, labels=labels
    )

    print("Saving partition to", os.path.join(args.exp_path, args.out))
    np.savez(
        os.path.join(args.exp_path, args.out),
        folds_indexes=np.array(partition, dtype=object),
        seed=np.array([args.seed]),
        sample_ids=sample_ids,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Partition data into folds. This script creates an array "
            "containing samples' indexes of every partition"
        )
    )

    parser.add_argument(
        "--exp-path", type=str, required=True, help="Path to directory where dataset is saved"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset.hdf5",
        help=("Filename of dataset created with create_dataset.py " "Default %(default)s"),
    )
    parser.add_argument(
        "--label-file",
        type=str,
        default=None,
        help="Label TSV (required for stratified PLINK partitioning)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help=(
            "Seed for fixing random shuffle of samples before "
            "partitioning samples. Default:  %(default)i"
        ),
    )

    parser.add_argument(
        "--nb-folds",
        type=int,
        default=5,
        help="Number of folds. Use 1 for no folds. Default: %(default)i",
    )

    parser.add_argument(
        "--train-valid-ratio",
        type=float,
        default=0.75,
        help=(
            "Ratio (between 0-1) for split of train and valid sets. "
            "For example, 0.75 will use 75%% of data for training "
            "and 25%% of data for validation. Default: %(default).2f"
        ),
    )
    parser.add_argument(
        "--stratify/--no-stratify",
        default=False,
        help="Stratify folds by label/population (default: no).",
    )

    parser.add_argument(
        "--out",
        default="partitioned_idx.npz",
        help=("Filename for returned samples indexes of each fold. " "Default: %(default)s"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    partition_data()
