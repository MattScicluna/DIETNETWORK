import argparse
import os
import time
from pathlib import PurePath

import h5py
import numpy as np

from Dietnet.helpers import dataset_utils as du

NB_POSSIBLE_GENOTYPES = 3


def generate_embedding():
    args = parse_args()
    generate_embedding_with_args(args)


def generate_embedding_with_args(args):
    start_time = time.time()

    # Set label field based on task type
    if hasattr(args, "task"):
        if args.task == "classification":
            args.emb_class_label = "class_labels"
        elif args.task == "regression":
            args.emb_class_label = "regression_labels"

    # Load partition indexes
    folds_indexes = du.load_folds_indexes(os.path.join(args.exp_path, args.partition))

    # Detect dataset type
    if args.dataset.endswith(".hdf5") or args.dataset.endswith(".h5"):
        # HDF5 mode
        print("Using HDF5 dataset")
        data = h5py.File(os.path.join(args.exp_path, args.dataset))

        embedding_by_fold = []
        for fold in range(len(folds_indexes)):
            print("Computing embedding of fold", str(fold))

            # Get fold data (x,y,samples) that are not test data
            _, _, _, x_train, y_train, _, x_valid, y_valid, _, _, _, _ = du.get_fold_data(
                fold, folds_indexes, data, label=args.emb_class_label
            )

            # Embedding on train+valid sets
            if args.include_valid:
                x = np.concatenate((x_train, x_valid))
                y = np.concatenate((y_train, y_valid))

            # Embedding on valid set
            elif args.only_valid:
                print("Emb on valid")
                x = x_valid
                y = y_valid

            # Embedding on training set
            else:
                x = x_train
                y = y_train

            # Compute embedding for the fold
            emb = compute_fold_embedding(x, y)
            embedding_by_fold.append(emb)

        data.close()

    else:
        # PLINK mode - load genotypes and labels
        print("Using PLINK dataset")
        from tqdm import tqdm

        from Dietnet.helpers.dataset_utils import load_plink_genotypes

        plink_prefix = os.path.join(args.exp_path, args.dataset).replace(".bed", "")
        cache_file = PurePath(args.exp_path, args.dataset.replace(".bed", "_genotypes.npy"))

        # Load genotypes
        genotypes, fam_data, bim_data = load_plink_genotypes(plink_prefix, cache_file)

        # Load and order labels
        label_samples, labels = du.load_labels(PurePath(args.exp_path, args.label_file))
        fam_samples = fam_data["iid"].values
        ordered_labels = du.order_labels(fam_samples, label_samples, labels)

        # Convert labels if classification
        if args.task == "classification":
            label_names = np.unique(ordered_labels)
            label_to_idx = {label: idx for idx, label in enumerate(label_names)}
            ordered_labels_idx = np.array([label_to_idx[label] for label in ordered_labels])
        else:
            ordered_labels_idx = ordered_labels.astype(np.float32)

        # Compute embeddings per fold
        embedding_by_fold = []
        for fold in tqdm(range(len(folds_indexes)), desc="Processing folds", unit="fold"):
            # Get fold indexes
            fold_indexes = folds_indexes[fold]
            train_indexes = fold_indexes[0]
            valid_indexes = fold_indexes[1]

            # Get training data
            x_train = genotypes[train_indexes]
            y_train = ordered_labels_idx[train_indexes]

            # Get validation data
            x_valid = genotypes[valid_indexes]
            y_valid = ordered_labels_idx[valid_indexes]

            # Embedding on train+valid sets
            if args.include_valid:
                x = np.concatenate((x_train, x_valid))
                y = np.concatenate((y_train, y_valid))

            # Embedding on valid set
            elif args.only_valid:
                x = x_valid
                y = y_valid

            # Embedding on training set
            else:
                x = x_train
                y = y_train

            # Compute embedding for the fold
            emb = compute_fold_embedding(x, y)
            embedding_by_fold.append(emb)

    # Save
    embedding_by_fold = np.array(embedding_by_fold)
    print("Saving embedding to", os.path.join(args.exp_path, args.out))
    np.savez(os.path.join(args.exp_path, args.out), emb=embedding_by_fold)

    end_time = time.time()
    print("End of execution. Execution time:", end_time - start_time, "seconds")


def compute_fold_embedding(xs, ys):
    import warnings

    from tqdm import tqdm

    # Total number of classes
    nb_class = ys.max() + 1  # class 0

    # Compute sum of genotypes (0-1-2) per class
    xs = xs.transpose()  # rows are snps, col are inds
    embedding = np.zeros((xs.shape[0], nb_class * NB_POSSIBLE_GENOTYPES))
    missing_classes = []
    for c in tqdm(range(nb_class), desc="Computing embedding", unit="class"):
        # Select genotypes for samples of same class
        class_genotypes = xs[:, ys == c]
        nb = class_genotypes.shape[1]  # nb of samples in that class
        if nb == 0:
            missing_classes.append(c)
            continue
        for genotype in range(NB_POSSIBLE_GENOTYPES):
            col = NB_POSSIBLE_GENOTYPES * c + genotype
            embedding[:, col] = (class_genotypes == genotype).sum(axis=1) / nb
    if missing_classes:
        warnings.warn(
            f"No samples for classes {missing_classes} in this fold; corresponding embedding columns left as 0."
        )

    return embedding


def compute_fold_embedding_(xs, onehot_ys):
    ys = onehot_ys.argmax(axis=1)

    # Total number of classes
    nb_class = onehot_ys.shape[1]

    # Compute sum of genotypes (0-1-2) per class
    xs = xs.transpose()  # rows are snps, col are inds
    embedding = np.zeros((xs.shape[0], nb_class * NB_POSSIBLE_GENOTYPES))
    for c in range(nb_class):
        # Select genotypes for samples of same class
        class_genotypes = xs[:, ys == c]
        nb = class_genotypes.shape[1]  # nb of samples in that class
        for genotype in range(NB_POSSIBLE_GENOTYPES):
            col = NB_POSSIBLE_GENOTYPES * c + genotype
            embedding[:, col] = (class_genotypes == genotype).sum(axis=1) / nb

    return embedding


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embedding")

    parser.add_argument(
        "--exp-path",
        type=str,
        required=True,
        help="Path to experiment directory where to save embedding. ",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset.hdf5",
        help=(
            "Filename of dataset returned by create_dataset.py "
            "The file must be in directory specidifed with exp-path. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--emb-class-label",
        default="labels",
        help=(
            "Key name in hdf5 dataset of class labels "
            "to use in embedding computation. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="partitioned_idx.npz",
        help=(
            "Filename of folds indexes returned by create_dataset.py "
            "The file must be in directory specified with exp-path. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--include-valid",
        action="store_true",
        help=(
            "Use this flag if to include samples from validation set "
            "in the embedding computation. Otherwise embedding is "
            "computed using only samples from training set."
        ),
    )

    parser.add_argument(
        "--only-valid", action="store_true", help="Compute embedding on validation set"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="embedding.npz",
        help="Filename for returned embedding. Default: %(default)s",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression"],
        default="classification",
        help="Task type: classification or regression. Default: %(default)s",
    )

    parser.add_argument(
        "--label-file",
        type=str,
        default=None,
        help="Path to label file (TSV format, required for PLINK datasets)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    generate_embedding()
