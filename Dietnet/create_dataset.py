"""
Script to parse and save data into a hdf5 file format
Creates dataset.hdf5 (default filename)
"""

import argparse
import multiprocessing as mp
import os
import time

import h5py
import numpy as np

from Dietnet.helpers import dataset_utils as du


def create_dataset():
    args = parse_args()
    create_dataset_with_args(args)


def create_dataset_with_args(args):
    start_time = time.time()

    # ----------------------------
    # Load samples and genotypes
    # ----------------------------
    genotype_start_time = time.time()
    if args.parallel_loading:
        # Multiprocessing to load lines in parallel
        ncpus = args.ncpus
        print("Loading genotypes in parallel using", ncpus, "processes")
        pool = mp.Pool(processes=ncpus)  # this starts ncpus nb of processes

        # Results returned by processes
        results = []

        # Every line is read by a different process
        with open(args.genotypes, "r") as f:
            header_line = next(f)
            for line in f:
                # Add sample, sample's genotype accros all SNPs
                results.append(pool.apply_async(du.load_genotypes_parallel, args=(line,)))

        # Get samples and genotypes from multiprocess results
        print("Parsed genotypes, getting the results")
        results_values = [p.get() for p in results]
        samples = np.array([i[0] for i in results_values])
        genotypes = np.array([i[1] for i in results_values])
        print("Loaded", genotypes.shape[0], "samples with", genotypes.shape[1], "genotypes")

        # Get snps names from header line
        print("Loading snps names")
        snps = np.array([i.strip() for i in header_line.split("\t")[1:]], dtype="S")
        print("Loaded", len(snps), "snps")

    else:
        print("Loading data")
        # Load samples, snp names and genotype values
        samples, snps, genotypes = du.load_genotypes(args.genotypes)

    genotype_end_time = time.time()
    print("Genotypes loading time:", genotype_end_time - genotype_start_time)

    # ----------------------------
    # Load samples and labels
    # ----------------------------
    label_start_time = time.time()
    print("\nLoading labels")
    samples_in_labels, labels = du.load_labels(args.labels)

    # Order labels to match samples order obtained from genotypes file
    print("Ordering labels to match genotypes order, using samples ids")
    ordered_labels = du.order_labels(samples, samples_in_labels, labels)

    # If labels are categories, encode labels as numbers
    if args.task == "classification":
        label_names, encoded_labels = numeric_encode_labels(ordered_labels)
    # If labels are for regression task
    elif args.task == "regression":
        # Convert string to float (this is label for regression task)
        ordered_labels = ordered_labels.astype("float64")

        # Get class labels for embedding computation
        print("Loading class labels")
        samples_in_class_labels, class_labels = du.load_labels(args.class_labels)
        print("Ordering class labels to match genotypes order, using samples ids")
        ordered_class_labels = du.order_labels(samples, samples_in_class_labels, class_labels)
        # Encode labels as numbers
        class_label_names, encoded_class_labels = numeric_encode_labels(ordered_class_labels)

    label_end_time = time.time()
    print("Labels loading time:", label_end_time - label_start_time)

    # ----------------------------
    # Save dataset to file
    # ----------------------------
    print("\nSaving dataset to", os.path.join(args.exp_path, args.out))
    f = h5py.File(os.path.join(args.exp_path, args.out), "w")
    # Input features
    f.create_dataset("inputs", data=genotypes)
    snps = snps.astype("S")  # hdf5 doesn't support np UTF-8 encoding
    f.create_dataset("snp_names", data=snps)
    # Samples
    samples = samples.astype("S")
    f.create_dataset("samples", data=samples)
    # Labels
    if args.task == "classification":
        f.create_dataset("labels", data=encoded_labels)
        label_names = label_names.astype("S")
        f.create_dataset("label_names", data=label_names)

    elif args.task == "regression":
        f.create_dataset("labels", data=ordered_labels)
        # Labels for embedding computation
        f.create_dataset("class_labels", data=encoded_class_labels)
        class_label_names = class_label_names.astype("S")
        f.create_dataset("class_label_names", data=class_label_names)

    f.close()

    end_time = time.time()
    print("\nEnd of execution. Execution time:", end_time - start_time)


def onehot_encode_labels(labels):
    label_names = np.sort(np.unique(labels))

    encoded_labels = np.zeros((len(labels), len(label_names)))
    for i, label in enumerate(labels):
        encoded_labels[i, np.where(label_names == label)[0][0]] = 1.0

    return label_names, encoded_labels


def numeric_encode_labels(labels):
    label_names = np.sort(np.unique(labels))

    encoded_labels = [np.where(label_names == i)[0][0] for i in labels]

    return label_names, encoded_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create hdf5 dataset from genotype and label files."
    )

    parser.add_argument(
        "--exp-path", type=str, required=True, help="Path to directory where dataset will be saved"
    )

    parser.add_argument(
        "--genotypes",
        type=str,
        required=True,
        help=(
            "File of genotypes (additive-encoding) in tab-separated "
            "format. Each line contains a sample id followed "
            "by its genotypes for every SNP. "
            "Missing genotypes can be encoded NA, ./. or -1 "
        ),
    )

    parser.add_argument(
        "--parallel-loading", action="store_true", help="Use this flag to load samples in parallel."
    )

    parser.add_argument("--ncpus", type=int, help="Number of cpus for parallel loading")

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help=(
            "File of samples labels. Each line contains a sample "
            "id followed by its label in tab-separated format."
        ),
    )

    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help=(
            "Task that determines labels encoding "
            "Classification: Labels are numerically encoded "
            "(one number per category). "
            "Regression: Labels are float. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--class-labels",
        help=(
            "Class labels if embedding is computed by class "
            "but the task is regression (--labels are not classes) "
            "Like the --labels, this must contain one column "
            "with individuals ids and the second column the label"
        ),
    )

    parser.add_argument(
        "--out",
        default="dataset.hdf5",
        help="Filename for the returned dataset. Default: %(default)s",
    )

    return parser.parse_args()


if __name__ == "__main__":
    create_dataset()
