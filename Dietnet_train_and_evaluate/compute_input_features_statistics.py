import argparse
import os
import multiprocessing as mp
import time
from datetime import datetime

import numpy as np

import h5py


def main():
    start_time = time.time()
    args = parse_args()

    # Flag to compute only mean (True or False)
    mean_only = args.mean_only

    # Load indices of every fold
    partition_data = np.load(args.partition, allow_pickle=True)
    indices_byfold = partition_data['folds_indexes']

    # Dataset
    dataset_file = h5py.File(args.dataset, 'r')

    # Compute inp feat stats by fold
    means_by_fold = []
    sds_by_fold = []
    for fold in range(len(indices_byfold)):
        print('\n---\nComputing input features statistic of fold', str(fold))
        fold_start_time = time.time()

        # Indices of the fold
        fold_indices = indices_byfold[fold]

        # Genotypes of training samples
        train_indices = np.sort(fold_indices[0]) # sort is a hdf5 requirement
        x_train = dataset_file['inputs'][train_indices]

        # Compute statistics of every features in parallel
        ncpus = args.ncpus
        print('Computing input features statistics of {} SNPs for {} '
              'samples using {} cpu(s)'.format(
              x_train.shape[1], x_train.shape[0], ncpus))

        pool = mp.Pool(processes=ncpus) #this starts ncpus nb of processes

        # Results that will be returned by processes
        results = []

        # The statistics of every snp feature is computed by a different process
        nb_snps = x_train.shape[1]
        for i in range(nb_snps):
            # Compute snps mean
            if mean_only :
                results.append(
                        pool.apply_async(
                            compute_feature_mean, args=(i, x_train[:,i])))

            # Compute mean and standard deviation (sd)
            else:
                results.append(
                        pool.apply_async(
                            compute_feature_mean_and_sd, args=(i, x_train[:,i])))

        # Get order (snp nb) and mean of features from multiprocess results
        print('Computed statistics of all input features, getting the results')
        results_values = np.array([np.array(p.get()) for p in results])

        # Order stats based on snp nb
        print('Ordering {} means returned by multiprocessing'.format(
            len(results_values[:,1])))
        idx = results_values[:,0].astype(int)
        ordered_idx_pos = idx.argsort()

        # Ordered means
        means = results_values[:,1][ordered_idx_pos]

        # Ordered sd
        if not mean_only:
            print('Ordering {} standard deviations returned by multiprocessing'.format(
                len(results_values[:,2])))
            sds = results_values[:,2][ordered_idx_pos]

        # Append fold statistics
        means_by_fold.append(means)
        sds_by_fold.append(sds)

        print('Computed fold input features stats in {} seconds\n---\n'.format(
            time.time()-fold_start_time))

    # Close access to h5py dataset file
    dataset_file.close()

    # Write input features stats to file
    if args.out is not None:
        stats_filename = args.out
    else:
        stats_filename = 'input_features_stats_' \
                + args.dataset.split('/')[-1][0:-5] \
                + '_' \
                + datetime.now().strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    stats_fullpath = os.path.join(args.exp_path, stats_filename)

    if mean_only:
        np.savez(stats_fullpath,
                 means_by_fold=means_by_fold)

    else:
        np.savez(stats_fullpath,
                 means_by_fold=means_by_fold,
                 sd_by_fold=sds_by_fold)

    print('\n---')
    print('Input features stats were saved to {}'.format(stats_fullpath))
    print('Execution time {} seconds\n---\n'.format(
          time.time() - start_time))


def compute_feature_mean(i, x):
    """
    i is the snp nb (from 0 to nb_snps-1)
    x is the vector of genotypes for the snp (dim : nb_samples x 1)
    """
    # Set to false missing value (missing values are encoded with -1)
    mask = (x >= 0)

    # Compute mean :
    # in numerator, missing values are 0
    # in the denominator we divise by the nb of non missing values
    mean = np.sum(x*mask) / np.sum(mask)

    return (i,mean)


def compute_feature_mean_and_sd(i, x):
    """
    i is the snp nb (from 0 to nb_snps-1)
    x is the vector of genotypes for the snp (dim : nb_samples x 1)
    """
    # Compute mean
    _,mean = compute_feature_mean(i,x)

    # --- Compute sd ---
    mask = (x >= 0) #set missing values (-1) to False

    numerator = np.sum((x*mask - mean*mask)**2)

    denominator = np.sum(mask) - 1

    sd = np.sqrt(numerator/denominator)

    sd += 1e-6

    return (i,mean,sd)


def parse_args():
    parser = argparse.ArgumentParser(
            description=('Compute features means and standard deviations '
                         'for missing values filing and input normalization '
                         'at training time')
            )

    # Path
    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help=('Path to directory where input features statistics will '
                  'be written')
            )

    # Files
    parser.add_argument(
            '--dataset',
            type=str,
            required=True,
            help=('Hdf5 dataset created with create_dataset.py. '
                  'Provide full path')
            )

    parser.add_argument(
            '--partition',
            type=str,
            required=True,
            help=('Npz dataset partition returned by partition_data.py '
                  'Provide full path')
            )

    # Which statistic to compute (mean or mean+sd)
    parser.add_argument(
            '--mean-only',
            action='store_true',
            help=('Use this flag to compute only input features means '
                  'and not the standard deviations')
            )

    parser.add_argument(
            '--ncpus',
            type=int,
            default=1,
            help=('Number of cpus for parallel computation of means '
                  'and of standard deviations. Default: %(default)i')
            )

    # Output
    parser.add_argument(
            '--out',
            type=str,
            help=('Optional filename for input features statistics file. '
                  'If not provided the file will be named '
                  'input_features_stats_datasetFilename_date')
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
