import argparse
import os
import time
from datetime import datetime
import multiprocessing as mp

import numpy as np

import h5py

import helpers.dataset_utils as du


# Additive encoding : genotypes 0, 1 and 2
NB_POSSIBLE_GENOTYPES = 3

# Class label key in dataset file
CLASS_LABEL = 'class_labels'


def main():
    start_time = time.time()
    args = parse_args()

    # Hdf5 dataset
    dataset_file = h5py.File(args.dataset, 'r')

    # Indices by fold
    partition_data = np.load(args.partition, allow_pickle=True)
    indices_byfold = partition_data['folds_indexes']

    print('\n---')
    print('Computing embedding by fold using {} cpu(s)'.format(args.ncpus))
    if args.ncpus > 1:
        pool = mp.Pool(processes=args.ncpus) #this starts ncpus nb of processes

    # Results that will be returned by processes (embedding for a given fold)
    results = []

    for fold in range(len(indices_byfold)):
        # Fold indices
        fold_indices = indices_byfold[fold]
        # Get fold data
        (_,_,_,
         x_train, y_train, _,
         x_valid, y_valid, _,
         x_test, y_test,_) = du.get_fold_data(fold, indices_byfold, dataset_file, label=CLASS_LABEL)

        # Embedding on train+valid sets
        if args.include_valid:
            print('Including valid samples in emb computation')
            x = np.concatenate((x_train, x_valid))
            y = np.concatenate((y_train, y_valid))
        # Embedding on valid set
        elif args.only_valid:
            x = x_valid
            y = y_valid
        # Embedding on test set
        elif args.only_test:
            x = x_test
            y = y_test
        # Embedding on training set
        else:
            x = x_train
            y = y_train

        # Compute embedding
        #emb = compute_fold_embedding(x, y)

        print('Computing embedding using {} samples'.format(len(y)))
        # One cpu
        if args.ncpus == 1:
            results.append(compute_fold_embedding(x, y, fold))
        else:
            results.append(pool.apply_async(
                compute_fold_embedding, args=(x,y,fold)))

    # Close access to dataset file
    dataset_file.close()

    # Results values: [(fold0, embedding0), (fold1, embedding1) ...]
    print('Computed embedding of every fold, getting the results')
    if args.ncpus > 1:
        results_values = [p.get() for p in results]
    else:
        results_values = results

    # Order results by fold
    print('Putting embeddings by fold order')
    folds_order = np.array([i[0] for i in results_values])

    # Order counts (nb samples) per class
    classes_counts = np.array([i[1] for i in results_values])
    classes_counts_ordered = classes_counts[folds_order]

    # Order embeddings
    embeddings = np.array([i[2] for i in results_values])
    embeddings_ordered = embeddings[folds_order]

    print('Computed embedding of {} folds with following '
            'number of samples per class for each fold: {}'.format(
              len(folds_order), classes_counts_ordered))
    print('---')

    # Write embedding to file
    if args.out is not None:
        emb_filename = args.out
    else:
        emb_filename = 'genotype_class_freq_embedding_' \
                + args.dataset.split('/')[-1][0:-5] \
                + '_' \
                + datetime.now().strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    emb_fullpath = os.path.join(args.exp_path, emb_filename)

    print('\n---\nEmbeddings generated in {} seconds'.format(
        time.time()-start_time))

    print('Saving results to {}'.format(emb_fullpath))
    np.savez(emb_fullpath, emb=embeddings_ordered)

    print('---\n')


def compute_fold_embedding(xs, ys, fold):
    # Total number of classes
    nb_class = ys.max() + 1 #class 0
    class_to_count = {}

    # Compute sum of genotypes (0-1-2) per class
    xs = xs.transpose() # rows are snps, col are inds
    embedding = np.zeros((xs.shape[0],nb_class*NB_POSSIBLE_GENOTYPES))
    for c in range(nb_class):
        # Select genotypes for samples of same class
        class_genotypes = xs[:,ys==c]
        nb = class_genotypes.shape[1] #nb of samples in that class
        #print('Class:', c, 'NB:', nb)
        class_to_count[c] = nb
        for genotype in range(NB_POSSIBLE_GENOTYPES):
            col = NB_POSSIBLE_GENOTYPES*c+genotype
            embedding[:,col] = (class_genotypes == genotype).sum(axis=1)/nb

    return fold, class_to_count, embedding



def parse_args():
    parser = argparse.ArgumentParser(
            description='Generate embedding'
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory where embedding will be saved. '
            )

    parser.add_argument(
            '--dataset',
            type=str,
            required=True,
            help=('Hdf5 dataset created with create_dataset.py '
                  'Provide full path')
            )

    parser.add_argument(
            '--partition',
            type=str,
            required=True,
            help=('Npz dataset partition returned by partition_data.py '
                  'Provide full path')
            )

    parser.add_argument(
            '--ncpus',
            type=int,
            default=1,
            help=('Number of cpus available to compute folds embedding in '
                  'parallel. Default:%(default)i')
            )

    parser.add_argument(
            '--include-valid',
            action='store_true',
            help=('Use this flag to include samples from the validation set '
                  'in the embedding computation.')
            )

    parser.add_argument(
            '--only-valid',
            action='store_true',
            help=('Use this flag to compute embedding only on samples '
                  ' in the validation set')
            )

    parser.add_argument(
            '--only-test',
            action='store_true',
            help=('Use this flag to compute embedding only on samples '
                  ' in the test set')
            )

    parser.add_argument(
            '--out',
            type=str,
            help=('Optional filename for embedding file. If not provided '
                  'the file will be named '
                  'genotype_class_freq_embedding_datasetFilename_date')
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
