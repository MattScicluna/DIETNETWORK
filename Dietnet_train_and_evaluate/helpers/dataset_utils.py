import math

import numpy as np

import h5py

import torch

class FoldDataset(torch.utils.data.Dataset):
    # These variables are set in train.py
    dataset_file = None #path to h5py dataset file
    f = None # Dataset file handler (h5py.File in reading mode)
    task_handler = None # Classification or regression
    data_x = None
    data_y = None
    data_samples = None

    def __init__(self, set_indexes):
        self.set_indexes = set_indexes

    def __len__(self):
        return len(self.set_indexes)

    def __getitem__(self, index):
        # dataloader index goes from 0 to length of set
        # data_x, data_y, data_samples groups data from all sets (train + valid + test)
        # to get the item corresponding to dataloader index, we get the index of the sample
        # in the whole dataset
        file_index = self.set_indexes[index]

        # Input features
        #x = np.array(self.f['inputs'][file_index], dtype=np.int8)
        x = self.data_x[file_index]


        # Label
        #y = self.task_handler.get_label(self.f, file_index)
        y = self.data_y[file_index]
        """
        if self.task == 'classification':
            #y = (self.f['class_labels'][file_index]).astype(np.int64)
            y = np.array(self.f['class_labels'][file_index],
                         dtype=np.int64)
        elif self.task == 'regression':
            #y = (self.f['regression_labels'][file_index]).astype(np.float32)
            y = np.array(self.f['regression_labels'][file_index],
                         dtype=np.float32)
        """
        # Sample id
        #sample = (self.f['samples'][file_index]).astype(np.str_)
        # SAMPLES ARE INDEX in dataset file (hdf5)
        sample = file_index


        return x, y, sample

    def get_samples(self):
        indexes = np.sort(self.set_indexes)
        samples = (self.f['samples'][indexes]).astype(np.str_)

        return samples

class FoldDataset_forBigdataset(torch.utils.data.Dataset):
    # These variables are set in train.py
    dataset_file = None #path to h5py dataset file
    f = None # Dataset file handler (h5py.File in reading mode)
    task_handler = None # Classification or regression

    def __init__(self, set_indexes):
        self.set_indexes = set_indexes

    def __len__(self):
        return len(self.set_indexes)

    def __getitem__(self, index):
        # Data of all sets (train, valid, test) is in one file
        # so we convert the index to match file index
        file_index = self.set_indexes[index]

        # Input features
        x = np.array(self.f['inputs'][file_index], dtype=np.int8)

        # Label
        y = self.task_handler.get_label(self.f, file_index)
        """
        if self.task == 'classification':
            #y = (self.f['class_labels'][file_index]).astype(np.int64)
            y = np.array(self.f['class_labels'][file_index],
                         dtype=np.int64)
        elif self.task == 'regression':
            #y = (self.f['regression_labels'][file_index]).astype(np.float32)
            y = np.array(self.f['regression_labels'][file_index],
                         dtype=np.float32)
        """
        # Sample id
        #sample = (self.f['samples'][file_index]).astype(np.str_)
        # SAMPLES ARE INDEX
        sample = np.array(index)


        return x, y, sample

    def get_samples(self):
        indexes = np.sort(self.set_indexes)
        samples = (self.f['samples'][indexes]).astype(np.str_)

        return samples


class IndepTestDataset(torch.utils.data.Dataset):
    
    # These variables are set in test_independent_dataset.py
    dataset_file = None # path to h5py dataset file
    f = None # Dataset file handler (h5py.File in reading mode)
    task_handler = None # Classification or regression
    data_x = None

    def __init__(self, samples):
        self.set_indexes = np.arange(len(samples))

    def __len__(self):
        return len(self.set_indexes)

    def __getitem__(self, index):
        # Input features
        #x = np.array(self.f['inputs'][file_index], dtype=np.int8)
        x = self.data_x[index]

        return index,x



def get_fold_data(which_fold, folds_indexes, data, label='labels'):
    # Indices of each set for the fold (0:train, 1:valid, 2:test)
    fold_indexes = folds_indexes[which_fold]
    train_indexes = np.sort(fold_indexes[0]) # sort is a hdf5 requirement
    valid_indexes = np.sort(fold_indexes[1])
    test_indexes = np.sort(fold_indexes[2])

    # Get data (x,y,samples) of each set (train, valid, test)
    x_train = data['inputs'][train_indexes]
    y_train = data[label][train_indexes]
    samples_train = data['samples'][train_indexes]

    x_valid = data['inputs'][valid_indexes]
    y_valid = data[label][valid_indexes]
    samples_valid = data['samples'][valid_indexes]

    x_test = data['inputs'][test_indexes]
    y_test = data[label][test_indexes]
    samples_test = data['samples'][test_indexes]

    return train_indexes, valid_indexes, test_indexes,\
           x_train, y_train, samples_train,\
           x_valid, y_valid, samples_valid,\
           x_test, y_test, samples_test


def compute_norm_values(x):
    """
    x is a tensor
    """
    # Non missing values
    mask = (x >= 0)

    # Compute mean of every column (feature)
    with torch.no_grad():
        per_feature_mean = torch.sum(x*mask, dim=0) / torch.sum(mask, dim=0)

        print('Computed per feature mean')

        # S.d. of every column (feature)
        per_feature_sd = torch.sqrt(
                torch.sum((x*mask-mask*per_feature_mean)**2, dim=0) / \
                        (torch.sum(mask, dim=0) - 1)
                        )
        per_feature_sd += 1e-6

        print('Computed per feature sd')

    return per_feature_mean, per_feature_sd


def replace_missing_values(x, per_feature_mean):
    """
    x and per_feature_mean are tensors
    """
    mask = (x >= 0)

    for i in range(x.shape[0]):
        x[i] =  mask[i]*x[i] + (~mask[i])*per_feature_mean


def normalize(x, per_feature_mean, per_feature_sd):
    """
    x, per_feature_mean and per_feature_sd are tensors
    """
    x_norm = (x - per_feature_mean) / per_feature_sd

    return x_norm
