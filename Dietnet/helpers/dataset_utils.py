import math
import os

import h5py
import numpy as np
import torch


class FoldDataset(torch.utils.data.Dataset):
    # These variables are set in train.py
    dataset_file = None  # path to h5py file
    label_type = None  # Int if classification, float if regression
    task = None  # 'classification' or 'regression'

    def __init__(self, set_indexes):
        self.set_indexes = set_indexes

    def __len__(self):
        return len(self.set_indexes)

    def __getitem__(self, index):
        # Data of all sets (train, valid, test) is in one file
        # so we convert the index to match file index
        file_index = self.set_indexes[index]

        # Input features
        x = np.array(self.f["inputs"][file_index], dtype=np.int8)

        # Label - use appropriate field based on task type
        if self.task == "classification":
            # Try class_labels first (new format), fall back to labels
            if "class_labels" in self.f:
                y = (self.f["class_labels"][file_index]).astype(self.label_type)
            else:
                y = (self.f["labels"][file_index]).astype(self.label_type)
        elif self.task == "regression":
            if "regression_labels" in self.f:
                y = (self.f["regression_labels"][file_index]).astype(self.label_type)
            else:
                y = (self.f["labels"][file_index]).astype(self.label_type)
        else:
            # Default behavior for backward compatibility
            y = (self.f["labels"][file_index]).astype(self.label_type)

        sample = (self.f["samples"][file_index]).astype(np.str_)
        return x, y, sample

    def get_samples(self):
        indexes = np.sort(self.set_indexes)
        samples = (self.f["samples"][indexes]).astype(np.str_)

        return samples


class ExternalTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file):
        self.dataset = h5py.File(dataset_file, "r")

    def __len__(self):
        return len(self.dataset["samples"])

    def __getitem__(self, index):
        x = np.array(self.dataset["inputs"][index], dtype=np.int8)
        sample = (self.dataset["samples"][index]).astype(np.str_)

        return x, sample


class PLINKFoldDataset(torch.utils.data.Dataset):
    """Dataset class for PLINK files. Matches FoldDataset interface."""

    # Class variables (set externally like FoldDataset)
    plink_prefix = None  # Path to .bed/.bim/.fam without extension
    label_file = None  # Path to label TSV file
    label_type = None  # int for classification, float for regression
    task = None  # 'classification' or 'regression'
    genotype_cache = None  # Shared across all datasets (train/valid/test)
    fam_data = None  # Shared FAM data
    bim_data = None  # Shared BIM data
    ordered_labels = None  # Shared labels

    def __init__(self, set_indexes):
        """
        Args:
            set_indexes: Array of sample indices for this fold/set
        """
        self.set_indexes = set_indexes

    def __len__(self):
        return len(self.set_indexes)

    def __getitem__(self, index):
        """Returns same format as FoldDataset: (genotypes, label, sample_id)"""
        # Convert dataset index to file index
        file_index = self.set_indexes[index]

        # Get genotype row from shared cache
        x = self.genotype_cache[file_index]

        # Get label
        y = self.ordered_labels[file_index]

        # Get sample ID
        sample = self.fam_data["iid"].values[file_index]

        return x, y, sample

    def get_samples(self):
        """Returns sample IDs for this dataset (matches FoldDataset interface)"""
        indexes = np.sort(self.set_indexes)
        samples = self.fam_data["iid"].values[indexes]
        return samples


def load_plink_genotypes(
    plink_prefix, cache_file=None, use_memmap=True, force_cache_rebuild=False, verbose=True
):
    """
    Load genotypes from PLINK files with optional memory mapping for scalability.

    For large datasets (e.g., 400K samples), uses memory mapping to avoid loading
    entire dataset into RAM. OS handles paging from disk automatically.

    Args:
        plink_prefix: Path to PLINK files without extension
        cache_file: Optional .npy file to cache genotypes
        use_memmap: If True, use memory-mapped array for large datasets (default: True)

    Returns:
        genotypes: ndarray or memmap of shape (n_samples, n_markers) dtype=int8
        fam_data: DataFrame with sample info
        bim_data: DataFrame with marker info
    """
    from pyplink import PyPlink
    from tqdm import tqdm

    # Get metadata first
    pedfile = PyPlink(plink_prefix)
    fam_data = pedfile.get_fam()
    bim_data = pedfile.get_bim()
    n_samples = pedfile.get_nb_samples()
    n_markers = pedfile.get_nb_markers()

    # Calculate dataset size
    dataset_size_gb = (n_samples * n_markers * 1) / (1024**3)  # int8 = 1 byte

    # Check if cache exists
    if cache_file and os.path.exists(cache_file) and not force_cache_rebuild:
        # Decide whether to use memory mapping based on dataset size
        if use_memmap and dataset_size_gb > 2.0:  # Use memmap for datasets > 2GB
            if verbose:
                print(f"✓ Reusing cached genotypes (memmap): {cache_file}")
            genotypes = np.load(cache_file, mmap_mode="r")  # Read-only memory map
        else:
            if verbose:
                print(f"Loading cached genotypes from {cache_file}")
            genotypes = np.load(cache_file)
        if verbose:
            print(f"Loaded cached genotypes: {genotypes.shape}")
        return genotypes, fam_data, bim_data
    elif cache_file and force_cache_rebuild and os.path.exists(cache_file):
        if verbose:
            print(f"Cache exists but force rebuild requested; regenerating: {cache_file}")
        os.remove(cache_file)

    # Load from PLINK files
    if verbose:
        print(f"Reading {n_markers:,} markers for {n_samples:,} samples into cache (one-time)...")
        print(f"Dataset size (int8): {dataset_size_gb:.2f}GB")

    if cache_file and use_memmap and dataset_size_gb > 2.0:
        # Create memory-mapped file for large datasets
        if verbose:
            print("Creating memory-mapped cache file for scalability")
        genotypes = np.lib.format.open_memmap(
            cache_file, mode="w+", dtype=np.int8, shape=(n_samples, n_markers)
        )
    else:
        # Load into RAM for small datasets
        genotypes = np.zeros([n_samples, n_markers], dtype=np.int8)

    # Iterate through markers and fill array with progress bar
    for i, (marker_id, marker_genotypes) in enumerate(
        tqdm(pedfile, total=n_markers, desc="Loading markers", unit="markers", disable=not verbose)
    ):
        genotypes[:, i] = marker_genotypes

    if verbose:
        print(f"✓ Loaded genotypes: {genotypes.shape}")

    # Flush to disk if memory-mapped
    if isinstance(genotypes, np.memmap):
        genotypes.flush()
        if verbose:
            print(f"✓ Flushed to disk: {cache_file}")
        # Reopen as read-only for safety
        genotypes = np.load(cache_file, mmap_mode="r")

    return genotypes, fam_data, bim_data


def shuffle(indices, seed=None):
    # Fix seed so shuffle is always the same
    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(indices)


def partition(indices, nb_folds, train_valid_ratio, seed=None, labels=None):
    """
    Partition indices into train/valid/test for each fold.

    If labels are provided, performs stratified splitting to preserve label
    proportions per fold.
    """
    if labels is not None:
        return _partition_stratified(indices, labels, nb_folds, train_valid_ratio, seed)
    return _partition_unstratified(indices, nb_folds, train_valid_ratio, seed)


def _partition_unstratified(indices, nb_folds, train_valid_ratio, seed=None):
    if seed is not None:
        np.random.seed(seed)
    shuffle(indices, seed=seed)

    step = math.floor(len(indices) / nb_folds)
    split_pos = [i for i in range(0, len(indices), step)]

    test_indices_byfold = []
    start = split_pos[0]  # same as start=0
    for i in range(nb_folds - 1):
        test_indices_byfold.append(indices[start : (start + step)])
        start = split_pos[i + 1]

    test_indices_byfold.append(indices[start:])  # append last fold

    train_indices_byfold = []
    valid_indices_byfold = []
    for i in range(nb_folds):
        other_folds = [f for f in range(nb_folds) if f != i]
        train_valid_indices = np.concatenate([test_indices_byfold[f] for f in other_folds])
        train_indices, valid_indices = split(train_valid_indices, train_valid_ratio, seed)
        train_indices_byfold.append(train_indices)
        valid_indices_byfold.append(valid_indices)

    indices_byfold = []
    for train_indices, valid_indices, test_indices in zip(
        train_indices_byfold, valid_indices_byfold, test_indices_byfold
    ):
        indices_byfold.append([train_indices, valid_indices, test_indices])

    return indices_byfold


def _partition_stratified(indices, labels, nb_folds, train_valid_ratio, seed=None):
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    per_label_splits = {}
    for lab in unique_labels:
        lab_indices = indices[labels == lab]
        lab_indices = lab_indices.copy()
        rng.shuffle(lab_indices)
        per_label_splits[lab] = np.array_split(lab_indices, nb_folds)

    folds = []
    for fold in range(nb_folds):
        train_parts = []
        valid_parts = []
        test_parts = []

        for lab in unique_labels:
            splits = per_label_splits[lab]
            test_part = splits[fold]
            remainder = np.concatenate([splits[i] for i in range(nb_folds) if i != fold])
            rng.shuffle(remainder)

            n_train = int(math.floor(train_valid_ratio * len(remainder)))
            train_parts.append(remainder[:n_train])
            valid_parts.append(remainder[n_train:])
            test_parts.append(test_part)

        train_indices = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
        valid_indices = np.concatenate(valid_parts) if valid_parts else np.array([], dtype=int)
        test_indices = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

        folds.append([train_indices, valid_indices, test_indices])

    return folds


def split(indices, split_ratio, seed):
    # Fix seed so shuffle is always the same
    if seed is not None:
        np.random.seed(seed)

    # Shuffle so that validation set is different between folds
    # np.random.shuffle(indices)

    split_pos = int(len(indices) * split_ratio)

    train_indexes = indices[0:split_pos]
    valid_indexes = indices[split_pos:]

    return train_indexes, valid_indexes


def load_genotypes(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    # SNP ids
    snps = np.array([i.strip() for i in lines[0].split("\t")[1:]])

    # Sample ids
    samples = np.array([i.split("\t")[0] for i in lines[1:]])

    # Genotypes
    genotypes = np.empty((len(samples), len(snps)), dtype="int8")
    for i, line in enumerate(lines[1:]):
        for j, genotype in enumerate(line.split("\t")[1:]):
            if genotype.strip() == "./." or genotype.strip() == "NA":
                genotype = -1
            else:
                genotype = int(genotype.strip())
            genotypes[i, j] = genotype

        # Log number of parsed samples
        if i % 100 == 0 and i != 0:
            print("Loaded", i, "out of", len(samples), "samples")

    print("Loaded", str(genotypes.shape[1]), "genotypes of", str(genotypes.shape[0]), "samples")

    return samples, snps, genotypes


def load_genotypes_parallel(line):
    # Line : Sample id and genotype values across all SNPs
    sample = (line.split("\t")[0]).strip()

    # Fill with genotypes of all SNPs for the individual
    genotypes = []
    for i in line.split("\t")[1:]:
        # Replace missing values with -1
        if i.strip() == "./." or i.strip() == "NA":
            genotype = -1
        else:
            genotype = int(i.strip())

        genotypes.append(genotype)

    genotypes = np.array(genotypes, dtype="int8")

    return sample, genotypes


def load_labels(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    mat = np.array([l.strip("\n").split("\t") for l in lines])

    samples = mat[1:, 0]
    labels = mat[1:, 1]

    print("Loaded", str(len(labels)), "labels of", str(len(samples)), "samples")

    return samples, labels


def order_labels(samples, samples_in_labels, labels):
    idx = [np.where(samples_in_labels == s)[0][0] for s in samples]

    return np.array([labels[i] for i in idx])


def load_data(filename):
    data = np.load(filename)

    return data


# Not sure if this will be used
def load_data_(filename):
    data = np.load(filename)

    return data["inputs"], data["labels"], data["samples"], data["label_names"], data["snp_names"]


def load_folds_indexes(filename):
    data = np.load(filename, allow_pickle=True)

    return data["folds_indexes"]


def load_embedding(filename, which_fold):
    data = np.load(filename)
    embs = data["emb"]
    emb = torch.from_numpy(embs[which_fold])

    return emb


def get_fold_data(which_fold, folds_indexes, data, label="labels"):
    # Indices of each set for the fold (0:train, 1:valid, 2:test)
    fold_indexes = folds_indexes[which_fold]
    train_indexes = np.sort(fold_indexes[0])  # sort is a hdf5 requirement
    valid_indexes = np.sort(fold_indexes[1])
    test_indexes = np.sort(fold_indexes[2])

    # Get data (x,y,samples) of each set (train, valid, test)
    x_train = data["inputs"][train_indexes]
    y_train = data[label][train_indexes]
    samples_train = data["samples"][train_indexes]

    x_valid = data["inputs"][valid_indexes]
    y_valid = data[label][valid_indexes]
    samples_valid = data["samples"][valid_indexes]

    x_test = data["inputs"][test_indexes]
    y_test = data[label][test_indexes]
    samples_test = data["samples"][test_indexes]

    return (
        train_indexes,
        valid_indexes,
        test_indexes,
        x_train,
        y_train,
        samples_train,
        x_valid,
        y_valid,
        samples_valid,
        x_test,
        y_test,
        samples_test,
    )


# !!This is the old function that has to be removed eventually!!
def _get_fold_data(which_fold, folds_indexes, data, split_ratio=None, seed=None):
    # Set aside fold nb of which_fold for test
    test_indexes = folds_indexes[which_fold]

    # Other folds are used for train and valid sets
    other_folds = [i for i in range(len(folds_indexes)) if i != which_fold]

    # Concat indices of other folds
    other_indexes = np.concatenate([folds_indexes[i] for i in other_folds])

    # If we are generating embeddings, we don't need train/valid sets
    if split_ratio is None:
        x = data["inputs"][other_indexes]
        y = data["labels"][other_indexes]
        samples = data["samples"][other_indexes]

        return x, y, samples

    # Split indexes into train and valid set
    train_indexes, valid_indexes = split(other_indexes, split_ratio, seed)

    # Get data (x,y,samples) of each set (train, valid, test)
    x_train = torch.from_numpy(data["inputs"][train_indexes])
    y_train = torch.from_numpy(data["labels"][train_indexes])
    samples_train = data["samples"][train_indexes]

    x_valid = torch.from_numpy(data["inputs"][valid_indexes])
    y_valid = torch.from_numpy(data["labels"][valid_indexes])
    samples_valid = data["samples"][valid_indexes]

    x_test = torch.from_numpy(data["inputs"][test_indexes])
    y_test = torch.from_numpy(data["labels"][test_indexes])
    samples_test = data["samples"][test_indexes]

    return (
        train_indexes,
        valid_indexes,
        test_indexes,
        x_train,
        y_train,
        samples_train,
        x_valid,
        y_valid,
        samples_valid,
        x_test,
        y_test,
        samples_test,
    )


def compute_norm_values(x):
    """
    x is a tensor
    """
    # Non missing values
    mask = x >= 0

    # Compute mean of every column (feature)
    with torch.no_grad():
        per_feature_mean = torch.sum(x * mask, dim=0) / torch.sum(mask, dim=0)

        # S.d. of every column (feature)
        per_feature_sd = torch.sqrt(
            torch.sum((x * mask - mask * per_feature_mean) ** 2, dim=0)
            / (torch.sum(mask, dim=0) - 1)
        )
        per_feature_sd += 1e-6

    return per_feature_mean, per_feature_sd


def replace_missing_values(x, per_feature_mean):
    """
    x and per_feature_mean are tensors
    """
    mask = x >= 0

    for i in range(x.shape[0]):
        x[i] = mask[i] * x[i] + (~mask[i]) * per_feature_mean


def normalize(x, per_feature_mean, per_feature_sd):
    """
    x, per_feature_mean and per_feature_sd are tensors
    """
    x_norm = (x - per_feature_mean) / per_feature_sd

    return x_norm


class InferenceDataset(torch.utils.data.Dataset):
    """
    Dataset for inference on PLINK files with automatic SNP alignment.

    This dataset handles:
    - Loading test PLINK genotypes
    - Aligning SNPs to model's training SNPs
    - Missing value imputation (using model's training means)
    - Normalization (using model's training stats)
    """

    def __init__(
        self,
        plink_prefix,
        model_package,
        use_memmap=True,
        cache_file=None,
        force_cache_rebuild=False,
        verbose=True,
    ):
        """
        Args:
            plink_prefix: Path to PLINK files (without .bed/.bim/.fam extension)
            model_package: ModelPackage instance with model metadata
            use_memmap: Whether to use memory mapping for large datasets
            cache_file: Optional file path for caching genotypes
            force_cache_rebuild: If True, rebuild cache even if it exists
        """
        from pathlib import Path

        from Dietnet.helpers.snp_alignment import check_alignment_quality, create_snp_mapping

        self.plink_prefix = plink_prefix
        self.model_package = model_package

        if verbose:
            print(f"\n=== Preparing inference dataset ===")
            print(f"Test PLINK: {plink_prefix}")
            print(f"Model: seed {model_package.seed}, fold {model_package.fold}")

        # Load test genotypes
        if verbose:
            print("\nCreating/using genotype cache (memmap)...")
        if cache_file:
            cache_file = Path(cache_file)

        self.genotypes, self.fam_data, self.bim_data = load_plink_genotypes(
            plink_prefix,
            cache_file=cache_file,
            use_memmap=use_memmap,
            force_cache_rebuild=force_cache_rebuild,
            verbose=verbose,
        )

        self.n_samples = len(self.fam_data)
        if verbose:
            print(f"Loaded {self.n_samples} samples, {len(self.bim_data)} SNPs")

        # Create SNP alignment mapping
        if verbose:
            print("\nAligning SNPs to model...")
        test_bim_path = f"{plink_prefix}.bim"
        self.snp_mapping, self.alignment_info = create_snp_mapping(
            test_bim=test_bim_path, model_snps=model_package.snps, fill_value=-1
        )

        # Check alignment quality (warns if poor overlap)
        check_alignment_quality(
            self.alignment_info, min_overlap=0.1, raise_on_poor=False, verbose=verbose
        )

        # Load model's input statistics for imputation and normalization
        input_stats = model_package.input_stats
        self.training_means = torch.from_numpy(input_stats["mean"]).float()

        if "std" in input_stats:
            self.training_stds = torch.from_numpy(input_stats["std"]).float()
        else:
            # If no std available, compute from means (won't normalize, just impute)
            self.training_stds = torch.ones_like(self.training_means)

        if verbose:
            print(f"✓ Inference dataset ready: {self.n_samples} samples")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        """
        Returns preprocessed genotypes for a single sample.

        Returns:
            Tuple of (aligned_genotypes, sample_id)
            - aligned_genotypes: torch.Tensor of shape (n_model_snps,)
            - sample_id: str
        """
        # Get genotypes for this sample (all test SNPs)
        sample_geno = self.genotypes[index]  # Shape: (n_test_snps,)

        # Align to model SNPs
        aligned_geno = self._align_sample(sample_geno)

        # Convert to tensor
        aligned_geno = torch.from_numpy(aligned_geno).float()

        # Impute missing values with training means
        mask = aligned_geno >= 0
        aligned_geno = mask * aligned_geno + (~mask) * self.training_means

        # Normalize using training statistics
        aligned_geno = (aligned_geno - self.training_means) / self.training_stds

        # Get sample ID from FAM file
        sample_id = self.fam_data.iloc[index]["iid"]

        return aligned_geno, sample_id

    def _align_sample(self, sample_genotypes):
        """
        Align a single sample's genotypes to model SNP order.

        Args:
            sample_genotypes: Array of shape (n_test_snps,)

        Returns:
            Aligned genotypes of shape (n_model_snps,)
        """
        n_model_snps = len(self.snp_mapping)
        aligned = np.full(n_model_snps, -1, dtype=np.float32)

        # Fill in matched SNPs
        valid_mask = self.snp_mapping >= 0
        valid_indices = self.snp_mapping[valid_mask]
        aligned[valid_mask] = sample_genotypes[valid_indices]

        return aligned

    def get_sample_ids(self):
        """Get all sample IDs in order."""
        return self.fam_data["iid"].tolist()
