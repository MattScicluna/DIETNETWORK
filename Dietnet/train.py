import argparse
import os
import pprint
import sys
import time
from pathlib import PurePath

import h5py
import numpy as np
import yaml

# Optional comet_ml for experiment tracking
try:
    from comet_ml import Experiment, Optimizer

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    Experiment = None
    Optimizer = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Dietnet.helpers import dataset_utils as du
from Dietnet.helpers import log_utils as lu
from Dietnet.helpers import mainloop_utils as mlu
from Dietnet.helpers import model


def main():
    args = parse_args()
    main_with_args(args)


def main_with_args(args):
    # Create dir where training info will be saved
    """
    The directory will be created in exp_path/exp_name with the name
    exp_name_foldi where i is the number of the fold
    """
    out_dir = lu.create_out_dir(
        args.exp_path, args.exp_name, args.which_fold, seed=getattr(args, "seed_override", None)
    )

    # Create the full config
    """
    The full config contains 2 level info
        - hyperparams : provided in the config file
        - specifics : paths and files used in the training process
                      (specified with command line arguments)
    """
    config = {}

    # Hyperparameters
    f = open(os.path.join(args.exp_path, args.exp_name, args.config), "r")
    config_hyperparams = yaml.load(f, Loader=yaml.FullLoader)
    if getattr(args, "seed_override", None) is not None:
        config_hyperparams["seed"] = args.seed_override

    config["params"] = config_hyperparams

    # Add fold to config hyperparams
    config["params"]["fold"] = args.which_fold

    # Specifics
    specifics = {}
    specifics["exp_path"] = args.exp_path
    specifics["exp_name"] = args.exp_name
    specifics["out_dir"] = out_dir
    specifics["partition"] = args.partition
    specifics["dataset"] = args.dataset
    specifics["embedding"] = args.embedding
    specifics["normalize"] = args.normalize
    # specifics['preprocess_params'] = args.preprocess_params
    specifics["input_features_means"] = args.input_features_means
    specifics["label_file"] = args.label_file
    specifics["task"] = args.task
    specifics["param_init"] = args.param_init

    config["specifics"] = specifics

    # This is the full configurations for the training
    print("\n --- Experiment hyperparameters and specifics")
    pprint.pprint(config)

    # Save experiment configurations (out_dir/full_config.log)
    if not args.optimization:
        lu.save_exp_params(config["specifics"]["out_dir"], "full_config.log", config)

    exp_name = (
        "model_params"
        + "_epochs_"
        + str(config["params"]["epochs"])
        + "_inpdrop_"
        + str(config["params"]["input_dropout"])
        + "_lr_aux_"
        + str(config["params"]["lr_aux"])
        + "_lr_main_"
        + str(config["params"]["lr_main"])
        + "_lra_"
        + str(config["params"]["learning_rate_annealing"])
        + "_auxu_"
        + str(config["params"]["nb_hidden_u_aux"])[1:-1].replace(", ", "_")
        + "_mainu_"
        + str(config["params"]["nb_hidden_u_aux"][-1])
        + "_"
        + str(config["params"]["nb_hidden_u_main"])[1:-1].replace(", ", "_")
        + "_uniform_init_limit_"
        + str(config["params"]["uniform_init_limit"])
        + "_patience_"
        + str(config["params"]["patience"])
        + "_seed_"
        + str(config["params"]["seed"])
        + ".pt"
    )

    # Training
    train(config, args.comet_ml, args.comet_ml_project_name, args.optimization)


def train(config, comet_log, comet_project_name, optimization_exp):
    # Monitoring time to execute the whole training function
    whole_exp_start_time = time.time()

    # ----------------------------------------
    #       EXPERIMENT IDENTIFIER
    # ----------------------------------------
    # Experiment identifier for naming files
    exp_identifier = (
        "auxu_"
        + str(config["params"]["nb_hidden_u_aux"])[1:-1].replace(", ", "_")
        + "_mainu_"
        + str(config["params"]["nb_hidden_u_aux"][-1])
        + "_"
        + str(config["params"]["nb_hidden_u_main"])[1:-1].replace(", ", "_")
        + "_lr_aux_"
        + str(config["params"]["lr_aux"])
        + "_lr_main_"
        + str(config["params"]["lr_main"])
        + "_lra_"
        + str(config["params"]["learning_rate_annealing"])
        + "_epochs_"
        + str(config["params"]["epochs"])
        + "_patience_"
        + str(config["params"]["patience"])
        + "_inpdrop_"
        + str(config["params"]["input_dropout"])
        + "_seed_"
        + str(config["params"]["seed"])
    )
    # ----------------------------------------
    #               COMET PROJECT
    # ----------------------------------------
    if comet_log:
        if not COMET_AVAILABLE:
            print("WARNING: comet_ml not available. Install with: uv pip install comet-ml")
            print("Continuing without experiment tracking...")
            comet_log = False
        else:
            # Init experiment
            if comet_project_name is None:
                experiment = Experiment(auto_histogram_weight_logging=True)

            else:
                experiment = Experiment(
                    project_name=comet_project_name, auto_metric_logging=False, parse_args=False
                )

            # Set experiment name
            experiment.set_name(exp_identifier)

            # Log hyperparams
            experiment.log_parameters(config["params"])

            # Log specifics
            experiment.log_others(config["specifics"])

    # ----------------------------------------
    #               SET DEVICE
    # ----------------------------------------
    print("\n --- Setting device ---")
    print("Cuda available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ----------------------------------------
    #               FIX SEED
    # ----------------------------------------
    seed = config["params"]["seed"]
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    #        LOAD INPUT FEATURES STATS
    # ----------------------------------------
    print("\n --- Loading input features stats ---")

    inp_feat_stats = np.load(
        os.path.join(config["specifics"]["exp_path"], config["specifics"]["input_features_means"])
    )

    fold = config["params"]["fold"]
    mus = torch.from_numpy(inp_feat_stats["means_by_fold"][fold]).float().to(device)
    sigmas = (
        torch.from_numpy(inp_feat_stats["sd_by_fold"][fold]).float().to(device)
        if config["specifics"]["normalize"]
        else None
    )

    # ----------------------------------------
    #           LOAD FOLD INDEXES
    # ----------------------------------------
    print("\n --- Loading fold indexes of train, valid and test sets ---")
    all_folds_idx = np.load(
        os.path.join(config["specifics"]["exp_path"], config["specifics"]["partition"]),
        allow_pickle=True,
    )

    fold_idx = all_folds_idx["folds_indexes"][config["params"]["fold"]]

    # ----------------------------------------
    #       MAKE TRAIN, VALID, TEST SETS
    # ----------------------------------------
    print("\n --- Making train, valid, test sets classes ---")

    # Dataset file
    dataset_file = os.path.join(config["specifics"]["exp_path"], config["specifics"]["dataset"])

    # Label conversion depending on task
    if config["specifics"]["task"] == "classification":
        label_type = np.int64
    elif config["specifics"]["task"] == "regression":
        label_type = np.float32

    # Detect dataset type
    if config["specifics"]["dataset"].endswith(".hdf5") or config["specifics"]["dataset"].endswith(
        ".h5"
    ):
        # HDF5 mode (backward compatibility)
        print("Using HDF5 dataset")
        du.FoldDataset.dataset_file = dataset_file
        du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, "r")
        du.FoldDataset.task = config["specifics"]["task"]
        du.FoldDataset.label_type = label_type
        dataset_class = du.FoldDataset

    else:
        # PLINK mode
        print("Using PLINK dataset")
        from Dietnet.helpers.dataset_utils import PLINKFoldDataset, load_plink_genotypes

        plink_prefix = dataset_file.replace(".bed", "")

        # Optional: use .npy cache for faster loading
        cache_file = PurePath(
            config["specifics"]["exp_path"],
            config["specifics"]["dataset"].replace(".bed", "_genotypes.npy"),
        )

        # Load all genotypes into memory
        genotypes, fam_data, bim_data = load_plink_genotypes(plink_prefix, cache_file)

        # Load and order labels
        label_file = config["specifics"].get("label_file")
        if label_file is None:
            raise ValueError("--label-file is required for PLINK datasets")

        label_samples, labels = du.load_labels(
            PurePath(config["specifics"]["exp_path"], label_file)
        )
        fam_samples = fam_data["iid"].values
        ordered_labels = du.order_labels(fam_samples, label_samples, labels)

        # Convert labels to appropriate type
        if config["specifics"]["task"] == "classification":
            label_names = np.unique(ordered_labels)
            label_to_idx = {label: idx for idx, label in enumerate(label_names)}
            ordered_labels = np.array([label_to_idx[label] for label in ordered_labels])
            # Store label names for later use
            config["_plink_label_names"] = label_names
        else:
            ordered_labels = ordered_labels.astype(np.float32)

        # Set class variables (shared across all datasets)
        PLINKFoldDataset.plink_prefix = plink_prefix
        PLINKFoldDataset.label_file = label_file
        PLINKFoldDataset.task = config["specifics"]["task"]
        PLINKFoldDataset.label_type = label_type
        PLINKFoldDataset.genotype_cache = genotypes
        PLINKFoldDataset.fam_data = fam_data
        PLINKFoldDataset.bim_data = bim_data
        PLINKFoldDataset.ordered_labels = ordered_labels

        dataset_class = PLINKFoldDataset

    # Create datasets (works for both HDF5 and PLINK)
    train_set = dataset_class(fold_idx[0])
    print("training set:", len(train_set))
    valid_set = dataset_class(fold_idx[1])
    print("valid set:", len(valid_set))
    test_set = dataset_class(fold_idx[2])
    print("test set:", len(test_set))

    # ----------------------------------------
    #             LOAD EMBEDDING
    # ----------------------------------------
    print("\n --- Loading embedding ---")
    emb = du.load_embedding(
        os.path.join(config["specifics"]["exp_path"], config["specifics"]["embedding"]),
        config["params"]["fold"],
    )

    # Send to device
    emb = emb.to(device)
    emb = emb.float()

    # Normalize embedding
    emb_norm = (emb**2).sum(0) ** 0.5
    emb = emb / emb_norm

    # ----------------------------------------
    #               MAKE MODEL
    # ----------------------------------------
    print("\n --- Initiating the model ---")

    # Aux net input size (nb of emb features)
    if len(emb.size()) == 1:
        n_feats_emb = 1  # input of aux net, 1 value per SNP
        emb = torch.unsqueeze(emb, dim=1)  # match size in Linear fct (nb_snpsx1)
    else:
        n_feats_emb = emb.size()[1]  # input of aux net

    # Main net input size (nb of features)
    n_feats = emb.size()[0]  # input of main net

    # Main net output size (nb targets)
    if config["specifics"]["task"] == "classification":
        # For PLINK, use stored label names; for HDF5, read from file
        if "_plink_label_names" in config:
            n_targets = len(config["_plink_label_names"])
        else:
            with h5py.File(dataset_file, "r") as f:
                # Try class_label_names first (new format), fall back to label_names
                if "class_label_names" in f:
                    n_targets = len(f["class_label_names"])
                else:
                    n_targets = len(f["label_names"])
    elif config["specifics"]["task"] == "regression":
        n_targets = 1

    print("\n***Nb features in models***")
    print("n_feats_emb:", n_feats_emb)
    print("n_feats:", n_feats)
    print("n_targets:", n_targets)

    # Model init
    model_init_start_time = time.time()
    comb_model = model.CombinedModel(
        n_feats=n_feats_emb,
        n_hidden_u_aux=config["params"]["nb_hidden_u_aux"],
        n_hidden_u_main=config["params"]["nb_hidden_u_aux"][-1:]
        + config["params"]["nb_hidden_u_main"],
        n_targets=n_targets,
        param_init=config["specifics"]["param_init"],
        aux_uniform_init_limit=config["params"]["uniform_init_limit"],
        input_dropout=config["params"]["input_dropout"],
    )
    print("Model initiated in: ", time.time() - model_init_start_time, "seconds")

    print("Sending model to device")
    comb_model.to(device)

    # ----------------------------------------
    #               OPTIMIZATION
    # ----------------------------------------
    # Loss
    if config["specifics"]["task"] == "classification":
        criterion = nn.CrossEntropyLoss()
    elif config["specifics"]["task"] == "regression":
        criterion = nn.MSELoss()

    print("For loss we use:", criterion)

    # Optimizer with separate learning rates for auxiliary and main networks
    optimizer = torch.optim.Adam(
        [
            {"params": comb_model.aux_net.parameters(), "lr": config["params"]["lr_aux"]},
            {"params": comb_model.main_net.parameters(), "lr": config["params"]["lr_main"]},
        ]
    )

    # Max nb of epochs
    n_epochs = config["params"]["epochs"]

    # ----------------------------------------
    #             BATCH GENERATORS
    # ----------------------------------------
    print("\n --- Making batch generators ---")
    batch_gen_start_time = time.time()

    batch_size = config["params"]["batch_size"]

    train_generator = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_generator = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_generator = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Batch generators initiated in:", time.time() - batch_gen_start_time, "seconds")

    # ----------------------------------------
    #          TRAINING LOOP SET UP
    # ----------------------------------------
    model_params_filename = "best_model.pt"

    # Baseline: forward pass on validation set before any training
    print("\n --- Computing baseline ---")
    comb_model.eval()
    baseline = mlu.eval_step(
        comb_model,
        device,
        valid_generator,
        len(valid_set),
        criterion,
        mus,
        sigmas,
        emb,
        config["specifics"]["task"],
        config["specifics"]["normalize"],
    )

    if config["specifics"]["task"] == "classification":
        print("baseline loss:", baseline[0], "baseline acc:", baseline[1])
    elif config["specifics"]["task"] == "regression":
        print("baseline loss:", baseline[0])

    best_result = baseline
    lu.save_model_params(config["specifics"]["out_dir"], comb_model, filename=model_params_filename)

    patience = 0
    max_patience = config["params"]["patience"]
    has_early_stoped = False

    # ----------------------------------------
    #           TRAINING LOOP
    # ----------------------------------------
    from tqdm import tqdm

    pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in pbar:

        # --- Train step (metrics computed for free during the forward pass) ---
        comb_model.train()
        epoch_train_result = mlu.train_step(
            comb_model,
            device,
            optimizer,
            train_generator,
            len(train_set),
            criterion,
            mus,
            sigmas,
            emb,
            config["specifics"]["task"],
            config["specifics"]["normalize"],
        )

        # --- Eval on valid set only (used for early stopping) ---
        comb_model.eval()
        epoch_valid_result = mlu.eval_step(
            comb_model,
            device,
            valid_generator,
            len(valid_set),
            criterion,
            mus,
            sigmas,
            emb,
            config["specifics"]["task"],
            config["specifics"]["normalize"],
        )

        # Update progress bar
        if config["specifics"]["task"] == "classification":
            pbar.set_postfix(
                {
                    "train_acc": f"{epoch_train_result[1]:.3f}",
                    "valid_acc": f"{epoch_valid_result[1]:.3f}",
                    "train_loss": f"{epoch_train_result[0]:.3f}",
                }
            )
        elif config["specifics"]["task"] == "regression":
            pbar.set_postfix(
                {
                    "train_mse": f"{epoch_train_result[0]:.3f}",
                    "valid_mse": f"{epoch_valid_result[0]:.3f}",
                    "train_loss": f"{epoch_train_result[0]:.3f}",
                }
            )

        # Log to comet-ml
        if comet_log:
            if config["specifics"]["task"] == "classification":
                experiment.log_metric("train_loss", epoch_train_result[0], epoch=epoch, step=epoch)
                experiment.log_metric(
                    "train_accuracy", epoch_train_result[1], epoch=epoch, step=epoch
                )
                experiment.log_metric("valid_loss", epoch_valid_result[0], epoch=epoch, step=epoch)
                experiment.log_metric(
                    "valid_accuracy", epoch_valid_result[1], epoch=epoch, step=epoch
                )
            elif config["specifics"]["task"] == "regression":
                experiment.log_metric("train_loss", epoch_train_result[0], epoch=epoch, step=epoch)
                experiment.log_metric("valid_loss", epoch_valid_result[0], epoch=epoch, step=epoch)

        # --- Check improvement and save best model ---
        if mlu.has_improved(best_result, epoch_valid_result):
            patience = 0
            best_result = mlu.update_best_result(best_result, epoch_valid_result)
            lu.save_model_params(
                config["specifics"]["out_dir"], comb_model, filename=model_params_filename
            )
            pbar.write(f"✓ Best validation at epoch {epoch+1} - saving model")
        else:
            patience += 1

        # --- Early stopping ---
        if patience >= max_patience:
            has_early_stoped = True
            # log best validation results to comet
            if comet_log:
                if config["specifics"]["task"] == "classification":
                    experiment.log_metric("best_valid_loss", best_result[0])
                    experiment.log_metric("best_valid_acc", best_result[1])

                if config["specifics"]["task"] == "regression":
                    experiment.log_metric("best_valid_loss", best_result[0])
            break  # exit training loop

        # ---Anneal learning rate---
        for param_group in optimizer.param_groups:
            param_group["lr"] *= config["params"]["learning_rate_annealing"]

    pbar.close()
    print(f"\nEarly stopping: {has_early_stoped}")

    # ----------------------------------------
    #                 TEST
    # ----------------------------------------
    # Monitoring time
    start_time = time.time()

    # Reload weights from early stoping
    model_weights_path = os.path.join(config["specifics"]["out_dir"], model_params_filename)
    comb_model.load_state_dict(torch.load(model_weights_path))

    # Put model in eval mode
    comb_model.eval()

    # Test step
    print("Testing model", flush=True)
    test_samples, test_ys, test_results = mlu.test_step(
        comb_model,
        device,
        test_generator,
        len(test_set),
        criterion,
        mus,
        sigmas,
        emb,
        config["specifics"]["task"],
        config["specifics"]["normalize"],
    )

    # Monitoring time
    print("Test time:", time.time() - start_time, flush=True)

    # Print result (and optional save to comet-ml)
    if config["specifics"]["task"] == "classification":
        print("Final accuracy:", str(test_results[2]), flush=True)

        if comet_log:
            experiment.log_metric("test accuracy", test_results[2])

    elif config["specifics"]["task"] == "regression":
        print("Test loss:", str(test_results[0]), flush=True)
        print("Pearson correlation between outputs and targets:", str(test_results[2]), flush=True)

        if comet_log:
            experiment.log_metric("Test loss", test_results[0])
            experiment.log_metric("Pearson_r", test_results[1])

    # Save test results (model_predictions.npz)
    if not optimization_exp:
        print("Saving results", flush=True)
        if config["specifics"]["task"] == "classification":
            # For PLINK, reuse stored label names; for HDF5, read from file
            if "_plink_label_names" in config:
                label_names = np.array(config["_plink_label_names"]).astype(np.str_)
            else:
                with h5py.File(dataset_file, "r") as f:
                    # Try class_label_names first (new format), fall back to label_names
                    if "class_label_names" in f:
                        label_names = np.array(f["class_label_names"]).astype(np.str_)
                    else:
                        label_names = np.array(f["label_names"]).astype(np.str_)

            lu.save_results(
                config["specifics"]["out_dir"],
                test_samples,
                test_ys,
                label_names,
                test_results[0].cpu(),
                test_results[1].cpu(),
            )

            lu.save_predictions_tsv(
                config["specifics"]["out_dir"],
                test_samples,
                test_results[1].cpu().numpy(),
                label_names,
            )

        elif config["specifics"]["task"] == "regression":
            lu.save_results_regression(
                config["specifics"]["out_dir"],
                test_samples,
                test_ys,
                test_results[1].detach().squeeze().cpu(),
            )

        print("\n--- End of execution ---")
        print("Executed training process in {} seconds".format(time.time() - whole_exp_start_time))


def parse_args():
    parser = argparse.ArgumentParser(description=("Train, eval and test model of a given fold"))

    # Paths
    parser.add_argument(
        "--exp-path",
        type=str,
        required=True,
        help="Path to directory of dataset, folds indexes and embedding.",
    )

    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help=(
            "Name of directory where to save the results. "
            "This direcotry must be in the directory specified with "
            "exp-path. "
        ),
    )

    # Files
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Yaml file of hyperparameter. Default: %(default)s",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset.hdf5",
        help=(
            "Filename of dataset returned by create_dataset.py "
            "The file must be in direcotry specified with exp-path "
            "Default: %(default)s"
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
        "--embedding",
        type=str,
        default="embedding.npz",
        help=(
            "Filename of embedding returned by generate_embedding.py "
            "The file must be in directory specified with exp-path. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--input-features-means",
        type=str,
        default="input_features_means.npz",
        help=(
            "Filename of computed input features means. The means are "
            "used to replace missing genotypes. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--label-file",
        type=str,
        default=None,
        help="Path to label file (TSV format, required for PLINK datasets)",
    )

    # Input features normalization
    parser.add_argument(
        "--normalize", action="store_true", help="Use this flag to normalize input features."
    )

    """
    parser.add_argument(
            '--preprocess-params',
            type=str,
            default='preprocessing_params.npz',
            help='Normalization parameters obtained with get_preprocessing_params.py'
            )
    """
    # Task
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        required=True,
        help="Type of prediction : classification or regression",
    )

    # Fold
    parser.add_argument(
        "--which-fold",
        type=int,
        default=0,
        help="Which fold to train (1st fold is 0). Default: %(default)i",
    )

    # Optional param init
    parser.add_argument("--param-init", type=str, help="File of parameters initialization values")

    # Comet-ml
    parser.add_argument(
        "--comet-ml", action="store_true", help="Use this flag to run experiment with comet ml"
    )

    parser.add_argument(
        "--comet-ml-project-name", type=str, help="Specific project were to send comet Experiment"
    )

    # Optimization process (do not save results)
    parser.add_argument(
        "--optimization",
        action="store_true",
        help=(
            "Use this flag when in optimization process. "
            "(Config files created with optimize.py)."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
