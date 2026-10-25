import inspect
import os
import sys
from pathlib import Path

import numpy as np
import torch


def create_out_dir(exp_path, exp_name, fold, seed=None):
    """
    This creates a directory exp_path/exp_name/exp_name_seed{seed}_fold{fold}
    (or exp_path/exp_name/exp_name_fold{fold} when seed is None) to save results.
    """
    if seed is not None:
        dir_name = f"{exp_name}_seed{seed}_fold{fold}"
    else:
        dir_name = exp_name + "_fold" + str(fold)
    dir_path = os.path.join(exp_path, exp_name, dir_name)

    # Create directory
    if not os.path.isdir(dir_path):
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        except OSError:
            print("Creation of directory %s failed" % dir_path)
            sys.exit(1)
        else:
            print("Created", dir_path, "to save experiment info and results")

    # Directory already exist
    else:
        print("Experiment info and results will be save in %s" % dir_path)

    return dir_path


def save_exp_params(out_dir, filename, config):
    """
    Creates the file exp_params.log where experiment
    configurations will be saved for reproducibility
    """
    with open(os.path.join(out_dir, filename), "w") as f:
        for k, v in config.items():
            f.write(k + ":" + str(v) + "\n")


def save_model_summary(out_dir, model, criterion, optimizer):
    filename = "model_summary.log"

    print("Saving model architecture summary to %s" % os.path.join(out_dir, filename))

    text = "*" * 50
    text += "\nAuxiliary network (feature embedding network):\n"
    text += "*" * 50 + "\n"
    text += str(model.aux_net)
    text += "\nForward function of auxiliary net:\n"
    text += inspect.getsource(model.aux_net.forward)
    text += "\n" + "-" * 80

    text += "\n" + "*" * 40
    text += "\nMain network (discriminative network):\n"
    text += "*" * 40 + "\n"
    text += str(model.main_net)
    text += "\nForward function of main net:\n"
    text += inspect.getsource(model.main_net.forward)
    bias = False if model.main_net.fat_bias is None else True
    text += "\n --> Fat layer has bias param:" + str(bias) + "\n"
    text += "\n" + "-" * 80

    text += "\n" + "*" * 20
    text += "\nCombined models:\n"
    text += "*" * 20 + "\n"
    text += str(model)
    text += "\nForward function of combined models:\n"
    text += inspect.getsource(model.main_net.forward)

    text += "\n" + "-" * 80
    text += "\n" + "*" * 20
    text += "\nLoss and Optimizer\n"
    text += "*" * 20 + "\n"
    text += "Loss:\n" + str(criterion)
    text += "\n\nOptimizer:\n" + str(optimizer)

    with open(os.path.join(out_dir, filename), "w") as f:
        f.write(text)


def save_model_params(out_dir, model, filename="model_params.pt"):
    # filename = 'model_params.pt'

    print("Saving model parameters to %s" % os.path.join(out_dir, filename))

    torch.save(model.state_dict(), os.path.join(out_dir, filename))


def save_predictions_tsv(out_dir, samples, pred_indices, label_names):
    filename = "predictions.tsv"

    print("Saving text predictions to %s" % os.path.join(out_dir, filename))

    with open(os.path.join(out_dir, filename), "w") as f:
        for sample_id, pred_idx in zip(samples, pred_indices):
            label_name = label_names[int(pred_idx)]
            f.write(f"{sample_id}\t{label_name}\n")


def save_results(out_dir, samples, labels, label_names, score, pred):
    filename = "model_predictions.npz"

    print("Saving model predictions to %s" % os.path.join(out_dir, filename))

    np.savez(
        os.path.join(out_dir, filename),
        test_samples=samples,
        test_labels=labels,
        test_scores=score,
        test_preds=pred,
        label_names=label_names,
    )


def save_results_regression(out_dir, samples, labels, pred):
    filename = "model_predictions.npz"

    print("Saving model predictions to %s" % os.path.join(out_dir, filename))

    np.savez(
        os.path.join(out_dir, filename), test_samples=samples, test_labels=labels, test_preds=pred
    )


def save_results_external_dataset(out_dir, samples, label_names, score, pred, test_name):
    print("Saving model predictions to %s" % os.path.join(out_dir, test_name))

    np.savez(
        os.path.join(out_dir, test_name),
        test_samples=samples,
        test_scores=score,
        test_preds=pred,
        label_names=label_names,
    )


def save_additional_data(
    out_dir,
    train_samples,
    valid_samples,
    test_samples,
    test_labels,
    pred,
    score,
    label_names,
    feature_names,
    norm_mus,
    norm_sigmas,
):
    filename = "additional_data.npz"

    print("Saving additional data to %s" % os.path.join(out_dir, filename))

    np.savez(
        os.path.join(out_dir, filename),
        sample_ids_train=train_samples,
        sample_ids_valid=valid_samples,
        sample_ids_test=test_samples,
        test_labels=test_labels,
        test_preds=pred,
        test_scores=score,
        label_names=label_names,
        feature_names=feature_names,
        norm_mus=norm_mus,
        norm_sigmas=norm_sigmas,
    )


"""
def save_additional_data_regression(out_dir,
        train_samples, valid_samples, test_samples,
        test_labels, pred, test_loss,feature_names):

    filename = 'additional_data.npz'

    print('Saving additional data to %s' % os.path.join(out_dir, filename))
"""
