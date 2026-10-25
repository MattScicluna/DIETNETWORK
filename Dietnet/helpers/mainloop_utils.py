import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Dietnet.helpers import dataset_utils as du
from Dietnet.helpers import model


def train_step(
    comb_model,
    device,
    optimizer,
    train_generator,
    set_size,
    criterion,
    mus,
    sigmas,
    emb,
    task,
    normalize,
):
    # Monitoring set up : Minibatch
    minibatch_loss = []
    minibatch_n_right = []  # nb of good classifications

    for x_batch, y_batch, _ in train_generator:
        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        if task == "regression":
            y_batch = y_batch.unsqueeze(1)

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        if normalize:
            x_batch = du.normalize(x_batch, mus, sigmas)

        # Reset optimizer
        optimizer.zero_grad()

        # Forward pass
        comb_model_out = comb_model(emb, x_batch)

        # Compute loss (softmax computation done in loss)
        loss = criterion(comb_model_out, y_batch)

        # Compute gradients
        loss.backward()

        # Optimize
        optimizer.step()

        # Monitoring : Minibatch
        minibatch_loss.append(loss.item())  # mean loss of the minibatch

        # Classification: keep nb of good predictions for accuracy computation
        if task == "classification":
            _, pred = get_predictions(comb_model_out)  # softmax computation
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

    # Monitoring: Epoch
    epoch_loss = np.array(minibatch_loss).mean()

    if task == "classification":
        epoch_acc = np.array(minibatch_n_right).sum() / float(set_size) * 100
        epoch_result = (epoch_loss, epoch_acc)

    elif task == "regression":
        epoch_result = (epoch_loss,)

    return epoch_result


def eval_step(
    comb_model, device, valid_generator, set_size, criterion, mus, sigmas, emb, task, normalize
):
    # Monitoring: Minibatch setup
    minibatch_loss = []
    minibatch_n_right = []  # nb of good classifications

    for x_batch, y_batch, _ in valid_generator:
        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        if task == "regression":
            y_batch = y_batch.unsqueeze(1)

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        if normalize:
            x_batch = du.normalize(x_batch, mus, sigmas)

        # Forward pass
        comb_model_out = comb_model(emb, x_batch)

        # Loss
        loss = criterion(comb_model_out, y_batch)

        # Monitoring : Minibatch
        weighted_loss = loss.item() * len(y_batch)  # for unequal minibatches
        minibatch_loss.append(weighted_loss)

        # Classification: keep nb of good predictions for accuracy computation
        if task == "classification":
            _, pred = get_predictions(comb_model_out)  # softmax computation
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

    epoch_loss = np.array(minibatch_loss).sum() / set_size
    # epoch_loss = np.array(minibatch_loss).mean()

    if task == "classification":
        epoch_acc = np.array(minibatch_n_right).sum() / float(set_size) * 100
        epoch_result = (epoch_loss, epoch_acc)

    elif task == "regression":
        epoch_result = (epoch_loss,)

    return epoch_result


def test_step(
    comb_model, device, test_generator, set_size, criterion, mus, sigmas, emb, task, normalize
):
    # Saving data seen while looping through minibatches
    minibatch_loss = []
    minibatch_n_right = []  # number of good classifications
    test_pred = torch.tensor([]).to(device)  # prediction of each sample
    test_score = torch.tensor([]).to(device)  # softmax values of each sample
    test_samples = np.array([])  # test samples
    test_ys = np.array([])  # true labels of test samples

    for i, (x_batch, y_batch, samples) in enumerate(test_generator):
        # Save samples
        test_samples = np.concatenate([test_samples, samples])
        # Save labels
        test_ys = np.concatenate([test_ys, y_batch])

        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        if task == "regression":
            y_batch = y_batch.unsqueeze(1)

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        if normalize:
            x_batch = du.normalize(x_batch, mus, sigmas)

        # Forward pass
        comb_model_out = comb_model(emb, x_batch)

        # Loss
        loss = criterion(comb_model_out, y_batch)

        # Monitoring : Minibatch
        weighted_loss = loss.item() * len(y_batch)  # for unequal minibatches
        minibatch_loss.append(weighted_loss)

        # Predictions
        if task == "classification":
            score, pred = get_predictions(comb_model_out)
            test_pred = torch.cat((test_pred, pred), dim=-1)
            test_score = torch.cat((test_score, score), dim=0)

            # Nb of good classifications for the minibatch
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

        elif task == "regression":
            test_pred = torch.cat((test_pred, comb_model_out.detach()), dim=0)

    test_loss = np.array(minibatch_loss).sum() / set_size
    # test_loss = np.array(minibatch_loss).mean()

    # Test results to return
    if task == "classification":
        # Total accuracy
        test_acc = np.array(minibatch_n_right).sum() / float(set_size) * 100

        test_results = (test_score, test_pred, test_acc)

    elif task == "regression":
        # Pearson correlation coefficient
        print("Computing Pearson correlation coefficient", flush=True)
        r = compute_correlation(test_pred, torch.from_numpy(test_ys).to(device).unsqueeze(1))
        test_results = (test_loss, test_pred, r)

    return test_samples, test_ys, test_results


def get_last_layers(comb_model, device, test_generator, set_size, mus, sigmas, emb, task):
    # Saving data seen while looping through minibatches
    minibatch_n_right = []  # number of good classifications
    test_pred = torch.tensor([]).to(device)  # prediction of each sample
    test_score = torch.tensor([]).to(device)  # softmax values of each sample
    test_samples = np.array([])  # test samples
    test_ys = np.array([])  # true labels of test samples

    before_last_layer = torch.tensor([])
    out_layer = torch.tensor([])

    for i, (x_batch, y_batch, samples) in enumerate(test_generator):
        # Save samples
        test_samples = np.concatenate([test_samples, samples])
        # Save labels
        test_ys = np.concatenate([test_ys, y_batch])

        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        x_batch = du.normalize(x_batch, mus, sigmas)

        # Forward pass
        comb_model_before_last, comb_model_out = comb_model(emb, x_batch, save_layers=True)

        # Save layers
        before_last_layer = torch.cat(
            (before_last_layer, comb_model_before_last.detach().cpu()), dim=0
        )
        out_layer = torch.cat((out_layer, comb_model_out.detach().cpu()), dim=0)

        # Predictions
        if task == "classification":
            score, pred = get_predictions(comb_model_out)
            test_pred = torch.cat((test_pred, pred), dim=-1)
            test_score = torch.cat((test_score, score), dim=0)

            # Nb of good classifications for the minibatch
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

        elif task == "regression":
            test_pred = torch.cat((test_pred, comb_model_out.detach()), dim=-1)

    # Total accuracy
    test_acc = 0.0
    if task == "classification":
        test_acc = np.array(minibatch_n_right).sum() / float(set_size) * 100

    return test_samples, test_ys, test_score, test_pred, test_acc, before_last_layer, out_layer


def get_predictions(model_output):
    with torch.no_grad():
        score = F.softmax(model_output, dim=1)
        _, pred = torch.max(score, dim=1)

    return score, pred


def compute_correlation_np(x, y):
    # Pearson's r : SUM[(xi - xmean)(yi-ymean)] / SQRT[SUM[(xi-xmean)^2]*SUM[(yi-ymean)^2]]
    vx = x - np.mean(x)
    vy = y - np.mean(y)

    r = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    return r


def compute_correlation(x, y):
    # Pearson's r : SUM[(xi - xmean)(yi-ymean)] / SQRT[SUM[(xi-xmean)^2]*SUM[(yi-ymean)^2]]
    with torch.no_grad():
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))

    return r.item()


def has_improved(best_result, actual_result):
    # Classification
    if len(best_result) == 2:
        # Improvement if actual acc is greater than best acheived acc
        if actual_result[1] > best_result[1]:
            return True
        # Improvement if acc is same as best acc and loss is min loss achieve
        if actual_result[1] == best_result[1] and actual_result[0] < best_result[0]:
            return True

        # No improvement
        return False

    # Regression
    elif len(best_result) == 1:
        # Improvement if loss is min loss achieve
        if actual_result[0] < best_result[0]:
            return True

        # No improvement
        return False


def update_best_result(best_result, actual_result):
    # Classification
    if len(best_result) == 2:
        # Accuracy
        if actual_result[1] > best_result[1]:
            updated_acc = actual_result[1]
        else:
            updated_acc = best_result[1]

        # Loss
        if actual_result[0] < best_result[0]:
            updated_loss = actual_result[0]
        else:
            updated_loss = best_result[0]

        return (updated_loss, updated_acc)

    # Regression
    if len(best_result) == 1:
        # Loss
        if actual_result[0] < best_result[0]:
            updated_loss = actual_result[0]
        else:
            updated_loss = best_result[0]
            print("Warning with updated loss")

        return (updated_loss,)


def has_improved_old(best_acc, actual_acc, min_loss, actual_loss):
    if actual_acc > best_acc:
        return True
    if actual_acc == best_acc and actual_loss < min_loss:
        return True

    return False


def train_step_mlp(mlp, device, optimizer, train_generator, set_size, criterion, mus, sigmas, task):
    # Monitoring : Minibatch setup
    minibatch_loss = []
    minibatch_n_right = []  # nb of good classifications

    for x_batch, y_batch, _ in train_generator:
        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch.float()

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        x_batch = du.normalize(x_batch, mus, sigmas)

        # Reset optimizer
        optimizer.zero_grad()

        # Forward pass
        mlp_out = mlp(x_batch)

        # Get prediction
        if task == "classification":
            # Softmax computation
            _, pred = get_predictions(mlp_out)

        # Compute loss (softmax computation done in loss if classification)
        loss = criterion(mlp_out, y_batch)

        # Compute gradients
        loss.backward()

        # Optimize
        optimizer.step()

        # Monitoring : Minibatch
        minibatch_loss.append(loss.item())  # mean loss of the minibatch
        if task == "classification":
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

    # Monitoring: Epoch
    epoch_loss = np.array(minibatch_loss).mean()
    epoch_acc = 0.0
    if task == "classification":
        epoch_acc = np.array(minibatch_n_right).sum() / float(set_size) * 100

    return epoch_loss, epoch_acc


def eval_step_mlp(mlp, device, valid_generator, set_size, criterion, mus, sigmas, task):
    # Monitoring: Minibatch setup
    minibatch_loss = []
    minibatch_n_right = []  # nb of good classifications

    for x_batch, y_batch, _ in valid_generator:
        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        x_batch = du.normalize(x_batch, mus, sigmas)

        # Forward pass
        mlp_out = mlp(x_batch)

        # Predictions
        if task == "classification":
            _, pred = get_predictions(mlp_out)

        # Loss
        loss = criterion(mlp_out, y_batch)

        # Monitoring : Minibatch
        weighted_loss = loss.item() * len(y_batch)  # for unequal minibatches
        minibatch_loss.append(weighted_loss)
        if task == "classification":
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

    epoch_loss = np.array(minibatch_loss).sum() / set_size
    epoch_acc = 0.0
    if task == "classification":
        epoch_acc = np.array(minibatch_n_right).sum() / float(set_size) * 100

    return epoch_loss, epoch_acc


def test_step_mlp(mlp, device, test_generator, set_size, mus, sigmas, task):
    # Saving data seen while looping through minibatches
    minibatch_n_right = []  # number of good classifications
    test_pred = torch.tensor([]).to(device)  # prediction of each sample
    test_score = torch.tensor([]).to(device)  # softmax values of each sample
    test_samples = np.array([])  # test samples
    test_ys = np.array([])  # true labels of test samples

    for i, (x_batch, y_batch, samples) in enumerate(test_generator):
        # Save samples
        test_samples = np.concatenate([test_samples, samples])
        # Save labels
        test_ys = np.concatenate([test_ys, y_batch])

        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        x_batch = du.normalize(x_batch, mus, sigmas)

        # Forward pass
        mlp_out = mlp(x_batch)

        # Predictions
        if task == "classification":
            score, pred = get_predictions(mlp_out)
            test_pred = torch.cat((test_pred, pred), dim=-1)
            test_score = torch.cat((test_score, score), dim=0)

            # Nb of good classifications for the minibatch
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

        elif task == "regression":
            test_pred = torch.cat((test_pred, mlp_out), dim=-1)

    # Total accuracy
    test_acc = 0.0
    if task == "classification":
        test_acc = np.array(minibatch_n_right).sum() / float(set_size) * 100

    return test_samples, test_ys, test_score, test_pred, test_acc
