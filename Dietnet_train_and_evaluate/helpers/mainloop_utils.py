import time
import pprint
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.profiler as profiler

from helpers import model
from helpers import dataset_utils as du


def train_step(mod_handler, device, train_dataset, train_generator,
               mus, sigmas, normalize, results_fullpath, epoch):

    task_handler = mod_handler.task_handler

    # Reset to 0 batches results from previous epoch
    task_handler.init_batches_results(train_dataset, train_generator)

    # Batch start pos (to compile samples and labels)
    bstart = 0
    for batch, (x_batch, y_batch, samples) in enumerate(train_generator):
        # Compile batch samples and labels
        bend = bstart + len(samples) # batch end pos
        task_handler.batches_results['samples'][bstart:bend] = samples
        task_handler.batches_results['ys'][bstart:bend] = y_batch

        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        y_batch = task_handler.format_ybatch(y_batch)

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        if normalize:
            x_batch = du.normalize(x_batch, mus, sigmas)
            

        # Reset optimizer
        optimizers = mod_handler.model.get_optimizers()
        for optimizer in optimizers:
            optimizer.zero_grad()

        # Forward pass
        model_out = mod_handler.model.forward(x_batch, results_fullpath,
                                        epoch, batch, 'train')

        # Loss
        loss = task_handler.compute_loss(model_out, y_batch)

        # Compute gradients
        loss.backward()

        # Optimize
        for optimizer in optimizers:
            optimizer.step()

        # Loss summed over all outputs (by default Pytorch returns mean loss
        # computed over nb of outputs)
        loss_wo_reduction = loss.item()*len(y_batch)

        # Compile batch loss wo reduction
        task_handler.batches_results['losses_wo_reduction'][batch] = \
                loss_wo_reduction

        # Compile batch predictions
        task_handler.update_batches_preds(model_out, y_batch, bstart, bend, batch)

        # Update batch start pos
        bstart = bend

    return task_handler.batches_results.copy()


def eval_step(mod_handler, device, eval_dataset, valid_generator,
              mus, sigmas, normalize, results_fullpath, epoch, scale=1.0):

    task_handler = mod_handler.task_handler

    # Reset to 0 batches results from previous epoch
    task_handler.init_batches_results(eval_dataset, valid_generator)

    # Batch start pos (to compile samples and labels)
    bstart = 0
    for batch, (x_batch, y_batch, samples) in enumerate(valid_generator):
        # Compile batch samples and labels
        bend = bstart + len(samples) # batch end pos
        task_handler.batches_results['samples'][bstart:bend] = samples
        task_handler.batches_results['ys'][bstart:bend] = y_batch

        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        y_batch = task_handler.format_ybatch(y_batch)

        # Replace missing values : missing values (-1) become the SNP mean
        du.replace_missing_values(x_batch, mus)

        # Normalize (missing values become 0, because we substract the mean)
        if normalize:
            x_batch = du.normalize(x_batch, mus, sigmas)
        
        # The scaling does not affect missing values because they are 0
        x_batch = x_batch*scale

        # Forward pass
        model_out = mod_handler.model.forward(x_batch, results_fullpath,
                                        epoch, batch, 'valid')
        # Loss
        loss = task_handler.compute_loss(model_out, y_batch)

        # Loss summed over all outputs (by default Pytorch returns mean loss
        # computed over nb of outputs)
        loss_wo_reduction = loss.item()*len(y_batch)

        # Compile batch loss wo reduction
        task_handler.batches_results['losses_wo_reduction'][batch] = \
                loss_wo_reduction

        # Compile batch predictions
        task_handler.update_batches_preds(model_out, y_batch, bstart, bend, batch)

        # Update batch start pos
        bstart = bend

    return task_handler.batches_results.copy()


def indep_test_step(mod_handler, device, test_dataset, test_generator,
                    mus, sigmas, normalize, results_fullpath, epoch, scale):
    
    task_handler = mod_handler.task_handler

    # Reset to 0 batches results from previous epoch
    task_handler.init_indep_test_batches_results(test_dataset, test_generator)

    # Batch start pos (to compile samples and labels)
    bstart = 0
    for batch, (idx, x_batch) in enumerate(test_generator):
        # Compile batch samples and labels
        bend = bstart + len(idx) # batch end pos
        task_handler.batches_results['samples'][bstart:bend] = idx

        # Send data to device
        x_batch = x_batch.to(device)
        x_batch = x_batch.float()

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        if normalize:
            x_batch = du.normalize(x_batch, mus, sigmas)
        
        x_batch = x_batch*scale

        # Forward pass
        model_out = mod_handler.model.forward(x_batch, results_fullpath,
                                              epoch, batch, 'test')

        # Compile batch predictions
        task_handler.update_indep_test_batches_preds(model_out, bstart, bend, batch)

        # Update batch start pos
        bstart = bend

    return task_handler.batches_results.copy()


def get_last_layers(comb_model, device, test_generator, set_size,
                    mus, sigmas, emb, task):
    # Saving data seen while looping through minibatches
    minibatch_n_right = [] #number of good classifications
    test_pred = torch.tensor([]).to(device) #prediction of each sample
    test_score = torch.tensor([]).to(device) #softmax values of each sample
    test_samples = np.array([]) #test samples
    test_ys = np.array([]) #true labels of test samples

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
        before_last_layer = torch.cat((before_last_layer,comb_model_before_last.detach().cpu()),dim=0)
        out_layer = torch.cat((out_layer,comb_model_out.detach().cpu()), dim=0)

        # Predictions
        if task == 'classification':
            score, pred = get_predictions(comb_model_out)
            test_pred = torch.cat((test_pred,pred), dim=-1)
            test_score = torch.cat((test_score,score), dim=0)

            # Nb of good classifications for the minibatch
            minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

        elif task == 'regression':
            test_pred = torch.cat((test_pred,comb_model_out.detach()), dim=-1)

    # Total accuracy
    test_acc = 0.0
    if task == 'classification':
        test_acc = np.array(minibatch_n_right).sum() / float(set_size)*100

    return test_samples, test_ys, test_score, test_pred, test_acc, before_last_layer, out_layer



def create_disc_model(comb_model, emb, device):
    """
    this only works with 1 GPU or CPU
    note: the function name is misleading. This does not create a new model. It just
    takes the pre-existing model and
    returns a function that performs the forward pass on it (with fixed embedding).
    this should be okay since python passes args by reference, so even if comb_model
    weights change, the corresponding function will also change

    Finally, remember that disc_model will be in whatever mode comb_model is,
    so if comb_model is in train mode, switch it to eval mode before calling
    ßthe output of this.
    """

    if torch.cuda.device_count() > 1:
        print('warning: this only works during training/inference with 1 GPU!')
    comb_model = comb_model.eval()
    comb_model.to(device)
    discrim_model = lambda x: comb_model(emb, x) # recreate discrim_model
    return discrim_model


def create_disc_model_multi_gpu(comb_model, emb, device, eps=1e-5, incl_softmax=False):
    """
    Transforms comb_model + emb into equivalent discrim model (with fatlayer weights added as parameters)
    This model can now be sent to multiple GPUs without any bugs
    (cannot do multi-GPU with comb_model since dataparallel will attempt to split the embedding up,
    which will result in size incompatibilities)

    Must pass batchnorm eps seperately in case loading from Theano model!
    Must pass incl_softmax seperately in case loading from Theano model!
    """

    n_feats_emb = emb.size()[1] # input of aux net
    n_feats = emb.size()[0] # input of main net
    # Hidden layers size
    emb_n_hidden_u = 100
    discrim_n_hidden1_u = 100
    discrim_n_hidden2_u = 100

    #  put in eval mode and send to correct device
    comb_model = comb_model.eval().to(device)
    emb = emb.to(device)
    fatLayer_weights = torch.transpose(comb_model.feat_emb(emb),1,0)

    #  initialize discriminitive network with fatlayer as a parameter
    #  send back to cpu while loading weights
    disc_net = model.Discrim_net(fatLayer_weights,
                                 n_feats=n_feats_emb,
                                 n_hidden1_u=discrim_n_hidden1_u,
                                 n_hidden2_u=discrim_n_hidden2_u,
                                 n_targets=26,
                                 eps=eps,
                                 incl_softmax=incl_softmax)

    #  copy over all weights
    disc_net.out.weight.data = comb_model.disc_net.out.weight.data
    disc_net.out.bias.data = comb_model.disc_net.out.bias.data
    disc_net.bn2.weight.data = comb_model.disc_net.bn2.weight.data
    disc_net.bn2.bias.data = comb_model.disc_net.bn2.bias.data
    disc_net.bn2.running_mean = comb_model.disc_net.bn2.running_mean
    disc_net.bn2.running_var = comb_model.disc_net.bn2.running_var
    disc_net.hidden_2.weight.data = comb_model.disc_net.hidden_2.weight.data
    disc_net.hidden_2.bias.data = comb_model.disc_net.hidden_2.bias.data
    disc_net.bn1.weight.data = comb_model.disc_net.bn1.weight.data
    disc_net.bn1.bias.data = comb_model.disc_net.bn1.bias.data
    disc_net.bn1.running_mean = comb_model.disc_net.bn1.running_mean
    disc_net.bn1.running_var = comb_model.disc_net.bn1.running_var
    disc_net.hidden_1.bias.data = comb_model.disc_net.fat_bias.data

    assert (disc_net.hidden_1.weight.data == fatLayer_weights).all()
    disc_net = disc_net.eval().to('cpu')

    #  Now we can do dataparallel
    if torch.cuda.device_count() > 1:
        disc_net = nn.DataParallel(disc_net)
        print("{} GPUs detected! Running in DataParallel mode".format(torch.cuda.device_count()))
    disc_net.to(device)
    return disc_net

def convert_theano_array_to_pytorch_tensor(tensor, array):
    array_as_tensor = torch.from_numpy(array.T)
    assert tensor.data.shape == array_as_tensor.shape
    tensor.data = array_as_tensor

def convert_theano_array_to_pytorch_tensor_1d(tensor, array):
    array_as_tensor = torch.from_numpy(array)
    assert tensor.data.shape == array_as_tensor.shape
    tensor.data = array_as_tensor

def convert_theano_bn_pytorch(bn, theano_arr_1, theano_arr_2, theano_arr_3, theano_arr_4):
    # Idea from: https://discuss.pytorch.org/t/is-there-any-difference-between-theano-convolution-and-pytorch-convolution/10580/10
    # Specifically how to load BN from theano to pytorch
    #extractor += [nn.BatchNorm2d(in_channel)]
    #extractor[-1].weight.data = torch.from_numpy(parameters[i-3])
    #extractor[-1].bias.data = torch.from_numpy(parameters[i-2])
    #extractor[-1].running_mean = torch.from_numpy(parameters[i-1])
    #extractor[-1].running_var = torch.from_numpy((1./(parameters[i]**2)) - 1e-4)

    convert_theano_array_to_pytorch_tensor_1d(bn.weight, theano_arr_2)
    convert_theano_array_to_pytorch_tensor_1d(bn.bias, theano_arr_1)
    convert_theano_array_to_pytorch_tensor_1d(bn.running_mean, theano_arr_3)
    convert_theano_array_to_pytorch_tensor_1d(bn.running_var, (1./(theano_arr_4**2)) - 1e-4)


def load_theano_model(n_feats_emb, emb_n_hidden_u, discrim_n_hidden1_u, discrim_n_hidden2_u, n_targets, theano_weight_file, device, only_discrim_model=True):
    #  loads theano weights file into PyTorch's comb_net OR Discrim_net

    theano_model_params = np.load(theano_weight_file)
    print('theano layer shapes:')
    for f in theano_model_params.files:
        print(f, theano_model_params[f].shape)

    comb_model = model.CombinedModel(
                 n_feats=n_feats_emb,
                 n_hidden_u=emb_n_hidden_u,
                 n_hidden1_u=discrim_n_hidden1_u,
                 n_hidden2_u=discrim_n_hidden2_u,
                 n_targets=n_targets,
                 param_init=None,
                 eps=1e-4, # Theano uses 1e-4 for batch norm instead of PyTorch default of 1e-5
                 input_dropout=0.,
                 incl_softmax=True) # theano includes softmax in output

    #  feat emb model
    convert_theano_array_to_pytorch_tensor(comb_model.feat_emb.hidden_1.weight, theano_model_params['arr_0'])
    convert_theano_array_to_pytorch_tensor(comb_model.feat_emb.hidden_2.weight, theano_model_params['arr_1'])

    #  embedding
    emb = torch.tensor(theano_model_params['arr_2'])

    #  disc model
    convert_theano_array_to_pytorch_tensor_1d(comb_model.disc_net.fat_bias, theano_model_params['arr_3'])

    convert_theano_bn_pytorch(comb_model.disc_net.bn1,
                          theano_model_params['arr_4'],
                          theano_model_params['arr_5'],
                          theano_model_params['arr_6'],
                          theano_model_params['arr_7'])

    convert_theano_array_to_pytorch_tensor(comb_model.disc_net.hidden_2.weight, theano_model_params['arr_8'])
    convert_theano_array_to_pytorch_tensor_1d(comb_model.disc_net.hidden_2.bias, theano_model_params['arr_9'])

    convert_theano_bn_pytorch(comb_model.disc_net.bn2,
                              theano_model_params['arr_10'],
                              theano_model_params['arr_11'],
                              theano_model_params['arr_12'],
                              theano_model_params['arr_13'])

    convert_theano_array_to_pytorch_tensor(comb_model.disc_net.out.weight, theano_model_params['arr_14'])
    convert_theano_array_to_pytorch_tensor_1d(comb_model.disc_net.out.bias, theano_model_params['arr_15'])

    model_to_return = comb_model.to(device)

    if only_discrim_model:

        emb = emb.to(device)
        #  create disc_net from loaded comb_model
        model_to_return = create_disc_model_multi_gpu(model_to_return,
                                                      emb, device,
                                                      eps=1e-4, # Theano uses 1e-4 for batch norm instead of PyTorch default of 1e-5
                                                      incl_softmax=True) # Theano includes softmax in model

    return model_to_return
