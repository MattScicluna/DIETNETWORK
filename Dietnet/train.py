import argparse
import os
import sys
import time
import yaml
import pprint

import h5py

import numpy as np

try:
    from comet_ml import Experiment, Optimizer
except:
    Experiment, Optimizer = None, None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import helpers.dataset_utils as du
import helpers.model as model
import helpers.mainloop_utils as mlu
import helpers.log_utils as lu


def main():
    args = parse_args()

    # Create dir where training info will be saved
    """
    The directory will be created in exp_path/exp_name with the name
    exp_name_foldi where i is the number of the fold
    """
    out_dir = lu.create_out_dir(args.exp_path, args.exp_name, args.which_fold)

    # Create the full config
    """
    The full config contains 2 level info
        - hyperparams : provided in the config file
        - specifics : paths and files used in the training process
                      (specified with command line arguments)
    """
    config = {}

    # Hyperparameters
    f = open(os.path.join(args.exp_path, args.exp_name, args.config), 'r')
    config_hyperparams = yaml.load(f, Loader=yaml.FullLoader)

    config['params'] = config_hyperparams

    # Add fold to config hyperparams
    config['params']['fold'] = args.which_fold

    # Specifics
    specifics = {}
    specifics['exp_path'] = args.exp_path
    specifics['exp_name'] = args.exp_name
    specifics['out_dir'] = out_dir
    specifics['partition'] = args.partition
    specifics['dataset'] = args.dataset
    specifics['embedding'] = args.embedding
    specifics['normalize'] = args.normalize
    #specifics['preprocess_params'] = args.preprocess_params
    specifics['input_features_means'] = args.input_features_means
    specifics['task'] = args.task
    specifics['param_init'] = args.param_init

    config['specifics'] = specifics

    # This is the full configurations for the training
    pprint.pprint(config)

    # Save experiment configurations (out_dir/full_config.log)
    if not args.optimization:
        lu.save_exp_params(config['specifics']['out_dir'],'full_config.log', config)

    exp_name = 'model_params' \
            + '_epochs_' + str(config['params']['epochs']) \
            + '_inpdrop_' + str(config['params']['input_dropout']) \
            + '_lr_' + str(config['params']['learning_rate']) \
            + '_lra_' + str(config['params']['learning_rate_annealing']) \
            + '_auxu_' \
                + str(config['params']['nb_hidden_u_aux'])[1:-1].replace(', ','_') \
            + '_mainu_' \
                + str(config['params']['nb_hidden_u_aux'][-1]) + '_' \
                + str(config['params']['nb_hidden_u_main'])[1:-1].replace(', ','_') \
            + '_patience_' + str(config['params']['patience']) \
            + '_seed_' + str(config['params']['seed']) \
            + '.pt'


    # Training
    train(config, args.comet_ml, args.comet_ml_project_name, args.optimization)


def train(config, comet_log, comet_project_name, optimization_exp):
    # ----------------------------------------
    #       EXPERIMENT IDENTIFIER
    # ----------------------------------------
    # Experiment identifier for naming files
    exp_identifier = 'auxu_' \
                + str(config['params']['nb_hidden_u_aux'])[1:-1].replace(', ','_') \
            + '_mainu_' \
                + str(config['params']['nb_hidden_u_aux'][-1]) + '_' \
                + str(config['params']['nb_hidden_u_main'])[1:-1].replace(', ','_') \
            + '_lr_' + str(config['params']['learning_rate']) \
            + '_lra_' + str(config['params']['learning_rate_annealing']) \
            + '_epochs_' + str(config['params']['epochs']) \
            + '_patience_' + str(config['params']['patience']) \
            + '_inpdrop_' + str(config['params']['input_dropout']) \
            + '_seed_' + str(config['params']['seed']) \

    # ----------------------------------------
    #               COMET PROJECT
    # ----------------------------------------
    if comet_log:
        # Init experiment
        if comet_project_name is None:
            experiment = Experiment(auto_histogram_weight_logging=True)

        else:
            experiment = Experiment(
                project_name=comet_project_name,
                auto_metric_logging=False,
                parse_args=False
                )

        # Set experiment name
        experiment.set_name(exp_identifier)

        # Log hyperparams
        experiment.log_parameters(config['params'])

        # Log specifics
        experiment.log_others(config['specifics'])

    # ----------------------------------------
    #               SET DEVICE
    # ----------------------------------------
    print('Cuda available:', torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # ----------------------------------------
    #               FIX SEED
    # ----------------------------------------
    seed = config['params']['seed']
    #torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    #        LOAD INPUT FEATURES MEANS
    # ----------------------------------------
    print('loading input features mean')

    # Mean and sd per feature computed on training set
    input_features_means = np.load(os.path.join(
        config['specifics']['exp_path'],
        config['specifics']['input_features_means'])
        )

    mus = input_features_means['means_by_fold'][config['params']['fold']]
    #sigmas = preprocess_params['sd_by_fold'][config['params']['fold']]
    sigmas = None

    # Send mus and sigmans to device
    mus = torch.from_numpy(mus).float().to(device)
    #sigmas = torch.from_numpy(sigmas).float().to(device)

    # ----------------------------------------
    #           LOAD FOLD INDEXES
    # ----------------------------------------
    print('Loading fold indexes split into train, valid, test sets')
    all_folds_idx = np.load(os.path.join(
        config['specifics']['exp_path'],
        config['specifics']['partition']),
        allow_pickle=True)

    fold_idx = all_folds_idx['folds_indexes'][config['params']['fold']]

    # ----------------------------------------
    #       MAKE TRAIN, VALID, TEST SETS
    # ----------------------------------------
    print('Making train, valid, test sets classes')

    # Dataset hdf5 file
    dataset_file = os.path.join(
            config['specifics']['exp_path'],
            config['specifics']['dataset'])

    du.FoldDataset.dataset_file = dataset_file
    du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, 'r')

    # Label conversion depending on task
    if config['specifics']['task'] == 'classification':
        du.FoldDataset.label_type = np.int64
    elif config['specifics']['task'] == 'regression':
        du.FoldDataset.label_type = np.float32

    train_set = du.FoldDataset(fold_idx[0])
    print('training set:', len(train_set))
    valid_set = du.FoldDataset(fold_idx[1])
    print('valid set:', len(valid_set))
    test_set = du.FoldDataset(fold_idx[2])
    print('test set:', len(test_set))

    # ----------------------------------------
    #             LOAD EMBEDDING
    # ----------------------------------------
    print('Loading embedding')
    emb = du.load_embedding(os.path.join(
        config['specifics']['exp_path'],
        config['specifics']['embedding']),
        config['params']['fold'])

    # Send to device
    emb = emb.to(device)
    emb = emb.float()

    # Normalize embedding
    emb_norm = (emb ** 2).sum(0) ** 0.5
    emb = emb/emb_norm

    # ----------------------------------------
    #               MAKE MODEL
    # ----------------------------------------
    # Aux net input size (nb of emb features)
    if len(emb.size()) == 1:
        n_feats_emb = 1 # input of aux net, 1 value per SNP
        emb = torch.unsqueeze(emb, dim=1) # match size in Linear fct (nb_snpsx1)
    else:
        n_feats_emb = emb.size()[1] # input of aux net

    # Main net input size (nb of features)
    n_feats = emb.size()[0] # input of main net

    # Main net output size (nb targets)
    if config['specifics']['task'] == 'classification':
        with h5py.File(dataset_file, 'r') as f:
            n_targets = len(f['label_names'])
    elif config['specifics']['task'] == 'regression':
        n_targets = 1

    print('\n***Nb features in models***')
    print('n_feats_emb:', n_feats_emb)
    print('n_feats:', n_feats)
    print('n_targets:', n_targets)

    # Model init
    print('Initiating the model')
    model_init_start_time = time.time()
    comb_model = model.CombinedModel(
            n_feats=n_feats_emb,
            n_hidden_u_aux=config['params']['nb_hidden_u_aux'],
            n_hidden_u_main=config['params']['nb_hidden_u_aux'][-1:] \
                            +config['params']['nb_hidden_u_main'],
            n_targets=n_targets,
            param_init=config['specifics']['param_init'],
            input_dropout=config['params']['input_dropout'])
    print('Model initiated in: ', time.time()-model_init_start_time, 'seconds')

    # Data parallel: this is not implemented yet
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        comb_model.disc_net = nn.DataParallel(comb_model.disc_net)


    # Note: runs script in single GPU mode only!
    print('Sending model to device')
    comb_model.to(device)
    #print(summary(comb_model.feat_emb, input_size=(294427,1,1,78)))
    #print(summary(comb_model.disc_net, input_size=[(138,1,1,294427),(100,294427)]))

    # ----------------------------------------
    #               OPTIMIZATION
    # ----------------------------------------
    # Loss
    if config['specifics']['task'] == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif config['specifics']['task'] == 'regression':
        criterion = nn.MSELoss()

    print('For loss we use:', criterion)

    # Optimizer
    lr = config['params']['learning_rate']
    optimizer = torch.optim.Adam(comb_model.parameters(), lr=lr)

    # Max nb of epochs
    n_epochs = config['params']['epochs']

    # ----------------------------------------
    #             BATCH GENERATORS
    # ----------------------------------------
    print('Making batch generators')
    batch_gen_start_time = time.time()

    batch_size = config['params']['batch_size']

    train_generator = DataLoader(train_set,
                                 batch_size=batch_size, num_workers=0)
    valid_generator = DataLoader(valid_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0)
    test_generator = DataLoader(test_set,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)

    print('Batch generators initiated in:', time.time()-batch_gen_start_time, 'seconds')

    # ----------------------------------------
    #          TRAINING LOOP SET UP
    # ----------------------------------------
    """
    # Monitoring: Epoch loss and accuracy setup
    train_losses = []
    train_acc = []
    valid_losses = []
    valid_acc = []
    """
    # Monitoring set up: Epoch
    train_results_by_epoch = []
    valid_results_by_epoch = []

    # File where to save model params
    model_params_filename = 'model_params_' + exp_identifier + '.pt'

    """
    # Baseline
    comb_model.eval()
    min_loss, best_acc = mlu.eval_step(comb_model, device,
            valid_generator, len(valid_set), criterion, mus, sigmas, emb,
            config['specifics']['task'], config['specifics']['normalize'])

    print('baseline loss:',min_loss, 'baseline acc:', best_acc)
    """

    # Baseline (and best result at this point)
    print('Computing baseline (forward pass in model)')
    baseline_start_time = time.time()
    baseline = mlu.eval_step(comb_model, device,
            valid_generator, len(valid_set), criterion, mus, sigmas, emb,
            config['specifics']['task'], config['specifics']['normalize'])

    if config['specifics']['task'] == 'classification':
        print('baseline loss:', baseline[0], 'baseline acc:', baseline[1])

    elif config['specifics']['task'] == 'regression':
        print('baseline loss:', baseline[0])

    best_result = baseline
    print('Baseline computed in:', time.time()-baseline_start_time, 'seconds')

    # Save the baseline model
    lu.save_model_params(config['specifics']['out_dir'], comb_model, filename=model_params_filename)

    # Log weights initialisation values to comet-ml
    if comet_log:
        # Layers in aux net
        for i,layer in enumerate(comb_model.feat_emb.hidden_layers):
            layer_name = 'auxNet_weights_layer' + str(i)
            experiment.log_histogram_3d(layer.weight.cpu().detach().numpy(),
                                        name=layer_name,
                                        step=0)
            """
            layer_name = 'auxNet_bias_layer' + str(i)
            experiment.log_histogram_3d(layer.bias.cpu().detach().numpy(),
                                        name=layer_name,
                                        step=0)
            """

        # Layers in main net
        experiment.log_histogram_3d(
                comb_model.fatLayer_weights.cpu().detach().numpy(),
                name='mainNet_fatLayer',
                step=0)

        for i,layer in enumerate(comb_model.disc_net.hidden_layers):
            layer_name = 'mainNet_layer' + str(i+1)
            experiment.log_histogram_3d(layer.weight.cpu().detach().numpy(),
                                        name=layer_name,
                                        step=0)
            layer_name = 'mainNet_bias_layer' + str(i)
            experiment.log_histogram_3d(layer.bias.cpu().detach().numpy(),
                                        name=layer_name,
                                        step=0)

    # Patience: Nb epoch without improvement after which to stop training
    patience = 0
    max_patience = config['params']['patience']
    has_early_stoped = False

    # ----------------------------------------
    #           TRAINING LOOP
    # ----------------------------------------
    total_time = 0
    for epoch in range(n_epochs):
        print('Epoch {} of {}'.format(epoch+1, n_epochs), flush=True)
        start_time = time.time()

        # ---Training---
        comb_model.train()

        """
        epoch_loss, epoch_acc = mlu.train_step(comb_model, device, optimizer,
                train_generator, len(train_set), criterion, mus, sigmas, emb,
                config['specifics']['task'], config['specifics']['normalize'])

        print('train loss:', epoch_loss, 'train acc:', epoch_acc, flush=True)

        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Comet
        if comet_log:
            experiment.log_metric("train_accuracy", epoch_acc, epoch=epoch, step=epoch)
            experiment.log_metric("train_loss", epoch_loss, epoch=epoch, step=epoch)
        """

        epoch_train_result = mlu.train_step(comb_model, device, optimizer,
                train_generator, len(train_set), criterion, mus, sigmas, emb,
                config['specifics']['task'], config['specifics']['normalize'])

        train_results_by_epoch.append(epoch_train_result)

        # Print result (and optional save to comet-ml)
        if config['specifics']['task'] == 'classification':
            print('train loss:', epoch_train_result[0],
                  'train acc:', epoch_train_result[1], flush=True)

            if comet_log:
                experiment.log_metric("train_loss", epoch_train_result[0], epoch=epoch, step=epoch)
                experiment.log_metric("train_accuracy", epoch_train_result[1], epoch=epoch, step=epoch)

        elif config['specifics']['task'] == 'regression':
            print('train loss:', epoch_train_result[0], flush=True)

            if comet_log:
                experiment.log_metric("train_loss", epoch_train_result[0], epoch=epoch, step=epoch)

        # ---Validation---
        comb_model.eval()

        epoch_valid_result = mlu.eval_step(comb_model, device,
                valid_generator, len(valid_set), criterion, mus, sigmas, emb,
                config['specifics']['task'], config['specifics']['normalize'])

        valid_results_by_epoch.append(epoch_valid_result)

        # Print result (and optional save to comet-ml)
        if config['specifics']['task'] == 'classification':
            print('valid loss:', epoch_valid_result[0],
                  'valid acc:', epoch_valid_result[1], flush=True)

            if comet_log:
                experiment.log_metric("valid_loss", epoch_valid_result[0], epoch=epoch, step=epoch)
                experiment.log_metric("valid_accuracy", epoch_valid_result[1], epoch=epoch, step=epoch)

        elif config['specifics']['task'] == 'regression':
            print('valid loss:', epoch_valid_result[0], flush=True)

            if comet_log:
                experiment.log_metric("valid_loss", epoch_valid_result[0], epoch=epoch, step=epoch)

        # ---Baseline: check  improvement---
        """
        if mlu.has_improved(best_acc, epoch_acc,min_loss, epoch_loss):
                patience = 0
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                if epoch_loss < min_loss:
                    min_loss = epoch_loss

                # Save model parameters (for later inference)
                print('best validation acc achieved: {} (loss {}) at epoch {} saving model ...'.format(best_acc, epoch_loss, epoch))
                lu.save_model_params(config['specifics']['out_dir'], comb_model, filename=model_params_filename)
        """

        if mlu.has_improved(best_result, epoch_valid_result):
            # Reset patience
            patience = 0
            # Update best results
            best_result = mlu.update_best_result(best_result, epoch_valid_result)

            # Save model parameters (for later inference)
            print('best validation achieved at epoch {} saving model'.format(epoch+1))

        else:
            patience += 1

        # ---Early stopping---
        if patience >= max_patience:
            has_early_stoped = True
            n_epochs = epoch - patience
            break # exit training loop

        # ---Anneal laerning rate---
        for param_group in optimizer.param_groups:
            param_group['lr'] = \
                    param_group['lr'] * config['params']['learning_rate_annealing']

        # ---Time---
        #end_time = time.time()
        epoch_time = time.time() - start_time
        total_time += epoch_time
        print('time:', epoch_time, flush=True)

        if comet_log:
            experiment.log_metric("epoch_time", epoch_time, epoch=epoch, step=epoch)

    # End of training phase
    print('Early stoping:', has_early_stoped, flush=True)

    # ----------------------------------------
    #                 TEST
    # ----------------------------------------
    # Monitoring time
    start_time = time.time()

    # Reload weights from early stoping
    model_weights_path = os.path.join(config['specifics']['out_dir'], model_params_filename)
    comb_model.load_state_dict(torch.load(model_weights_path))

    # Put model in eval mode
    comb_model.eval()

    # Test step
    print('Testing model', flush=True)
    """
    test_samples, test_ys, score, pred, acc = mlu.test_step(comb_model, device,
            test_generator, len(test_set), mus, sigmas, emb,
            config['specifics']['task'], config['specifics']['normalize'])

    print('Final accuracy:', str(acc), flush=True)
    print('total running time:', str(total_time), flush=True)

    # Comet
    if comet_log:
        experiment.log_metric("test accuracy", acc)
    """

    test_samples, test_ys, test_results = mlu.test_step(comb_model, device,
            test_generator, len(test_set), mus, sigmas, emb,
            config['specifics']['task'], config['specifics']['normalize'])

    # Monitoring time
    print('Test time:', time.time()-start_time, flush=True)

    # Print result (and optional save to comet-ml)
    if config['specifics']['task'] == 'classification':
        print('Final accuracy:', str(test_results[2]), flush=True)

        if comet_log:
            experiment.log_metric("test accuracy", test_results[2])

    elif config['specifics']['task'] == 'regression':
        print('Pearson correlation between outputs and targets:',
              str(test_results[1]), flush=True)

        if comet_log:
            experiment.log_metric("Pearson_r", test_results[1])

    # Save test results (model_predictions.npz)
    if not optimization_exp:
        print('Saving results', flush=True)
        if config['specifics']['task'] == 'classification':
            with h5py.File(dataset_file, 'r') as f:
                label_names = np.array(f['label_names']).astype(np.str_)

            lu.save_results(config['specifics']['out_dir'],
                    test_samples, test_ys, label_names, score.cpu(), pred.cpu())

        elif config['specifics']['task'] == 'regression':
            lu.save_results_regression(config['specifics']['out_dir'],
                    test_samples, test_ys, test_results[0].detach().squeeze().cpu())

        # Save additional data (additional_data.npz)
        print('saving additional results', flush=True)
        print('TO DO')
        """
        train_samples = train_set.get_samples()
        valid_samples = valid_set.get_samples()
        with h5py.File(dataset_file, 'r') as f:
            snp_names = np.array(f['snp_names']).astype(np.str_)

        lu.save_additional_data(config['specifics']['out_dir'],
                                train_samples, valid_samples, test_samples,
                                test_ys, pred.cpu(), score.cpu(),
                                label_names, snp_names, mus.cpu(), sigmas.cpu())
        """

def parse_args():
    parser = argparse.ArgumentParser(
            description=('Train, eval and test model of a given fold')
            )

    # Paths
    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory of dataset, folds indexes and embedding.'
            )

    parser.add_argument(
            '--exp-name',
            type=str,
            required=True,
            help=('Name of directory where to save the results. '
                  'This direcotry must be in the directory specified with '
                  'exp-path. ')
            )

    # Files
    parser.add_argument(
            '--config',
            type=str,
            default='config.yaml',
            help='Yaml file of hyperparameter. Default: %(default)s'
            )

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.hdf5',
            help=('Filename of dataset returned by create_dataset.py '
                  'The file must be in direcotry specified with exp-path '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--partition',
            type=str,
            default='partitioned_idx.npz',
            help=('Filename of folds indexes returned by create_dataset.py '
                  'The file must be in directory specified with exp-path. '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--embedding',
            type=str,
            default='embedding.npz',
            help=('Filename of embedding returned by generate_embedding.py '
                  'The file must be in directory specified with exp-path. '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--input-features-means',
            type=str,
            default='input_features_means.npz',
            help=('Filename of computed input features means. The means are '
                  'used to replace missing genotypes. Default: %(default)s')
            )

    # Input features normalization
    parser.add_argument(
            '--normalize',
            action='store_true',
            help='Use this flag to normalize input features.'
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
            '--task',
            choices = ['classification', 'regression'],
            required=True,
            help='Type of prediction : classification or regression'
            )

    # Fold
    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    # Optional param init
    parser.add_argument(
            '--param-init',
            type=str,
            help='File of parameters initialization values'
            )

    # Comet-ml
    parser.add_argument(
            '--comet-ml',
            action='store_true',
            help='Use this flag to run experiment with comet ml'
            )

    parser.add_argument(
            '--comet-ml-project-name',
            type=str,
            help='Specific project were to send comet Experiment'
            )

    # Optimization process (do not save results)
    parser.add_argument(
            '--optimization',
            action='store_true',
            help=('Use this flag when in optimization process. '
                  '(Config files created with optimize.py).')
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
