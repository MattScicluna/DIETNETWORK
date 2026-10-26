import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import helpers.model as model


class DietNetworkHandler():

    def __init__(self, task_handler, fold, emb_filename, device,
                 dataset_filename, config, param_init):
        # Make the Dietnet model
        dn_model = model.DietNetwork(fold, emb_filename, device,
                                     dataset_filename, config,
                                     task_handler.name, param_init)
        
        # store what is needed to make simple model
        self.fold = fold
        self.emb_filename = emb_filename
        self.device = device
        self.dataset_filename = dataset_filename
        self.config = config
        self.task_handler = task_handler
        self.param_init = param_init

        #super(DietNetworkHandler, self).__init__(dn_model, task_handler)
        self.model = dn_model

    def get_exp_identifier(self, config, fold):
        exp_identifier =  self.task_handler.name \
                + '_auxu_' \
                    + str(config['nb_hidden_u_aux'])[1:-1].replace(', ','_') \
                + '_mainu_' \
                    + str(config['nb_hidden_u_aux'][-1]) + '_' \
                    + str(config['nb_hidden_u_main'])[1:-1].replace(', ','_') \
                + '_inpdrop_' + str(config['input_dropout']) \
                + '_maindrop_' + str(config['dropout_main']) \
                + '_lraux_' + str(config['lr_aux']) \
                + '_lrmain_' + str(config['lr_main']) \
                + '_lra_' + str(config['learning_rate_annealing']) \
                + '_uniform_init_limit_' + str(config['uniform_init_limit']) \
                + '_epochs_' + str(config['epochs']) \
                + '_patience_' + str(config['patience']) \
                + '_seed_' + str(config['seed']) \
                + '_fold' + str(fold)

        return exp_identifier

    def get_attribution_model(self):
        # Make simple model
        dn_model_attr = model.DietNetworkAttr(self.fold, self.emb_filename, self.device,
                                              self.dataset_filename, self.config,
                                              self.task_handler.name, self.param_init)
        self.model_attr = dn_model_attr
        return self.model_attr


class MlpHandler():

    def __init__(self, task_handler, dataset_filename, config):
        # Make the MLP model
        mlp_model = model.Mlp(task_handler, dataset_filename, config)

        #super(MlpHandler, self).__init__(mlp_model, task_handler)
        self.model = mlp_model
        self.task_handler = task_handler

    def get_exp_identifier(self, config, fold):
        exp_identifier =  self.task_handler.name \
                + '_mlp' \
                + '_layers_' \
                    + str(config['nb_hidden_u_main'])[1:-1].replace(', ','_') \
                + '_inpdrop_' + str(config['input_dropout']) \
                + '_drop_' + str(config['dropout_main']) \
                + '_lr_' + str(config['lr_main']) \
                + '_lra_' + str(config['learning_rate_annealing']) \
                + '_epochs_' + str(config['epochs']) \
                + '_patience_' + str(config['patience']) \
                + '_seed_' + str(config['seed']) \
                + '_fold' + str(fold)

        return exp_identifier
    
    def get_attribution_model(self):
        return self.model
