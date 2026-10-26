import os
import sys
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers.task_handlers import ClassificationHandler, RegressionHandler


class AuxiliaryNetwork(nn.Module):
    def __init__(self, n_feats_emb, config, param_init):
        super(AuxiliaryNetwork, self).__init__()

        # Hidden layers : self.hidden layers
        nb_hidden_u = config['nb_hidden_u_aux']
        uniform_init_limit = config['uniform_init_limit']
        self.hidden_layers = []
        
        # Theano values for patam init
        if param_init is not None:
            params = np.load(param_init)
        
        for i in range(len(nb_hidden_u)):
            # First layer
            if i == 0:
                self.hidden_layers.append(
                        nn.Linear(n_feats_emb, nb_hidden_u[i], bias=False)
                        )
                if param_init is not None:
                    self.hidden_layers[-1].weight = torch.nn.Parameter(
                        torch.from_numpy(params['w1_aux']))
                else:
                    nn.init.uniform_(self.hidden_layers[-1].weight, a=-uniform_init_limit, b=uniform_init_limit)
            
            # Other layers
            else:
                self.hidden_layers.append(
                        nn.Linear(nb_hidden_u[i-1], nb_hidden_u[i], bias=False)
                        )
                if param_init is not None:
                    self.hidden_layers[-1].weight = torch.nn.Parameter(
                        torch.from_numpy(params['w2_aux']))
                else:
                    nn.init.uniform_(self.hidden_layers[-1].weight, a=-uniform_init_limit, b=uniform_init_limit)

        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        # Parameters initialization with Theno values or Xavier glorot init method
        # This was moved aboved when layers are defined
        """
        if param_init is not None and len(self.hidden_layers)==2:
            print('Initializing auxiliary network with weights from Theano')

            # Load weights from Theano
            params = np.load(param_init)

            # Init layers with Theano weights
            self.hidden_layers[0].weight = torch.nn.Parameter(
                    torch.from_numpy(params['w1_aux']))
            self.hidden_layers[1].weight = torch.nn.Parameter(
                    torch.from_numpy(params['w2_aux']))
                    
        else:
            uniform_init_limit = config['uniform_init_limit']
            for layer in self.hidden_layers:
                nn.init.uniform_(layer.weight, a=-uniform_init_limit, b=uniform_init_limit)
        """


    def forward(self, x, results_fullpath, epoch, batch, step, save_weights):
        for i,layer in enumerate(self.hidden_layers):
            # Save layer params
            """
            if ((batch == 0) and (step == 'valid')):
                # Save layer weights
                filename = 'auxLayer_'+str(i)+'_weights_epoch'+str(epoch)+'_batch'+str(batch)
                np.savez(os.path.join(results_fullpath, filename),
                         weights=layer.weight.detach().cpu())
                # Save layer bias
                filename = 'auxLayer_'+str(i)+'_bias_epoch'+str(epoch)+'_batch'+str(batch)
                if layer.bias is not None:
                    np.savez(os.path.join(results_fullpath, filename),
                             bias=layer.bias.detach().cpu())
                else:
                    np.savez(os.path.join(results_fullpath, filename),
                             bias=np.array([]))
            """

            
            # Forward pass
            ze = layer(x)
            ae = torch.tanh(ze)
            x = ae

        return ae


class MainNetwork(nn.Module):
    """
    Discrim_net modified to take fatLayer_weights as a forward arg.
    Does not have weights for first layer;
    Uses F.linear with passed weights instead
    """
    def __init__(self, n_feats, n_targets, config, param_init,
                 input_dropout=0., eps=1e-5, incl_bias=True, incl_softmax=False):
        super(MainNetwork, self).__init__()
        
        self.hidden_layers = []
        self.bn_fatLayer = None
        self.bn = [] # batch normalization
        self.out = None # Output layer
        self.incl_softmax = incl_softmax
        
        # Dropout on input layer
        self.input_dropout = nn.Dropout(config['input_dropout'])

        # Dropout
        self.dropout = nn.Dropout(config['dropout_main'])

        # --- Layers and batchnorm ---
        nb_hidden_u = config['nb_hidden_u_aux'][-1:] + config['nb_hidden_u_main']
        if param_init is not None:
            params = np.load(param_init)
        
        for i in range(len(nb_hidden_u)):
            # First layer: linear function handle in forward function below
            if i == 0:
                self.bn_fatLayer = nn.BatchNorm1d(num_features=nb_hidden_u[i], eps=eps)
            
            # Hidden layers
            else:
                self.hidden_layers.append(
                        nn.Linear(nb_hidden_u[i-1], nb_hidden_u[i]))
                self.bn.append(
                        nn.BatchNorm1d(num_features=nb_hidden_u[i], eps=eps))
                
                # Layer init
                if param_init is not None:
                    self.hidden_layers[-1].weight = torch.nn.Parameter(
                        torch.from_numpy(params['w2_main']))
                else:
                    nn.init.xavier_uniform_(self.hidden_layers[-1].weight)
                
                nn.init.zeros_(self.hidden_layers[-1].bias)
            
            # Output layer : pas la bonne place (une indentation de trop)
            #self.out = nn.Linear(nb_hidden_u[-1], n_targets)
        
        # Output layer
        self.out = nn.Linear(nb_hidden_u[-1], n_targets)
        # Init of output layer
        if param_init is not None:
            self.out.weight = torch.nn.Parameter(
                torch.from_numpy(params['w3_main']))
        else:
            nn.init.xavier_uniform_(self.out.weight)
        
        nn.init.zeros_(self.out.bias)

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.bn = nn.ModuleList(self.bn)

        # ---Parameters initialization---
        # Theno init
        # This was moved above with layers definition
        """
        if param_init is not None and len(self.hidden_layers)==2:
            print('Initializing main network with weights from Theano')

            # Load weights from Theano
            params = np.load(param_init)

            # Init layers with Theano weights
            self.hidden_layers[0].weight = torch.nn.Parameter(
                    torch.from_numpy(params['w2_main']))
            self.out.weight = torch.nn.Parameter(
                    torch.from_numpy(params['w3_main']))
        """

        # Regular init
        # this was moved above in the layer creation
        """
        else:
            for layer in self.hidden_layers:
                nn.init.xavier_uniform_(layer.weight)

            nn.init.xavier_uniform_(self.out.weight)
        """

        # ---Bias initialization---
        # Fat layer
        if incl_bias:
            self.fat_bias = nn.Parameter(data=torch.rand(nb_hidden_u[0]), requires_grad=True)
            nn.init.zeros_(self.fat_bias)
        else:
            self.fat_bias = None

        # Hidden layers
        for layer in self.hidden_layers:
            nn.init.zeros_(layer.bias)

        # Output layer
        nn.init.zeros_(self.out.bias)


    def forward(self, x, fatLayer_weights, results_fullpath, epoch, batch,
                step, save_layers=False):
        # input size: batch_size x n_feats
        # weight = comes from feat embedding net
        # now ^^^ is passed with forward
        
        x = self.input_dropout(x)

        # Fat layer
        z1 = F.linear(x, fatLayer_weights, bias=self.fat_bias)
        a1 = torch.relu(z1)
        a1 = self.bn_fatLayer(a1)
        a1 = self.dropout(a1)

        # Hidden layers
        next_input = a1
        for i,(layer, bn) in enumerate(zip(self.hidden_layers, self.bn)):
            # Save layer params
            
            #if ((batch == 0) and (step == 'valid')):
            #    # Save layer weights
            #    filename = 'mainLayer_'+str(i)+'_weights_epoch'+str(epoch)+'_batch'+str(batch)
            #    np.savez(os.path.join(results_fullpath, filename),
            #             weights=layer.weight.detach().cpu())
            #    # Save layer bias
            #    filename = 'mainLayer_'+str(i)+'_bias_epoch'+str(epoch)+'_batch'+str(batch)
            #    np.savez(os.path.join(results_fullpath, filename),
            #             bias=layer.bias.detach().cpu())
            

            # Forward pass
            z = layer(next_input)
            a = torch.relu(z)
            a = bn(a)
            a = self.dropout(a)
            next_input = a

        # Output layer
        
        #if ((batch == 0) and (step == 'valid')):
        #    filename = 'mainLayer_out_weights_epoch'+str(epoch)+'_batch'+str(batch)
        #    np.savez(os.path.join(results_fullpath, filename),
        #             weights=self.out.weight.detach().cpu())
        #    filename = 'mainLayer_out_bias_epoch'+str(epoch)+'_batch'+str(batch)
        #    np.savez(os.path.join(results_fullpath, filename),
        #             bais=self.out.bias.detach().cpu())
        
        out = self.out(next_input)

        # Softmax will be computed in the loss. But want this during attributions
        if self.incl_softmax:
            out = torch.softmax(out, 1)

        if save_layers:
            return next_input, out

        return out


class DietNetwork(nn.Module):
    def __init__(self, fold, emb_filename, device,
                 dataset_filename, config, task, param_init,
                 input_dropout=0., eps=1e-5, incl_bias=True, incl_softmax=False):
        super(DietNetwork, self).__init__()

        self.fatLayer_weights = None

        # ----------------------------------------
        #               EMBEDDING
        # ----------------------------------------
        # Load embedding
        emb = np.load(emb_filename)['emb']
        if len(emb.shape) == 3:
            # One embedding per fold
            emb = np.load(emb_filename)['emb'][fold]
        elif len(emb.shape) == 2:
            # Same embedding for every fold
            emb = np.load(emb_filename)['emb']

        # Send to device
        emb = torch.from_numpy(emb)
        emb = emb.to(device).float()

        # Normalize embedding
        emb_norm = (emb ** 2).sum(0) ** 0.5
        emb = emb/emb_norm

        # Send to device
        self.embedding = emb
        print('Embedding size: {}'.format(emb.size()))

        # ----------------------------------------
        #           AUXILIARY NETWORK
        # ----------------------------------------
        # Auxiliary network input size
        if len(emb.size()) == 1:
            n_feats_emb = 1 # 1 value per SNP
            emb = torch.unsqueeze(emb, dim=1)
        else:
            n_feats_emb = emb.size()[1]

        # Instantiate auxiliary network
        self.aux_net = AuxiliaryNetwork(n_feats_emb, config, param_init)

        # ----------------------------------------
        #               MAIN NETWORK
        # ----------------------------------------
        # Main network input size
        n_feats = emb.size()[0]

        # Main Network output size (nb targets)
        if task == 'regression':
            n_targets = 1
        elif task == 'classification':
            with h5py.File(dataset_filename, 'r') as f:
                n_targets = len(f['class_label_names'])

        # Instantiate main network
        self.main_net = MainNetwork(n_feats, n_targets, config, param_init)
        

        # ----------------------------------------
        #               OPTIMIZERS
        # ----------------------------------------
        # Optimizer for auxiliary net
        self.aux_net.optimizer = torch.optim.Adam(
            self.aux_net.parameters(),
            lr=config['lr_aux'])
        
        # Optimizer for main net
        self.main_net.optimizer = torch.optim.Adam(
            self.main_net.parameters(),
            lr=config['lr_main']
        )


    def forward(self, x_batch, results_fullpath,
                epoch, batch, step, save_layers=False):
        # Forward pass in auxilliary net
        aux_net_out = self.aux_net(self.embedding, results_fullpath,
                                   epoch, batch, step, save_weights=False)

        # Forward pass in discrim net
        self.fatLayer_weights = torch.transpose(aux_net_out,1,0)
        main_net_out = self.main_net(x_batch, self.fatLayer_weights,
                                     results_fullpath,
                                     epoch, batch, step, save_layers)

        return main_net_out


    def get_optimizers(self):
        return self.aux_net.optimizer, self.main_net.optimizer
    
    
    def save_parameters(self, filename):
        print(self.aux_net.hidden_layers)
        print(self.fatLayer_weights.size())
        print(self.main_net.hidden_layers)

        
class DietNetworkAttr(DietNetwork):
    def __init__(self, fold, emb_filename, device,
                 dataset_filename, config, task, param_init,
                 input_dropout=0., eps=1e-5, incl_bias=True, incl_softmax=False):
        super(DietNetworkAttr, self).__init__(fold, emb_filename, device, dataset_filename, config, task, param_init, input_dropout, eps, incl_bias, incl_softmax)
        self.results_fullpath = 'Dummy'

    def forward(self, x_batch):
        # Forward pass in auxilliary net
        aux_net_out = self.aux_net(self.embedding, self.results_fullpath,
                                   1, 1, 1, save_weights=False)

        # Forward pass in discrim net
        self.fatLayer_weights = torch.transpose(aux_net_out,1,0)
        main_net_out = self.main_net(x_batch, self.fatLayer_weights,
                                     self.results_fullpath,
                                     1, 1, 1, False)

        return main_net_out


    def save_parameters(self, filename):
        print(self.aux_net.hidden_layers)
        print(self.fatLayer_weights.size())
        print(self.main_net.hidden_layers)



class Mlp(nn.Module):
    def __init__(self, task_handler, dataset_filename, config, input_dropout=0., eps=1e-5):
        super(Mlp, self).__init__()

        with h5py.File(dataset_filename, 'r') as f:
            # Number of input features
            n_feats = len(f['snp_names'])

            # Number of targets (network output size)
            if task_handler.name == 'regression':
                n_targets = 1
            elif task_handler.name == 'classification':
                n_targets = len(f['class_label_names'])

        # Layers to be defined
        self.hidden_layers = []
        self.bn = [] # batch normalization
        self.out = None # Output layer

        # Dropout on input layer
        self.input_dropout = nn.Dropout(p=input_dropout)

        # Dropout
        self.dropout = nn.Dropout(config['dropout_main'])

        # Layers definition
        n_hidden_u = config['nb_hidden_u_main'] # nb of hidden units per layer
        for i in range(len(n_hidden_u)):
            # First layer
            if i == 0:
                self.hidden_layers.append(
                        nn.Linear(n_feats, n_hidden_u[i]))
                self.bn.append(
                        nn.BatchNorm1d(num_features=n_hidden_u[i], eps=eps))
            # Hidden layers
            else:
                self.hidden_layers.append(
                        nn.Linear(n_hidden_u[i-1], n_hidden_u[i]))
                self.bn.append(
                        nn.BatchNorm1d(num_features=n_hidden_u[i], eps=eps))
            # Output layer
            self.out = nn.Linear(n_hidden_u[-1], n_targets)

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.bn = nn.ModuleList(self.bn)

        # ---Parameters initialization---
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)

        nn.init.xavier_uniform_(self.out.weight)

        # ---Bias initialization---
        for layer in self.hidden_layers:
            nn.init.zeros_(layer.bias)

        # Output layer
        nn.init.zeros_(self.out.bias)

        # ----------------------------------------
        #               OPTIMIZER
        # ----------------------------------------
        # Optimizer 
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config['lr_main'])

    
    def forward(self, x, results_fullpath, epoch, batch, step):
        x = self.input_dropout(x)

        next_input = x
        for layer, bn in zip(self.hidden_layers, self.bn):
            z = layer(next_input)
            a = torch.relu(z)
            a = bn(a)
            a = self.dropout(a)
            next_input = a

        # Output layer
        out = self.out(next_input)

        return out
    
    
    def get_optimizers(self):
        # We return the optimizer in a list because 
        # in train step we iterate over optimizers 
        # (because the DN model has more than one
        # optimizer, since the aux and main nets have
        # different lr)
        return [self.optimizer]
