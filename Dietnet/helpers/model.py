import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Feat_emb_net(nn.Module):
    def __init__(self, n_feats, n_hidden_u, param_init, uniform_init_limit):
        super(Feat_emb_net, self).__init__()

        # Hidden layers
        self.hidden_layers = []
        for i in range(len(n_hidden_u)):
            # First layer
            if i == 0:
                self.hidden_layers.append(nn.Linear(n_feats, n_hidden_u[i], bias=False))
            else:
                self.hidden_layers.append(nn.Linear(n_hidden_u[i - 1], n_hidden_u[i], bias=False))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        # Parameters initialization
        if param_init is not None and len(self.hidden_layers) == 2:
            print("Initializing auxiliary network with weights from Theano")

            # Load weights from Theano
            params = np.load(param_init)

            # Init layers with Theano weights
            self.hidden_layers[0].weight = torch.nn.Parameter(torch.from_numpy(params["w1_aux"]))
            self.hidden_layers[1].weight = torch.nn.Parameter(torch.from_numpy(params["w2_aux"]))

        else:
            for layer in self.hidden_layers:
                nn.init.uniform_(layer.weight, a=-uniform_init_limit, b=uniform_init_limit)

    def forward(self, x):
        for layer in self.hidden_layers:
            ze = layer(x)
            ae = torch.tanh(ze)
            x = ae

        return ae


class Discrim_net(nn.Module):
    def __init__(
        self,
        fatLayer_weights,
        n_feats,
        n_hidden1_u,
        n_hidden2_u,
        n_targets,
        input_dropout=0.0,
        eps=1e-05,
        incl_softmax=False,
    ):
        super(Discrim_net, self).__init__()

        # Dropout on input layer
        self.input_dropout = nn.Dropout(p=input_dropout)

        # 1st hidden layer
        self.hidden_1 = nn.Linear(n_feats, n_hidden1_u)
        self.hidden_1.weight = torch.nn.Parameter(fatLayer_weights)
        nn.init.zeros_(self.hidden_1.bias)
        self.bn1 = nn.BatchNorm1d(num_features=n_hidden1_u, eps=eps)

        # 2nd hidden layer
        self.hidden_2 = nn.Linear(n_hidden1_u, n_hidden2_u)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.zeros_(self.hidden_2.bias)
        self.bn2 = nn.BatchNorm1d(num_features=n_hidden2_u, eps=eps)

        # Output layer
        self.out = nn.Linear(n_hidden2_u, n_targets)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # Dropout
        self.dropout = nn.Dropout()

        self.incl_softmax = incl_softmax

    def forward(self, x):
        # input size: batch_size x n_feats
        x = self.input_dropout(x)

        z1 = self.hidden_1(x)
        a1 = torch.relu(z1)
        a1 = self.bn1(a1)
        a1 = self.dropout(a1)

        z2 = self.hidden_2(a1)
        a2 = torch.relu(z2)
        a2 = self.bn2(a2)
        a2 = self.dropout(a2)

        out = self.out(a2)

        # Softmax will be computed in the loss
        if self.incl_softmax:
            out = torch.softmax(out, 1)

        return out


class Discrim_net2(nn.Module):
    """
    Discrim_net modified to take fatLayer_weights as a forward arg.
    Does not have weights for first layer;
    Uses F.linear with passed weights instead
    """

    def __init__(
        self,
        n_feats,
        n_hidden_u,
        n_targets,
        param_init,
        input_dropout=0.0,
        eps=1e-5,
        incl_bias=True,
        incl_softmax=False,
    ):
        super(Discrim_net2, self).__init__()

        self.hidden_layers = []
        self.bn_fatLayer = None
        self.bn = []  # batch normalization (all layers except first)
        self.out = None  # Output layer
        self.incl_softmax = incl_softmax

        # Dropout on input layer
        self.input_dropout = nn.Dropout(p=input_dropout)
        # Dropout
        self.dropout = nn.Dropout()

        # ---Layers and batchnorm (bn)  definition ---
        for i in range(len(n_hidden_u)):
            # First layer: linear function handle in forward function below
            if i == 0:
                self.bn_fatLayer = nn.BatchNorm1d(num_features=n_hidden_u[i], eps=eps)
            # Hidden layers
            else:
                self.hidden_layers.append(nn.Linear(n_hidden_u[i - 1], n_hidden_u[i]))
                self.bn.append(nn.BatchNorm1d(num_features=n_hidden_u[i], eps=eps))
            # Output layer
            self.out = nn.Linear(n_hidden_u[-1], n_targets)

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.bn = nn.ModuleList(self.bn)
        # ---Parameters initialization---
        # Theno init
        if param_init is not None and len(self.hidden_layers) == 2:
            print("Initializing main network with weights from Theano")

            # Load weights from Theano
            params = np.load(param_init)

            # Init layers with Theano weights
            self.hidden_layers[0].weight = torch.nn.Parameter(torch.from_numpy(params["w2_main"]))
            self.out.weight = torch.nn.Parameter(torch.from_numpy(params["w3_main"]))

        # Regular init
        else:
            for layer in self.hidden_layers:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.xavier_uniform_(self.out.weight)

        # ---Bias initialization---
        # Fat layer
        if incl_bias:
            self.fat_bias = nn.Parameter(data=torch.rand(n_hidden_u[0]), requires_grad=True)
            nn.init.zeros_(self.fat_bias)
        else:
            self.fat_bias = None

        # Hidden layers
        for layer in self.hidden_layers:
            nn.init.zeros_(layer.bias)

        # Output layer
        nn.init.zeros_(self.out.bias)

    def forward(self, x, fatLayer_weights, save_layers=False):
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
        for layer, bn in zip(self.hidden_layers, self.bn):
            z = layer(next_input)
            a = torch.relu(z)
            a = bn(a)
            a = self.dropout(a)
            next_input = a

        # Output layer
        out = self.out(next_input)

        # Softmax will be computed in the loss. But want this during attributions
        if self.incl_softmax:
            out = torch.softmax(out, 1)

        if save_layers:
            return next_input, out

        return out


class CombinedModel(nn.Module):
    def __init__(
        self,
        n_feats,
        n_hidden_u_aux,
        n_hidden_u_main,
        n_targets,
        param_init,
        aux_uniform_init_limit,
        input_dropout=0.0,
        eps=1e-5,
        incl_bias=True,
        incl_softmax=False,
    ):
        super(CombinedModel, self).__init__()

        # Initialize auxiliary and main networks (using old naming for compatibility)
        self.aux_net = Feat_emb_net(n_feats, n_hidden_u_aux, param_init, aux_uniform_init_limit)
        self.main_net = Discrim_net2(
            n_feats,
            n_hidden_u_main,
            n_targets,
            param_init,
            input_dropout,
            eps,
            incl_bias,
            incl_softmax,
        )
        self.fatLayer_weights = None
        # Backward-compatible attribute names expected elsewhere
        self.feat_emb = self.aux_net
        self.disc_net = self.main_net

    def forward(self, emb, x_batch, save_layers=False):
        # Forward pass in auxiliary net
        aux_net_out = self.aux_net(emb)
        # Forward pass in main net
        self.fatLayer_weights = torch.transpose(aux_net_out, 1, 0)
        main_net_out = self.main_net(x_batch, self.fatLayer_weights, save_layers)

        return main_net_out


class Mlp(nn.Module):
    def __init__(self, n_feats, n_hidden_u, n_targets, input_dropout=0.0, eps=1e-5):
        super(Mlp, self).__init__()

        self.hidden_layers = []
        self.bn = []  # batch normalization
        self.out = None  # Output layer

        # Dropout on input layer
        self.input_dropout = nn.Dropout(p=input_dropout)
        # Dropout
        self.dropout = nn.Dropout()

        # ---Layers and batchnorm (bn)  definition ---
        for i in range(len(n_hidden_u)):
            # First layer: linear function handle in forward function below
            if i == 0:
                self.hidden_layers.append(nn.Linear(n_feats, n_hidden_u[i]))
                self.bn.append(nn.BatchNorm1d(num_features=n_hidden_u[i], eps=eps))
            # Hidden layers
            else:
                self.hidden_layers.append(nn.Linear(n_hidden_u[i - 1], n_hidden_u[i]))
                self.bn.append(nn.BatchNorm1d(num_features=n_hidden_u[i], eps=eps))
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

    def forward(self, x):
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


if __name__ == "__main__":
    # Let's do a little test just to see if the run fails
    # Dummy data
    x = torch.ones((30, 3000), requires_grad=True)
    y = torch.LongTensor(30).random_(0, 2)
    x_emb = torch.ones((3000, 3 * 3), requires_grad=True)

    # Intantiate models
    emb_model = Feat_emb_net(n_feats=x_emb.size()[1], n_hidden_u=100)
    emb_model_out = emb_model(x_emb)
    fatLayer_weights = torch.transpose(emb_model_out, 1, 0)
    discrim_model = Discrim_net(
        fatLayer_weights=fatLayer_weights,
        n_feats=x.size()[1],
        n_hidden1_u=100,
        n_hidden2_u=100,
        n_targets=y.size()[0],
    )
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    params = list(discrim_model.parameters()) + list(emb_model.parameters())
    optimizer = torch.optim.SGD(params, lr=0.1)

    # Training loop
    for epoch in range(10):
        print("epoch:", epoch)
        optimizer.zero_grad()

        # Forward pass in feat emb net
        emb_out = emb_model(x_emb)

        # Set fat layer weights in discrim net
        fatLayer_weights = torch.transpose(emb_out, 1, 0)
        discrim_model.hidden_1.weight.data = fatLayer_weights

        # Forward pass in discrim net
        discrim_out = discrim_model(x)

        # Compute loss
        loss = criterion(discrim_out, y)
        print("Loss:", loss)

        # Compute gradients in discrim net
        loss.backward()
        # Copy weights of W1 in discrim net to emb_out.T
        fatLayer_weights.grad = discrim_model.hidden_1.weight.grad
        # Compute gradients in feat. emb. net
        torch.autograd.backward(fatLayer_weights, fatLayer_weights.grad)

        # Optim
        optimizer.step()
