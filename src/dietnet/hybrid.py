"""Free per-SNP weights on top of the auxiliary network: a hybrid Diet Network.

Upstream's Diet Network makes every SNP's fat-layer weights a function of that SNP's
features alone, ``W_j = aux(emb_j)``. It cannot then discount a SNP for its
neighbours in LD, which is what joint linear models (lasso, LDpred2) do. The hybrid
adds a free term,

    W_j = aux(emb_j) + delta_j,        delta: hidden x SNPs, initialised to 0,

trained by its own Adam optimiser under a group-lasso penalty
``lambda * sum_j ||delta_j||_2`` (one group per SNP, so whole SNPs are selected),
applied as a proximal step after every optimiser step (:func:`prox`). With lambda
large every delta_j stays 0 and the model is upstream's; with lambda small it
contains a penalised regression on the genotypes, the auxiliary network acting as a
prior, as annotations do in SBayesRC.

Nothing in upstream's files changes. :func:`attach` registers ``delta`` on upstream's
``DietNetwork`` instance (so it is in ``state_dict()``, checkpoints and packages) and
replaces that instance's ``forward`` with :func:`forward`, which upstream's own
``eval_step`` and ``indep_test_step`` then call as they call the original.
"""

from __future__ import annotations

import types

import torch
from torch import nn

PARAMETER = "free_snp_weights"


def fat_weights(model, cols=None) -> torch.Tensor:
    """The fat layer's weights (hidden x SNPs, or x ``cols``): the auxiliary network's
    output for those SNPs, plus their free weights if the model has them."""
    emb = model.embedding if cols is None else model.embedding[cols]
    weights = model.aux_net(emb, None, 0, 0, "", save_weights=False).T
    delta = getattr(model, PARAMETER, None)
    if delta is not None:
        weights = weights + (delta if cols is None else delta[:, cols])
    return weights


def forward(self, x_batch, results_fullpath, epoch, batch, step, save_layers=False):
    """``DietNetwork.forward`` with :func:`fat_weights` in place of the auxiliary
    network's output alone."""
    self.fatLayer_weights = fat_weights(self)
    return self.main_net(
        x_batch, self.fatLayer_weights, results_fullpath, epoch, batch, step, save_layers
    )


def attach(model) -> nn.Parameter:
    """Give upstream's ``DietNetwork`` ``model`` free per-SNP weights, all 0: the model
    computes exactly what it did. Returns the new parameter."""
    if hasattr(model, PARAMETER):
        return getattr(model, PARAMETER)
    n_snps = model.embedding.shape[0]
    hidden = model.aux_net.hidden_layers[-1].out_features
    delta = nn.Parameter(torch.zeros(hidden, n_snps, device=model.embedding.device))
    model.register_parameter(PARAMETER, delta)
    model.forward = types.MethodType(forward, model)
    return delta


@torch.no_grad()
def prox(delta: torch.Tensor, threshold: float) -> None:
    """The group-lasso proximal step, in place: each SNP's column delta_j becomes
    ``delta_j * max(0, 1 - threshold / ||delta_j||)``, so a column shorter than the
    threshold becomes exactly 0. ``threshold`` is the step size times lambda."""
    if threshold <= 0:
        return
    norms = delta.norm(dim=0)
    scale = torch.clamp(1 - threshold / norms.clamp_min(torch.finfo(delta.dtype).tiny), min=0)
    delta.mul_(scale)


def nonzero(delta: torch.Tensor) -> int:
    """SNPs with a non-zero free weight."""
    return int((delta.detach().abs().sum(dim=0) > 0).sum())
