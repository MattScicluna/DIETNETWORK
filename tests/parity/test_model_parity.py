"""The model and training loop compute what upstream's compute.

Both models are built, upstream's weights are copied into this repository's,
and then the two are run side by side on identical batches from an identical
RNG state. Any difference is then a difference in the code, not in the draw.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from support import stratified_folds
from torch.utils.data import DataLoader, Dataset

from Dietnet.generate_embedding import compute_fold_embedding
from Dietnet.helpers import dataset_utils as du
from Dietnet.helpers import mainloop_utils as mlu
from Dietnet.helpers import model

pytestmark = pytest.mark.upstream

BASE_CONFIG = {
    "nb_hidden_u_aux": [16, 16],
    "nb_hidden_u_main": [16],
    "uniform_init_limit": 0.02,
    "input_dropout": 0.9,
    "lr_aux": 6e-4,
    "lr_main": 3e-4,
}


class Rows(Dataset):
    """(genotypes, label, row index): the triple both training loops expect."""

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], i


class Pair:
    """Upstream's DietNetwork and ours, holding the same weights."""

    def __init__(self, upstream, dataset, tmp_path, dropout_main):
        genotypes, labels, _ = dataset.upstream_arrays()
        train = stratified_folds(labels)[0][0]
        self.x, self.y = genotypes[train], labels[train]
        self.n_targets = int(labels.max()) + 1

        emb = compute_fold_embedding(self.x, self.y)
        emb_file = tmp_path / "embedding.npz"
        np.savez(emb_file, emb=emb[None])

        self.config = {**BASE_CONFIG, "dropout_main": dropout_main}
        self.task = upstream["helpers.task_handlers"].ClassificationHandler(
            str(dataset.upstream_hdf5), nn.CrossEntropyLoss()
        )
        torch.manual_seed(0)
        self.handler = upstream["helpers.model_handlers"].DietNetworkHandler(
            self.task, 0, str(emb_file), "cpu", str(dataset.upstream_hdf5), self.config, None
        )
        self.theirs = self.handler.model

        # Normalised as train.py does before handing the embedding to the model.
        emb = torch.from_numpy(emb).float()
        self.emb = emb / (emb**2).sum(0) ** 0.5

        self.ours = model.CombinedModel(
            n_feats=self.emb.shape[1],
            n_hidden_u_aux=self.config["nb_hidden_u_aux"],
            n_hidden_u_main=self.config["nb_hidden_u_aux"][-1:] + self.config["nb_hidden_u_main"],
            n_targets=self.n_targets,
            param_init=None,
            aux_uniform_init_limit=self.config["uniform_init_limit"],
            input_dropout=self.config["input_dropout"],
        )
        self.ours.load_state_dict(with_aliases(self.theirs.state_dict()), strict=True)

        x = torch.from_numpy(self.x.astype(np.float32))
        self.mus, self.sigmas = du.compute_norm_values(x)

    def batches(self, batch_size=32):
        return DataLoader(Rows(self.x, self.y), batch_size=batch_size, shuffle=False)


def with_aliases(state):
    """Upstream's state dict plus the duplicate keys CombinedModel also expects (B12)."""
    out = dict(state)
    for key, value in state.items():
        for name, alias in (("aux_net.", "feat_emb."), ("main_net.", "disc_net.")):
            if key.startswith(name):
                out[alias + key[len(name) :]] = value
    return out


def test_forward_pass_matches_upstream(upstream, dataset, tmp_path):
    pair = Pair(upstream, dataset, tmp_path, dropout_main=0.5)
    x = torch.from_numpy(pair.x[:64].astype(np.float32))
    x = du.normalize(x, pair.mus, pair.sigmas)

    pair.theirs.eval()
    pair.ours.eval()
    with torch.no_grad():
        theirs = pair.theirs(x, None, 0, 0, "valid")
        ours = pair.ours(pair.emb, x)

    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "dropout_main",
    [
        0.5,
        pytest.param(
            0.1,
            marks=pytest.mark.xfail(
                reason="B1: dropout_main is ignored; Discrim_net2 hardcodes nn.Dropout() (p=0.5)"
            ),
        ),
    ],
)
def test_one_training_epoch_matches_upstream(upstream, synthetic, tmp_path, dropout_main):
    """Same batches, same RNG state: same weights after an epoch of Adam.

    The learning rates differ between the networks on purpose, so that a mix-up
    between upstream's two optimizers and our two parameter groups would show.
    """
    pair = Pair(upstream, synthetic, tmp_path, dropout_main=dropout_main)

    torch.manual_seed(1)
    pair.theirs.train()
    upstream["helpers.mainloop_utils"].train_step(
        pair.handler,
        "cpu",
        Rows(pair.x, pair.y),
        pair.batches(),
        pair.mus,
        pair.sigmas,
        True,
        None,
        0,
    )

    optimizer = torch.optim.Adam(
        [
            {"params": pair.ours.aux_net.parameters(), "lr": pair.config["lr_aux"]},
            {"params": pair.ours.main_net.parameters(), "lr": pair.config["lr_main"]},
        ]
    )
    torch.manual_seed(1)
    pair.ours.train()
    mlu.train_step(
        pair.ours,
        "cpu",
        optimizer,
        pair.batches(),
        len(pair.y),
        nn.CrossEntropyLoss(),
        pair.mus,
        pair.sigmas,
        pair.emb,
        "classification",
        True,
    )

    ours = pair.ours.state_dict()
    for key, theirs in pair.theirs.state_dict().items():
        torch.testing.assert_close(ours[key], theirs, rtol=1e-4, atol=1e-6, msg=key)


def epoch_results(loss, acc, n=100):
    """An epoch's results in the shape upstream's task handler consumes."""
    return {
        "samples": np.arange(n),
        "losses_wo_reduction": np.array([loss * n]),
        "n_right": np.array([acc * n / 100]),
    }


def improvements(upstream, synthetic, epochs):
    """Which epochs each implementation would checkpoint, given (loss, acc) per epoch."""
    handler = upstream["helpers.task_handlers"].ClassificationHandler(
        str(synthetic.upstream_hdf5), nn.CrossEntropyLoss()
    )
    handler.init_best_epoch_results(epoch_results(*epochs[0]))
    theirs = [handler.update_best_results(epoch_results(*e)) for e in epochs[1:]]

    best, ours = epochs[0], []
    for epoch in epochs[1:]:
        improved = mlu.has_improved(best, epoch)
        ours.append(improved)
        if improved:
            best = mlu.update_best_result(best, epoch)
    return ours, theirs


def test_best_model_selection_agrees_when_accuracy_decides(upstream, synthetic):
    ours, theirs = improvements(
        upstream, synthetic, [(1.0, 50.0), (0.9, 55.0), (1.2, 52.0), (0.8, 60.0)]
    )
    assert ours == theirs == [True, False, True]


@pytest.mark.xfail(
    reason="B3: update_best_result keeps the lowest loss of any epoch, not the loss of "
    "the best-accuracy epoch, so a later tie on accuracy is judged against the wrong loss"
)
def test_best_model_selection_agrees_on_accuracy_ties(upstream, synthetic):
    """Accuracy rises while loss gets worse, then a later epoch ties that accuracy.

    Upstream checkpoints the tie if its loss beats the loss of the epoch that set
    the accuracy record (2.0). Ours compares it with the baseline's 1.0.
    """
    ours, theirs = improvements(upstream, synthetic, [(1.0, 50.0), (2.0, 60.0), (1.5, 60.0)])
    assert ours == theirs
