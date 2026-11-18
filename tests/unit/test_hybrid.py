"""Free per-SNP weights (dietnet.hybrid): attach, the forward pass, the proximal step."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dietnet import hybrid


class Aux(nn.Module):
    """Upstream's AuxiliaryNetwork in miniature: tanh layers without bias."""

    def __init__(self, n_feats, hidden):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(n_feats, hidden, bias=False)])

    def forward(self, x, results, epoch, batch, step, save_weights):
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        return x


class Main(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fat_bias = nn.Parameter(torch.rand(hidden))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, weights, results, epoch, batch, step, save_layers=False):
        return self.out(torch.relu(F.linear(x, weights, bias=self.fat_bias)))


class Net(nn.Module):
    """Upstream's DietNetwork.forward."""

    def __init__(self, n_snps=40, n_feats=3, hidden=5):
        super().__init__()
        self.embedding = torch.randn(n_snps, n_feats)
        self.aux_net, self.main_net = Aux(n_feats, hidden), Main(hidden)

    def forward(self, x, results, epoch, batch, step, save_layers=False):
        aux_out = self.aux_net(self.embedding, results, epoch, batch, step, save_weights=False)
        self.fatLayer_weights = torch.transpose(aux_out, 1, 0)
        return self.main_net(x, self.fatLayer_weights, results, epoch, batch, step, save_layers)


def test_attached_weights_start_at_zero_and_change_nothing():
    torch.manual_seed(0)
    net, x = Net(), torch.randn(6, 40)
    before = net(x, "", 0, 0, "valid")
    delta = hybrid.attach(net)
    assert delta.shape == (5, 40) and not delta.any()
    assert torch.equal(net(x, "", 0, 0, "valid"), before)  # bit for bit
    assert hybrid.PARAMETER in net.state_dict()
    assert hybrid.attach(net) is delta  # attaching twice is a no-op


def test_the_fat_layer_is_the_auxiliary_output_plus_the_free_weights():
    torch.manual_seed(1)
    net, x = Net(), torch.randn(6, 40)
    delta = hybrid.attach(net)
    with torch.no_grad():
        delta.normal_()
    aux = net.aux_net(net.embedding, "", 0, 0, "", save_weights=False).T
    expected = net.main_net(x, aux + delta, "", 0, 0, "")
    assert torch.allclose(net(x, "", 0, 0, "valid"), expected)
    cols = torch.tensor([1, 5, 7])
    assert torch.allclose(hybrid.fat_weights(net, cols), aux[:, cols] + delta[:, cols])


def test_the_proximal_step_is_group_soft_thresholding():
    """Each column delta_j: 0 if ||delta_j|| <= t, else shrunk by t along itself."""
    torch.manual_seed(2)
    delta = torch.randn(4, 30) * torch.linspace(0.1, 3, 30)
    original = delta.clone()
    hybrid.prox(delta, 2.0)
    norms = original.norm(dim=0).numpy()
    for j in range(30):
        if norms[j] <= 2.0:
            assert not delta[:, j].any()
        else:
            np.testing.assert_allclose(
                delta[:, j].numpy(), original[:, j].numpy() * (1 - 2.0 / norms[j]), rtol=1e-6
            )
    assert hybrid.nonzero(delta) == int((norms > 2.0).sum())
    untouched = original.clone()
    hybrid.prox(untouched, 0.0)
    assert torch.equal(untouched, original)


def test_per_snp_thresholds_scale_each_columns_penalty():
    """A threshold per SNP: each column is soft-thresholded by its own; all equal is the
    scalar step, and 0 leaves a column as it was."""
    torch.manual_seed(3)
    delta = torch.randn(4, 30) * torch.linspace(0.1, 3, 30)
    weights = torch.rand(30) * 2
    weights[:5] = 0
    thresholds = 2.0 * weights
    shrunk = delta.clone()
    hybrid.prox(shrunk, thresholds)
    norms = delta.norm(dim=0)
    expected = delta * torch.clamp(1 - thresholds / norms, min=0)
    torch.testing.assert_close(shrunk, expected)
    assert torch.equal(shrunk[:, :5], delta[:, :5])

    scalar, vector = delta.clone(), delta.clone()
    hybrid.prox(scalar, 2.0)
    hybrid.prox(vector, torch.full((30,), 2.0))
    torch.testing.assert_close(vector, scalar)


def test_penalty_weights_are_checked(tmp_path):
    """One finite, non-negative weight per SNP, returned as float32."""
    import pytest

    good = tmp_path / "good.npy"
    np.save(good, np.array([0.0, 0.5, 2.0]))
    loaded = hybrid.load_penalty_weights(good, 3)
    assert loaded.dtype == np.float32 and loaded.tolist() == [0.0, 0.5, 2.0]
    with pytest.raises(ValueError, match=r"\(3,\) penalty weights, expected \(4,\)"):
        hybrid.load_penalty_weights(good, 4)
    for bad in ([0.0, -1.0, 1.0], [0.0, np.nan, 1.0], [0.0, np.inf, 1.0]):
        np.save(path := tmp_path / "bad.npy", np.array(bad))
        with pytest.raises(ValueError, match="finite and non-negative"):
            hybrid.load_penalty_weights(path, 3)
    np.save(path := tmp_path / "matrix.npy", np.ones((3, 1)))
    with pytest.raises(ValueError, match="expected"):
        hybrid.load_penalty_weights(path, 3)
