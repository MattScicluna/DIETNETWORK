"""--snp-sample: a Diet Network trained on a shared random share of the SNPs per batch."""

import torch
from torch import nn
from torch.nn import functional as F

from dietnet.parallel import SnpSampled


class Aux(nn.Module):
    def __init__(self, n_feats, hidden):
        super().__init__()
        self.layer = nn.Linear(n_feats, hidden, bias=False)

    def forward(self, x, results, epoch, batch, step, save_weights):
        return torch.tanh(self.layer(x))


class Main(nn.Module):
    """Upstream's main network in miniature: input dropout, then the fat layer."""

    def __init__(self, hidden):
        super().__init__()
        self.input_dropout = nn.Identity()
        self.fat_bias = nn.Parameter(torch.rand(hidden))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, weights, results, epoch, batch, step):
        x = self.input_dropout(x)
        return self.out(torch.relu(F.linear(x, weights, bias=self.fat_bias)))


class Net(nn.Module):
    """Upstream's DietNetwork.forward: fat-layer weights from the whole embedding."""

    def __init__(self, n_snps=50, n_feats=4, hidden=6):
        super().__init__()
        self.embedding = torch.randn(n_snps, n_feats)
        self.aux_net, self.main_net = Aux(n_feats, hidden), Main(hidden)

    def forward(self, x, results, epoch, batch, step):
        weights = self.aux_net(self.embedding, results, epoch, batch, step, False).T
        return self.main_net(x, weights, results, epoch, batch, step)


def test_evaluation_and_keep_one_are_the_plain_model():
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 50)
    plain = net(x, "", 0, 0, "valid")
    net.eval()
    assert torch.equal(SnpSampled(net, 0.5)(x, "", 0, 0, "valid"), net(x, "", 0, 0, "valid"))
    net.train()
    assert torch.equal(SnpSampled(net, 1.0)(x, "", 0, 0, "train"), plain)


def test_a_training_batch_sees_a_scaled_shared_subset_of_snps():
    """The weights of the sampled SNPs only, from the auxiliary network on their rows,
    and their columns scaled by n / k: what the model gives with the other SNPs'
    inputs zeroed and the rest scaled, as one shared dropout mask would."""
    torch.manual_seed(1)
    net, x = Net(), torch.randn(8, 50)
    net.train()
    torch.manual_seed(2)
    out = SnpSampled(net, 0.4)(x, "", 0, 0, "train")

    torch.manual_seed(2)
    cols = torch.randperm(50)[:20].sort().values
    mask = torch.zeros(50)
    mask[cols] = 50 / 20
    assert torch.allclose(out, net(x * mask, "", 0, 0, "train"), atol=1e-6)


def test_gradients_reach_the_sampled_snps_auxiliary_rows_only():
    """The point of sharing the mask: SNPs left out of the batch cost nothing, so the
    auxiliary network's input rows for them get no gradient through the fat layer."""
    torch.manual_seed(3)
    net, x = Net(), torch.randn(8, 50)
    net.embedding.requires_grad_(True)
    net.train()
    torch.manual_seed(4)
    SnpSampled(net, 0.3)(x, "", 0, 0, "train").sum().backward()
    torch.manual_seed(4)
    cols = torch.randperm(50)[:15]
    touched = net.embedding.grad.abs().sum(1) > 0
    assert set(touched.nonzero().flatten().tolist()) <= set(cols.tolist())
