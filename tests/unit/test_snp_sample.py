"""--snp-sample: a Diet Network trained on a shared random share of the SNPs per batch."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dietnet import genotypes
from dietnet.parallel import ChunkBuffers, ChunkPlan, SnpSampled, eval_batch_size, plan_columns


def test_a_chunk_plan_covers_the_rows_and_shares_a_block_choice_per_buffer():
    train = np.random.default_rng(0).permutation(10_000)[:6_000]
    plan = ChunkPlan(train, 1_003, 100, 4, 16, 0.5, seed=1)
    buffers = list(plan)
    assert len(buffers) == len(plan) == 15
    seen = np.concatenate([rows for rows, _ in buffers])
    assert len(set(seen)) == len(seen) and set(seen) <= set(train)
    assert len(seen) == 15 * 400  # equal buffers, every row of these chunks
    for rows, ranges in buffers:
        assert (np.diff(rows) > 0).all()  # read in file order
        assert len(ranges) == 8
        assert (ranges[:-1, 1] <= ranges[1:, 0]).all()
    # whole bytes of the 2-bit copy, except the file's last SNP
    assert all(a % 4 == 0 for a, _ in plan.blocks) and plan.blocks[-1, 1] == 1_003
    plan.set_epoch(1)
    again = list(plan)
    assert not all(np.array_equal(a[0], b[0]) for a, b in zip(buffers, again))


def test_ranks_get_as_many_buffers_of_the_same_size():
    train = np.arange(10_007)
    plans = [ChunkPlan(train, 400, 64, 4, 8, 0.5, seed=1, rank=r, world=3) for r in range(3)]
    assert len({len(p) for p in plans}) == 1
    sizes = {len(rows) for p in plans for rows, _ in p}
    assert len(sizes) == 1
    rank0, rank1 = [np.concatenate([r for r, _ in p]) for p in plans[:2]]
    assert not set(rank0) & set(rank1)


def test_a_buffer_decodes_to_the_kept_columns_of_the_store():
    """Packed bytes of the kept blocks decode to exactly those SNPs, whichever blocks,
    including the last one, whose final byte is part padding."""
    rng = np.random.default_rng(2)
    n_snps = 1_003
    x = rng.integers(-1, 3, size=(50, n_snps)).astype(np.int8)
    packed = genotypes.encode(x)

    class Store:
        def __init__(self):
            self.reader = packed

    labels = np.arange(50.0)
    plan = ChunkPlan(np.arange(50), n_snps, 10, 2, 8, 0.5, seed=3)
    table = torch.from_numpy(genotypes.DECODE_TABLE)
    for rows, ranges in plan:
        got, y, cols = ChunkBuffers(Store(), labels, packed=True)[(rows, ranges)]
        decoded = table[got.long()].reshape(len(got), -1)[:, : len(cols)]
        assert np.array_equal(cols.numpy(), plan_columns(ranges))
        assert np.array_equal(decoded.numpy(), x[rows][:, cols.numpy()])
        assert np.array_equal(y.numpy(), labels[rows])


def test_given_columns_use_those_snps_scaled():
    torch.manual_seed(5)
    net, x = Net(), torch.randn(8, 50)
    net.train()
    cols = torch.tensor([0, 1, 2, 3, 20, 21, 22, 23, 40, 41])
    out = SnpSampled(net, 0.2)(x[:, cols], "", 0, 0, "train", cols=cols)
    mask = torch.zeros(50)
    mask[cols] = 5.0
    assert torch.allclose(out, net(x * mask, "", 0, 0, "train"), atol=1e-6)


def test_evaluation_batches_fit_a_gpu_at_any_number_of_snps():
    assert eval_batch_size(138, 416_084) == 2048  # the array: as before
    assert eval_batch_size(138, 1_290_149) == 832
    assert eval_batch_size(4096, 416_084) == 4096  # never below a training batch
    assert eval_batch_size(138, 40_000_000) == 138


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
