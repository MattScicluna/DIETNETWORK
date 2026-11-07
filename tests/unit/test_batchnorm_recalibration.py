"""Batch-norm statistics recomputed without dropout (dietnet.parallel.recalibrate_batchnorm).

Upstream's network batch-normalises the fat layer right after input dropout, so
the running statistics it learns in training describe activations with dropout,
and evaluation applies them to activations without it.
"""

import pytest
import torch
from torch import nn

from dietnet import training
from dietnet.parallel import recalibrate_batchnorm


def test_statistics_become_those_of_the_inputs_without_dropout():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Dropout(0.5), nn.Linear(20, 8), nn.ReLU(), nn.BatchNorm1d(8))
    model.train()
    for _ in range(5):  # training statistics, with dropout
        model(torch.randn(64, 20) * 3)
    batches = [torch.randn(50, 20) for _ in range(4)]

    seen = recalibrate_batchnorm(model, iter(batches), model)

    bn = model[3]
    with torch.no_grad():
        acts = [model[2](model[1](x)) for x in batches]  # no dropout
    torch.testing.assert_close(bn.running_mean, torch.cat(acts).mean(0))
    torch.testing.assert_close(bn.running_var, torch.stack([a.var(0) for a in acts]).mean(0))
    assert seen == 200 and bn.momentum == 0.1 and not model.training


def test_the_upstream_engine_refuses_it():
    with pytest.raises(ValueError, match="needs engine 'dietnet'"):
        training.train(
            dataset="d.h5", partition="p.npz", embedding="e.npz", input_stats="s.npz",
            config="c.yaml", exp_dir="x", seeds=[1], folds=[0],
            engine="upstream", bn_recalibrate=100,
        )  # fmt: skip
