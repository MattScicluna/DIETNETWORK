"""Edge cases of dietnet.parallel: its building blocks, refusals and rarer paths.

``main`` runs in this process as a single gloo rank, on the synthetic fixture, with
a config that stops after the first epoch, so each run takes a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.distributed as dist
import yaml
from support import CONFIG, DATASETS, dietnet

from dietnet import parallel, training, upstream


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """The synthetic fixture's dataset, partition and input statistics, plus a copy
    of the statistics without standard deviations and a one-epoch config."""
    root = tmp_path_factory.mktemp("parallel_edges")
    data = DATASETS["synthetic"]
    dietnet("create-dataset", plink=data.plink_prefix, out=root / "dataset.h5", labels=data.labels)
    common = dict(dataset=root / "dataset.h5", partition=root / "partition.npz")
    dietnet("partition", dataset=root / "dataset.h5", out=root / "partition.npz")
    dietnet("compute-input-stats", **common, out=root / "stats.npz")
    stats = np.load(root / "stats.npz", allow_pickle=True)
    np.savez(root / "means_only.npz", means_by_fold=stats["means_by_fold"])
    config = {**CONFIG, "epochs": 3, "patience": 0}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    return root


@pytest.fixture
def single_rank(monkeypatch):
    """The environment torchrun would give one process; the process group and the
    upstream function main replaces are restored afterwards."""
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(training._free_port()))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    if training._loopback():
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", training._loopback())
    utils = upstream.modules().dataset_utils
    monkeypatch.setattr(utils, "replace_missing_values", utils.replace_missing_values)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _argv(root, exp, *extra, stats="stats.npz"):
    return [
        "--exp-path", str(exp), "--exp-name", "runs", "--config", str(root / "config.yaml"),
        "--dataset", str(root / "dataset.h5"), "--partition", str(root / "partition.npz"),
        "--input-features-stats", str(root / stats), "--which-fold", "0", *extra,
    ]  # fmt: skip


def test_a_chunk_plan_without_snp_blocks_keeps_every_snp():
    """Without blocks, every buffer holds all SNPs, and an epoch reads every row once."""
    plan = parallel.ChunkPlan(np.arange(100)[::-1], 10, 10, 2, 0, 1.0, seed=0)
    assert plan.blocks is None
    assert len(plan) == 5
    buffers = list(plan)
    assert len(buffers) == 5
    for rows, ranges in buffers:
        assert ranges.tolist() == [[0, 10]]
        assert len(rows) == 20 and (np.diff(rows) > 0).all()
    assert sorted(np.concatenate([rows for rows, _ in buffers]).tolist()) == list(range(100))

    first = [rows.tolist() for rows, _ in buffers]
    plan.set_epoch(1)
    assert [rows.tolist() for rows, _ in plan] != first


def test_chunk_buffers_cut_the_kept_columns_from_unpacked_rows():
    """An unpacked store gives whole rows; the buffer keeps the ranges' columns, in order."""
    store = np.arange(60, dtype=np.int8).reshape(6, 10)
    labels = np.arange(6) * 10
    buffers = parallel.ChunkBuffers(store, labels, packed=False)

    x, y, cols = buffers[(np.array([1, 4]), np.array([[2, 5], [7, 9]]))]
    assert cols.tolist() == [2, 3, 4, 7, 8]
    assert x.tolist() == [store[1, cols.numpy()].tolist(), store[4, cols.numpy()].tolist()]
    assert y.tolist() == [10, 40]

    x, _, cols = buffers[(np.array([0]), np.array([[0, 10]]))]
    assert x.tolist() == [store[0].tolist()] and cols.tolist() == list(range(10))


def test_the_mlp_trains_from_chunks_unnormalised_and_stops_early(
    prepared, tmp_path, single_rank, capsys
):
    """The MLP, with chunked reads in this process (no loader workers), batch-norm
    recalibration, no normalisation and no standard deviations: one epoch, then
    early stopping (patience 0), with the best model and test results written."""
    argv = _argv(
        prepared, tmp_path, "--model", "Mlp", "--inputs", "memmap", "--workers", "0",
        "--chunk-rows", "16", "--chunks-per-buffer", "2", "--bn-recalibrate", "20",
        stats="means_only.npz",
    )  # fmt: skip
    parallel.main(argv)

    out = capsys.readouterr().out
    assert "data-parallel training on 1 CPU processes; global batch 32" in out
    assert "batch-norm statistics recomputed on 20 training samples" in out
    assert "Epoch 1:" in out and "Epoch 2:" not in out
    assert "Early stopping, exiting training loop" in out
    config = yaml.safe_load((prepared / "config.yaml").read_text())
    name = upstream.exp_identifier(config, 0, "classification", "Mlp")
    results = tmp_path / "runs" / f"RESULTS_{name}"
    assert (results / "best_model.pt").is_file()
    assert len(list(results.glob("test_results_epoch*.npz"))) == 1
    state = torch.load(results / "best_model.pt", weights_only=False)["model_state_dict"]
    assert any(key.startswith("hidden_layers") for key in state)


@pytest.mark.parametrize(
    "flag, message",
    [
        (["--free-snp-weights", "0.1"], "--free-snp-weights applies to the Diet Network only"),
        (["--snp-sample", "0.5"], "--snp-sample applies to the Diet Network only"),
    ],
)
def test_diet_network_only_options_refuse_the_mlp(prepared, tmp_path, single_rank, flag, message):
    """Options that change the Diet Network's fat layer stop an MLP run before training."""
    with pytest.raises(SystemExit, match=message):
        parallel.main(_argv(prepared, tmp_path, "--model", "Mlp", *flag))
    assert not (tmp_path / "runs").exists()


def test_a_buffer_smaller_than_a_batch_is_refused(prepared, tmp_path, single_rank):
    """Chunks so small that a buffer holds fewer rows than one batch: refused, saying so."""
    argv = _argv(prepared, tmp_path, "--model", "Mlp", "--chunk-rows", "1")
    with pytest.raises(SystemExit, match="a buffer of 4 rows holds no batch of 32"):
        parallel.main(argv)


def test_penalty_weights_without_free_weights_are_refused(prepared, tmp_path, single_rank):
    """--snp-penalty-weights scales the free weights' penalty: without them, refused."""
    argv = _argv(prepared, tmp_path, "--snp-penalty-weights", str(tmp_path / "w.npy"))
    with pytest.raises(SystemExit, match="--snp-penalty-weights needs --free-snp-weights"):
        parallel.main(argv)
