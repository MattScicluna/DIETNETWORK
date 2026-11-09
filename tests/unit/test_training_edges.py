"""Edge cases of dietnet.training: argument checks, input checks and the data-parallel
launcher's failure handling."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml
from support import CONFIG

from dietnet import training, upstream

N_SAMPLES, N_SNPS, N_FOLDS = 30, 12, 2


def _partition(path, n_samples=N_SAMPLES, folds=N_FOLDS):
    """A partition whose every fold trains on all but its last ten samples."""
    parts = np.empty(folds, dtype=object)
    rows = np.arange(n_samples)
    for fold in range(folds):
        parts[fold] = [rows[:-10], rows[-10:-5], rows[-5:]]
    np.savez(path, folds_indexes=parts)
    return path


def _dataset(path, labels=True, chunked=False):
    """A small HDF5 dataset in upstream's schema; ``chunked`` stores the genotypes
    chunked and compressed, as a memory map cannot read them."""
    rng = np.random.default_rng(0)
    inputs = rng.integers(0, 3, size=(N_SAMPLES, N_SNPS), dtype=np.int8)
    with h5py.File(path, "w") as f:
        options = dict(chunks=(10, N_SNPS), compression="gzip") if chunked else {}
        f.create_dataset("inputs", data=inputs, **options)
        f.create_dataset("samples", data=np.array([f"s{i}" for i in range(N_SAMPLES)], "S"))
        f.create_dataset("snp_names", data=np.array([f"rs{i}_A" for i in range(N_SNPS)], "S"))
        if labels:
            f.create_dataset("class_labels", data=np.arange(N_SAMPLES) % 3)
            f.create_dataset("class_label_names", data=np.array([b"A", b"B", b"C"]))
    return path


def _stats(path, nan_in=(), sds=True):
    """Input statistics per fold; ``nan_in`` holds (key, fold) pairs to spoil."""
    stats = {"means_by_fold": np.ones((N_FOLDS, N_SNPS))}
    if sds:
        stats["sd_by_fold"] = np.ones((N_FOLDS, N_SNPS))
    for key, fold in nan_in:
        stats[key][fold, 3] = np.nan
    np.savez(path, **stats)
    return path


@pytest.fixture
def inputs(tmp_path):
    """Files train() accepts, for runs stopped before any training."""
    np.savez(tmp_path / "embedding.npz", emb=np.ones((N_FOLDS, N_SNPS, 3)))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    return dict(
        dataset=_dataset(tmp_path / "dataset.h5", chunked=True),
        partition=_partition(tmp_path / "partition.npz"),
        embedding=tmp_path / "embedding.npz",
        input_stats=_stats(tmp_path / "stats.npz"),
        config=tmp_path / "config.yaml",
        exp_dir=tmp_path / "exp",
        seeds=[5],
        folds=[0],
    )


UNCHECKED = dict(
    dataset="unused.h5",
    partition="unused.npz",
    embedding="unused_embedding.npz",
    input_stats="unused_stats.npz",
    config="unused.yaml",
    exp_dir="unused",
    seeds=[1],
    folds=[0],
)


@pytest.mark.parametrize(
    "options, message",
    [
        (dict(engine="torch"), "engine must be 'upstream' or 'dietnet', not 'torch'"),
        (dict(loader="slow"), "loader must be 'fast' or 'upstream', not 'slow'"),
        (dict(gpus=0), "gpus must be at least 1"),
        (dict(gpus=2, engine="upstream"), "engine 'upstream' is train.py's own loop"),
        (dict(bn_recalibrate=10), "--bn-recalibrate needs engine 'dietnet'"),
        (dict(snp_sample=0.0), r"snp_sample must be in \(0, 1\], not 0.0"),
        (dict(snp_sample=1.5, engine="dietnet"), r"snp_sample must be in \(0, 1\], not 1.5"),
        (dict(snp_sample=0.5), "--snp-sample needs engine 'dietnet' and the Diet Network"),
        (
            dict(snp_sample=0.5, engine="dietnet", model="Mlp", embedding=None),
            "--snp-sample needs engine 'dietnet' and the Diet Network",
        ),
        (dict(chunk_rows=64), "--chunk-rows needs engine 'dietnet'"),
        (dict(free_snp_weights=-1.0, engine="dietnet"), "free_snp_weights must be >= 0, not -1.0"),
        (dict(free_snp_weights=0.1), "--free-snp-weights needs engine 'dietnet'"),
        (
            dict(free_snp_weights=0.1, engine="dietnet", model="Mlp", embedding=None),
            "--free-snp-weights needs engine 'dietnet' and the Diet Network",
        ),
        (dict(snp_blocks=8, engine="dietnet", snp_sample=0.5), "--snp-blocks needs --chunk-rows"),
        (dict(snp_blocks=8, engine="dietnet", chunk_rows=64), "--snp-blocks needs --chunk-rows"),
        (dict(engine="dietnet", loader="upstream"), "--loader upstream applies to engine"),
        # Several GPUs choose the dietnet engine, which has no upstream loader.
        (dict(gpus=2, loader="upstream"), "--loader upstream applies to engine 'upstream'"),
        (dict(model="Resnet"), r"model must be one of \['Dietnet', 'Mlp'\]"),
    ],
)
def test_train_refuses_inconsistent_options(options, message):
    """Each inconsistent option is refused, by name, before any file is read."""
    with pytest.raises(ValueError, match=message):
        training.train(**{**UNCHECKED, **options})


@pytest.mark.parametrize(
    "mode, message",
    [
        ("memmap", "cannot memory-map the genotypes: .*not contiguous and uncompressed"),
        ("packed", "no up-to-date packed copy; run `dietnet pack-genotypes` first"),
        ("disk", r"inputs mode must be one of \['auto', 'ram', 'memmap', 'packed'\]"),
    ],
)
def test_an_unusable_genotype_store_is_an_input_error(inputs, mode, message):
    """A store the dataset cannot give is an InputError, without the chained traceback."""
    with pytest.raises(training.InputError, match=message) as caught:
        training.train(**inputs, inputs=mode)
    assert caught.value.__suppress_context__
    assert not (inputs["exp_dir"] / "configs").exists()


class _Stop(Exception):
    """Raised by the stand-in launcher, so that no training is attempted."""


@pytest.mark.parametrize("lr", [0.05, None])
def test_the_free_weights_learning_rate_is_passed_only_when_set(inputs, monkeypatch, lr):
    """The dietnet engine gets --free-snp-lr when one is given, and otherwise none (its
    default, lr_main, applies); the seed's config is written first."""
    seen = []

    def launch(args, processes, log):
        seen.append((args, processes, log))
        raise _Stop

    monkeypatch.setattr(training, "_run_parallel", launch)
    with pytest.raises(_Stop):
        training.train(
            **inputs, engine="dietnet", inputs="ram", workers=0, free_snp_weights=1e-3,
            free_snp_lr=lr,
        )  # fmt: skip

    ((args, processes, log),) = seen
    assert processes == 1
    assert log == inputs["exp_dir"] / "logs" / "train_seed5_fold0.log"
    assert args[args.index("--free-snp-weights") + 1] == 1e-3
    assert args[args.index("--inputs") + 1] == "ram"
    if lr is None:
        assert "--free-snp-lr" not in args
    else:
        assert args[args.index("--free-snp-lr") + 1] == lr
    written = yaml.safe_load((inputs["exp_dir"] / "configs" / "config_seed5.yaml").read_text())
    assert written == {**CONFIG, "seed": 5}


def test_check_inputs_names_nan_statistics_without_class_labels(tmp_path):
    """A dataset without class labels (regression) skips the class check; NaN input
    statistics are still reported, per fold and per array."""
    dataset = _dataset(tmp_path / "d.h5", labels=False)
    partition = _partition(tmp_path / "p.npz")
    stats = _stats(tmp_path / "s.npz", nan_in=[("sd_by_fold", 1)])

    with pytest.raises(training.InputError, match="fold 1: NaN in sd_by_fold") as caught:
        training.check_inputs(None, stats, dataset, partition)
    assert "fold 0" not in str(caught.value)
    assert "no training samples" not in str(caught.value)
    training.check_inputs(None, stats, dataset, partition, folds=[0])  # fold 0 is clean

    both = _stats(tmp_path / "b.npz", nan_in=[("means_by_fold", 0), ("sd_by_fold", 0)])
    with pytest.raises(training.InputError, match="fold 0: NaN in means_by_fold, sd_by_fold"):
        training.check_inputs(None, both, dataset, partition)


def test_check_inputs_accepts_statistics_without_standard_deviations(tmp_path):
    """Means alone (no sd_by_fold) are valid statistics."""
    dataset = _dataset(tmp_path / "d.h5")
    stats = _stats(tmp_path / "s.npz", sds=False)
    training.check_inputs(None, stats, dataset, _partition(tmp_path / "p.npz"))


def test_a_failed_data_parallel_run_raises_with_its_log(tmp_path):
    """dietnet.parallel refusing its arguments: the error carries the exit status and
    the log's tail."""
    log = tmp_path / "logs" / "train.log"
    with pytest.raises(upstream.UpstreamError, match="exited with status 2") as caught:
        training._run_parallel(["--not-an-option"], 1, log)
    message = str(caught.value)
    assert f"log {log}:" in message
    assert "the following arguments are required" in message
    assert log.is_file()


def _fake_popen(scripts, launched):
    """A subprocess stand-in that runs ``scripts[rank]`` in place of dietnet.parallel."""

    def popen(cmd, env, stdout, stderr):
        assert cmd[1:4] == ["-u", "-m", "dietnet.parallel"]
        proc = subprocess.Popen(
            [sys.executable, "-c", scripts[int(env["RANK"])]], env=env, stdout=stdout, stderr=stderr
        )
        launched.append((env, proc))
        return proc

    return SimpleNamespace(Popen=popen, STDOUT=subprocess.STDOUT)


def test_one_failed_rank_stops_the_others(tmp_path, monkeypatch):
    """When rank 1 fails while rank 0 still runs, rank 0 is killed and the error shows
    the last 25 lines of rank 1's log; an interface set by the caller is kept."""
    launched = []
    scripts = [
        "import time; time.sleep(60)",
        "print('\\n'.join(f'line {i}' for i in range(40))); raise SystemExit(3)",
    ]
    monkeypatch.setattr(training, "subprocess", _fake_popen(scripts, launched))
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "custom0")
    log = tmp_path / "train.log"

    with pytest.raises(
        upstream.UpstreamError, match=r"status 3; log .*train\.log\.rank1:"
    ) as caught:
        training._run_parallel(["--anything"], 2, log)
    message = str(caught.value)

    assert "line 39" in message and "line 15\n" in message and "line 14\n" not in message
    for env, proc in launched:
        assert env["GLOO_SOCKET_IFNAME"] == "custom0"
        assert env["WORLD_SIZE"] == "2" and env["MASTER_ADDR"] == "127.0.0.1"
        assert env["RANK"] == env["LOCAL_RANK"]
    assert launched[0][1].returncode < 0  # killed, not finished
    assert launched[1][1].returncode == 3


def test_ranks_that_all_succeed_raise_nothing(tmp_path, monkeypatch):
    """Every rank exiting 0 is a successful run; each rank writes its own log."""
    launched = []
    scripts = ["print('rank 0 done')", "print('rank 1 done')"]
    monkeypatch.setattr(training, "subprocess", _fake_popen(scripts, launched))
    log = tmp_path / "train.log"

    assert training._run_parallel([], 2, log) is None
    assert [proc.returncode for _, proc in launched] == [0, 0]
    assert log.read_text() == "rank 0 done\n"
    assert (tmp_path / "train.log.rank1").read_text() == "rank 1 done\n"
