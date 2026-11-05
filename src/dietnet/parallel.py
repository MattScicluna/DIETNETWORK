"""Data-parallel training over several GPUs: upstream's model, in a DDP loop.

``dietnet train --gpus N`` (N > 1) runs this under ``torchrun``, one process
per GPU. What it shares with upstream's train.py, and what it does not:

- The same model, loss and optimisers: upstream's model and task handlers,
  built as train.py builds them; DDP averages the gradients across GPUs before
  the model's own two Adam optimisers step.
- The same epoch logic: a baseline on the validation set, learning-rate
  annealing after every epoch, the best model chosen by upstream's own rule
  (``update_best_results``: validation accuracy, then loss), patience-based
  early stopping, and the test set scored with the best model. Validation and
  test run on rank 0 through upstream's ``eval_step`` and write the same
  results directory (``best_model.pt``, ``test_results_epoch<N>.npz``), so
  packaging, ``predict`` and ``check`` treat the run like any other.
- A different training pass: each GPU reads its own share of every epoch's
  shuffled samples (``DistributedSampler``), whole batches per read from the
  genotype store, from loader workers; packed genotypes are decoded on the
  GPU. The config's ``batch_size`` is the *global* batch, split across GPUs, so
  the hyperparameters mean the same on one GPU or eight. The pass over the
  training set that upstream evaluates every epoch for monitoring is skipped.

The batches therefore differ from train.py's, and so do the weights; the check
is held-out accuracy (docs/scaling.md), not bit-identity. On a machine without
GPUs it runs N CPU processes (gloo), which is how the tests exercise it.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, SequentialSampler

from dietnet import genotypes, hybrid, upstream
from dietnet._launch import _fill_missing

EVAL_BATCH = 2048
TRAIN_GAP = 8  # chunked training reads span gaps of up to this many rows (~1.7x the bytes)
EVAL_GAP = 16  # the one pass that caches the evaluation rows
EVAL_INPUT_BYTES = 4 * 2**30  # float32 inputs of one evaluation batch; preprocessing copies them


def eval_batch_size(batch_size: int, n_snps: int) -> int:
    """Samples per evaluation batch: :data:`EVAL_BATCH`, fewer where a batch's inputs would
    exceed :data:`EVAL_INPUT_BYTES` (2048 x 1.29M SNPs is 9.84 GiB, copied several times
    over by preprocessing: more than an A100 holds), never fewer than a training batch.
    """
    return max(batch_size, min(EVAL_BATCH, EVAL_INPUT_BYTES // (4 * n_snps)))


class Split(torch.utils.data.Dataset):
    """Samples ``rows`` (dataset row numbers) of a genotype store, read a batch at a
    time: (genotypes, labels, rows). With ``packed``, the genotypes are the 2-bit
    bytes, for the GPU to decode."""

    def __init__(self, store, labels, rows, packed=False, source=None):
        self.store, self.labels, self.packed = store, labels, packed
        self.rows = np.asarray(rows)
        self.source = source  # rows from here (e.g. a CachedRows) rather than the store

    def __len__(self):
        return len(self.rows)

    def __getitems__(self, positions):
        rows = self.rows[np.asarray(positions)]
        source = (
            self.source
            if self.source is not None
            else (self.store.reader if self.packed else self.store)
        )
        x = genotypes.read_rows(source, rows)
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.asarray(self.labels[rows])),
            torch.from_numpy(rows.astype(np.int64)),
        )


def _as_is(batch):
    return batch


def _loader(dataset, batch_sampler, workers):
    options = dict(
        batch_sampler=batch_sampler,
        collate_fn=_as_is,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    if workers:
        # Workers inherit the opened store, and stay up between epochs.
        options.update(multiprocessing_context="fork", persistent_workers=True)
    return DataLoader(dataset, **options)


def recalibrate_batchnorm(model, batches, forward) -> int:
    """Recompute every batch-norm layer's running statistics as plain averages over
    ``batches`` (each fed to ``forward``), with the rest of ``model`` in evaluation
    mode, so without dropout: the statistics it then evaluates with describe the
    activations it gets at evaluation, not those of training with dropout. Leaves
    the model in evaluation mode; returns the number of samples seen.
    """
    norms = [mod for mod in model.modules() if isinstance(mod, nn.modules.batchnorm._BatchNorm)]
    momenta = [bn.momentum for bn in norms]
    model.eval()
    for bn in norms:
        bn.reset_running_stats()
        bn.momentum = None  # a cumulative average over the batches
        bn.train()
    seen = 0
    with torch.no_grad():
        for x in batches:
            forward(x)
            seen += len(x)
    for bn, momentum in zip(norms, momenta):
        bn.momentum = momentum
    model.eval()
    return seen


class SnpSampled(nn.Module):
    """Upstream's Diet Network, trained on a random share of the SNPs per batch.

    In training, each batch draws ``keep`` x n SNPs (the same for every sample of
    the batch): the auxiliary network computes fat-layer weights for those SNPs
    only, and the main network sees only their columns, scaled by n / k so the fat
    layer's expected input is unchanged, as dropout's 1 / (1 - p) scaling keeps it.
    This is input dropout with one mask per batch instead of one per sample, which
    is what lets the dropped SNPs be skipped: the auxiliary network and the fat
    layer do ~``keep`` of the work. It replaces the main network's per-sample input
    dropout (set to the identity by the caller); keep = 1 - input_dropout keeps the
    rate. In evaluation every SNP is used, as upstream's model does.
    """

    def __init__(self, net, keep: float):
        super().__init__()
        self.net, self.keep = net, keep

    def forward(self, x, results, epoch, batch, step, cols=None):
        """With ``cols`` (sorted SNP indices), ``x`` already holds only those columns,
        chosen by the data loader (:class:`ChunkPlan`); otherwise SNPs are drawn here."""
        net = self.net
        if cols is None and (not net.training or self.keep >= 1):
            return net(x, results, epoch, batch, step)
        n = net.embedding.shape[0]
        if cols is None:
            k = max(1, round(self.keep * n))
            cols = torch.randperm(n, device=x.device)[:k].sort().values
            x = x[:, cols]
        weights = hybrid.fat_weights(net, cols)  # with free per-SNP weights, if any
        return net.main_net(x * (n / len(cols)), weights, results, epoch, batch, step)


class ChunkPlan:
    """Which rows and SNPs each in-memory buffer of an epoch holds.

    A training pass reading random rows needs the whole genotype file in the page
    cache (137 GB at 1.29M SNPs; from a cold disk an epoch took 96 min instead of
    39 s). Here the sorted training rows are cut into contiguous chunks of
    ``chunk_rows``; every epoch the chunks are shuffled, and each buffer is read from
    ``chunks_per_buffer`` of them, in file order, then shuffled in memory into batches.
    With ``snp_blocks``, the SNPs are cut into that many contiguous blocks and each
    buffer keeps ``keep`` of them: only those columns are held and decoded. Block edges
    are multiples of 4 SNPs, so that a block is whole bytes of the 2-bit copy.

    Iterating gives ``(rows, ranges)``, the sorted rows and the kept SNP ranges of each
    buffer of this rank. Every rank has the same number of buffers of the same size
    (DDP needs equal batch counts), so up to a batch per buffer is left out, at random.
    """

    def __init__(self, train_rows, n_snps, chunk_rows, chunks_per_buffer, snp_blocks, keep,
                 seed, rank=0, world=1):  # fmt: skip
        self.rows = np.sort(np.asarray(train_rows))
        self.n_snps, self.chunks_per_buffer = n_snps, chunks_per_buffer
        self.keep, self.seed, self.rank, self.world = keep, seed, rank, world
        # At least one buffer per rank: smaller chunks, where there are few rows.
        self.n_chunks = max(chunks_per_buffer * world, len(self.rows) // chunk_rows)
        self.buffer_rows = chunks_per_buffer * (len(self.rows) // self.n_chunks)
        self.epoch = 0
        edges = np.linspace(0, n_snps, snp_blocks + 1).astype(np.int64) // 4 * 4
        edges[-1] = n_snps
        self.blocks = np.stack([edges[:-1], edges[1:]], axis=1) if snp_blocks else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return (self.n_chunks // self.world) // self.chunks_per_buffer

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch, self.rank])
        shared = np.random.default_rng([self.seed, self.epoch])
        chunks = np.array_split(self.rows, self.n_chunks)
        order = shared.permutation(self.n_chunks)[self.rank :: self.world]
        for b in range(len(self)):
            picked = order[b * self.chunks_per_buffer : (b + 1) * self.chunks_per_buffer]
            rows = np.concatenate([chunks[c] for c in picked])
            rows = np.sort(rng.permutation(rows)[: self.buffer_rows])
            if self.blocks is None:
                ranges = np.array([[0, self.n_snps]])
            else:
                k = max(1, round(self.keep * len(self.blocks)))
                ranges = self.blocks[np.sort(rng.permutation(len(self.blocks))[:k])]
            yield rows, ranges


def plan_columns(ranges: np.ndarray) -> np.ndarray:
    """The SNP indices of the kept ranges, in order."""
    return np.concatenate([np.arange(a, b) for a, b in ranges])


class ChunkBuffers(torch.utils.data.Dataset):
    """One item is one buffer of a :class:`ChunkPlan`: (genotypes of the kept columns,
    labels, SNP indices). Whole rows are read in file order and the columns cut in
    memory, so disk access stays sequential; packed genotypes stay packed, for the GPU
    to decode, as whole bytes of each kept block."""

    def __init__(self, store, labels, packed):
        self.store, self.labels, self.packed = store, labels, packed

    def __getitem__(self, item):
        rows, ranges = item
        if self.packed:  # sorted rows: a few large reads, through gaps of up to TRAIN_GAP rows
            x = self.store.reader.read(rows, max_gap=TRAIN_GAP)
        else:
            x = np.asarray(genotypes.read_rows(self.store, rows))
        unit = 4 if self.packed else 1
        kept = [x[:, a // unit : -(-b // unit)] for a, b in ranges]
        return (
            torch.from_numpy(np.concatenate(kept, axis=1)),
            torch.from_numpy(np.asarray(self.labels[rows])),
            torch.from_numpy(plan_columns(ranges)),
        )


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # train.py's names, so dietnet.training builds both command lines alike.
    p.add_argument("--exp-path", required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--model", choices=sorted(upstream.MODELS), default="Dietnet")
    p.add_argument("--dataset", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--embedding")
    p.add_argument("--input-features-stats", required=True)
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--which-fold", type=int, required=True)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--inputs", choices=genotypes.MODES, default="memmap")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--bn-recalibrate", type=int, default=0,
        help="training samples to recompute batch-norm statistics on, without dropout, "
        "before each validation (0: off, as train.py)",
    )  # fmt: skip
    p.add_argument(
        "--snp-sample", type=float, default=1.0,
        help="Diet Network: train each batch on this share of the SNPs, shared by the batch, "
        "in place of per-sample input dropout (1: off, as train.py)",
    )  # fmt: skip
    p.add_argument(
        "--chunk-rows", type=int, default=0,
        help="read training rows in contiguous chunks of this size into shuffled in-memory "
        "buffers (ChunkPlan), rather than batches of random rows (0: off)",
    )  # fmt: skip
    p.add_argument("--chunks-per-buffer", type=int, default=4)
    p.add_argument(
        "--snp-blocks", type=int, default=0,
        help="with --chunk-rows and --snp-sample: cut the SNPs into this many contiguous "
        "blocks, and hold only a --snp-sample share of the blocks per buffer (0: off)",
    )  # fmt: skip
    p.add_argument(
        "--free-snp-weights", type=float, default=0.0,
        help="Diet Network: add free per-SNP fat-layer weights under a group-lasso penalty "
        "of this strength (dietnet.hybrid; 0: off)",
    )  # fmt: skip
    p.add_argument("--free-snp-lr", type=float, help="their Adam learning rate (default lr_main)")
    return p.parse_args(argv)


def main(argv=None) -> None:  # noqa: C901 - one training run, in train.py's order
    args = _args(argv)
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    lead = rank == 0

    def say(*parts):
        if lead:
            print(*parts, flush=True)

    m = upstream.modules()
    m.dataset_utils.replace_missing_values = _fill_missing  # same values, one operation
    config = yaml.safe_load(Path(args.config).read_text())
    fold = args.which_fold

    # As train.py: seeds first, so every rank builds the same initial model.
    seed = config["seed"]
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

    if args.task == "classification":
        task = m.task_handlers.ClassificationHandler(args.dataset, nn.CrossEntropyLoss())
    else:
        task = m.task_handlers.RegressionHandler(args.dataset, nn.MSELoss())
    if args.model == "Dietnet":
        handler = m.model_handlers.DietNetworkHandler(
            task, fold, args.embedding, device, args.dataset, config, None
        )
    else:
        handler = m.model_handlers.MlpHandler(task, args.dataset, config)
    model = handler.model.to(device)
    delta = None
    if args.free_snp_weights > 0:
        if args.model != "Dietnet":
            raise SystemExit("--free-snp-weights applies to the Diet Network only")
        delta = hybrid.attach(model)  # all 0: the model computes what it did
    trained = model
    if args.snp_sample < 1:
        if args.model != "Dietnet":
            raise SystemExit("--snp-sample applies to the Diet Network only")
        model.main_net.input_dropout = nn.Identity()  # replaced by the shared SNP sample
        trained = SnpSampled(model, args.snp_sample)
    ddp = DistributedDataParallel(trained, device_ids=[local_rank] if cuda else None)
    optimizers = model.get_optimizers()
    delta_optimizer = None
    if delta is not None:
        delta_optimizer = torch.optim.Adam([delta], lr=args.free_snp_lr or config["lr_main"])
        optimizers = (*optimizers, delta_optimizer)  # stepped and annealed with the others
    torch.manual_seed(seed + 1000 * rank)  # each rank its own dropout masks

    # Data.
    fold_rows = np.load(args.partition, allow_pickle=True)["folds_indexes"][fold]
    train_rows, valid_rows, test_rows = (np.asarray(r) for r in fold_rows)
    stats = np.load(args.input_features_stats, allow_pickle=True)
    mus = torch.from_numpy(stats["means_by_fold"][fold]).float().to(device)
    sigmas = None
    if "sd_by_fold" in stats.files:
        sigmas = torch.from_numpy(stats["sd_by_fold"][fold]).float().to(device)
    store = genotypes.open_store(args.dataset, args.inputs)
    with h5py.File(args.dataset, "r") as f:
        key = "class_labels" if args.task == "classification" else "regression_labels"
        labels = np.array(f[key], dtype=np.int64 if args.task == "classification" else np.float32)
    packed = args.inputs == "packed"
    table = torch.from_numpy(genotypes.DECODE_TABLE).to(device)
    n_snps = store.shape[1]

    batch_size = config["batch_size"]
    per_rank = -(-batch_size // world)
    train = Split(store, labels, train_rows, packed=packed)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=seed)
    train_loader = _loader(train, BatchSampler(sampler, per_rank, drop_last=False), args.workers)
    if args.chunk_rows:
        sampler = ChunkPlan(
            train_rows, n_snps, args.chunk_rows, args.chunks_per_buffer, args.snp_blocks,
            args.snp_sample, seed, rank, world,
        )  # fmt: skip
        if sampler.buffer_rows < per_rank:
            raise SystemExit(
                f"a buffer of {sampler.buffer_rows} rows holds no batch of {per_rank}: "
                "raise --chunk-rows or --chunks-per-buffer"
            )
        buffers = ChunkBuffers(store, labels, packed)
        options = dict(
            batch_size=None, sampler=sampler, collate_fn=_as_is, num_workers=args.workers
        )
        if args.workers:  # one buffer ahead per worker: buffers are large
            options.update(
                multiprocessing_context="fork", persistent_workers=True, prefetch_factor=1
            )
        train_loader = DataLoader(buffers, **options)

    def training_batches():
        """(genotypes, labels, SNP indices or None) per training batch."""
        if not args.chunk_rows:
            for x, y, _ in train_loader:
                yield x, y, None
            return
        for x, y, cols in train_loader:
            cols = cols.to(device) if args.snp_blocks else None
            shuffled = torch.randperm(len(x))
            for b in range(len(x) // per_rank):
                picked = shuffled[b * per_rank : (b + 1) * per_rank]
                yield x[picked], y[picked], cols

    # Evaluation has no dropout, so its batch size changes no prediction, only how
    # many batches upstream's eval_step does its per-batch bookkeeping for.
    eval_batch = eval_batch_size(batch_size, n_snps)

    def decoded(batches):
        """Batches with packed genotypes decoded on the GPU: upstream's eval_step
        then moves them to the device (a no-op) and carries on as with int8 rows."""
        for x, y, rows in batches:
            x = x.to(device, non_blocking=True)
            yield table[x.long()].reshape(len(x), -1)[:, :n_snps], y, rows

    class _Decoded:
        def __init__(self, loader):
            self.loader = loader

        def __len__(self):  # eval_step sizes its per-batch results by it
            return len(self.loader)

        def __iter__(self):
            return decoded(self.loader)

    def cached(rows):
        """With chunked reads, rows that every epoch reads again are read once, in a few
        large reads, and kept in memory (packed: ~34 GB for validation and the BN
        recalibration subset at 1.29M SNPs)."""
        if not (args.chunk_rows and packed):
            return None
        started = time.time()
        cache = genotypes.CachedRows(lambda r: store.reader.read(r, max_gap=EVAL_GAP), rows)
        say(f"dietnet: {len(cache.rows)} evaluation rows cached in memory "
            f"({cache.data.nbytes / 2**30:.1f} GiB, {time.time() - started:.0f} s)")  # fmt: skip
        return cache

    def eval_loader(rows, source=None):
        split = Split(store, labels, rows, packed=packed, source=source)
        sampler = BatchSampler(SequentialSampler(split), eval_batch, False)
        loader = _loader(split, sampler, args.workers)
        return split, (_Decoded(loader) if packed else loader)

    pick = None
    if args.bn_recalibrate:
        # A fixed random subset of the training rows, read in file order.
        pick = np.random.default_rng(seed).choice(
            train_rows, min(args.bn_recalibrate, len(train_rows)), replace=False
        )
    evaluated = cached(valid_rows if pick is None else np.concatenate([valid_rows, pick]))
    valid_set, valid_loader = eval_loader(valid_rows, evaluated)
    recalibration = None
    if args.bn_recalibrate:
        _, recalibration = eval_loader(np.sort(pick), evaluated)
        say(f"dietnet: batch-norm statistics recomputed on {len(pick)} training samples, "
            "without dropout, before each validation")  # fmt: skip

    def recalibrated_inputs():
        """The recalibration subset, preprocessed as training batches are."""
        for x, _, _ in recalibration:
            x = x.to(device, non_blocking=True).float()
            _fill_missing(x, mus)
            if args.normalize:
                x = m.dataset_utils.normalize(x, mus, sigmas)
            yield x

    results = Path(args.exp_path) / args.exp_name
    results = results / f"RESULTS_{upstream.exp_identifier(config, fold, args.task, args.model)}"
    best_path, last_path = results / "best_model.pt", results / "last_model.pt"
    if lead:
        results.mkdir(parents=True, exist_ok=True)
    say(f"dietnet: data-parallel training on {world} {'GPUs' if cuda else 'CPU processes'}; "
        f"global batch {batch_size} ({per_rank} per process), {args.inputs} store, "
        f"{args.workers} loader workers")  # fmt: skip

    def validate(epoch):
        if recalibration is not None:
            recalibrate_batchnorm(
                model, recalibrated_inputs(), lambda x: model(x, str(results), epoch, 0, "valid")
            )
        model.eval()
        with torch.no_grad():
            out = m.mainloop_utils.eval_step(
                handler, device, valid_set, valid_loader, mus, sigmas, args.normalize,
                str(results), epoch,
            )  # fmt: skip
        return out

    if lead:
        baseline = validate("baseline")
        task.init_best_epoch_results(baseline)
        task.print_baseline_results(baseline)
        torch.save(
            {"epoch": 0, "model_state_dict": model.state_dict(),
             "best_results": task.best_epoch_results},
            best_path,
        )  # fmt: skip

    patience, max_patience = 0, config["patience"]
    started = time.time()
    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        model.train()
        totals = torch.zeros(3, device=device)  # loss x n, n, correct
        for batch, (x, y, cols) in enumerate(training_batches()):
            x = x.to(device, non_blocking=True)
            y = task.format_ybatch(y.to(device, non_blocking=True))
            if packed:
                width = n_snps if cols is None else len(cols)
                x = table[x.long()].reshape(len(x), -1)[:, :width]
            x = x.float()
            batch_mus = mus if cols is None else mus[cols]
            _fill_missing(x, batch_mus)
            if args.normalize:
                x = m.dataset_utils.normalize(
                    x, batch_mus, sigmas if cols is None else sigmas[cols]
                )
            for optimizer in optimizers:
                optimizer.zero_grad()
            if cols is None:
                out = ddp(x, str(results), epoch, batch, "train")
            else:
                out = ddp(x, str(results), epoch, batch, "train", cols=cols)
            loss = task.compute_loss(out, y)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            if delta_optimizer is not None:  # the group-lasso proximal step
                step_size = delta_optimizer.param_groups[0]["lr"]
                hybrid.prox(delta, step_size * args.free_snp_weights)
            totals[0] += loss.detach() * len(y)
            totals[1] += len(y)
            if args.task == "classification":
                totals[2] += (out.argmax(1) == y).sum()
        dist.all_reduce(totals)
        train_time = time.time() - epoch_start

        stop = False
        if lead:
            valid = validate(epoch)
            n = len(valid["samples"])
            say(f"Epoch {epoch + 1}: train loss {totals[0].item() / totals[1].item():.6f}"
                + (f" acc {100 * totals[2].item() / totals[1].item():.3f}%"
                   if args.task == "classification" else "")
                + f" | valid loss {valid['losses_wo_reduction'].sum() / n:.6f}"
                + (f" acc {valid['n_right'].sum() / n * 100:.3f}%"
                   if args.task == "classification" else "")
                + f" | train pass {train_time:.1f} s"
                + (f" | free weights on {hybrid.nonzero(delta)} SNPs" if delta is not None else ""))  # fmt: skip
        for optimizer in optimizers:  # annealed on every rank alike
            for group in optimizer.param_groups:
                group["lr"] *= config["learning_rate_annealing"]
        if lead:
            if task.update_best_results(valid):
                patience = 0
                torch.save(
                    {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                     "best_results": task.best_epoch_results},
                    best_path,
                )  # fmt: skip
                say("Saving best model")
            else:
                patience += 1
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "patience": patience},
                last_path,
            )  # fmt: skip
            say(f"Epoch execution time: {time.time() - epoch_start} seconds")
            stop = patience >= max_patience
        decision = [stop]
        dist.broadcast_object_list(decision, src=0)
        if decision[0]:
            say("Early stopping, exiting training loop")
            break
    say(f"Executed training in {time.time() - started} seconds")

    if lead:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        say(f"Loading best model from epoch {checkpoint['epoch']}")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        test_set, test_loader = eval_loader(test_rows, cached(test_rows))
        with torch.no_grad():
            test = m.mainloop_utils.eval_step(
                handler, device, test_set, test_loader, mus, sigmas, args.normalize,
                str(results), "test_step",
            )  # fmt: skip
        task.print_test_results(test)
        task.save_predictions(test, str(results / f"test_results_epoch{checkpoint['epoch']}"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
