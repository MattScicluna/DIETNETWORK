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

from dietnet import genotypes, upstream
from dietnet._launch import _fill_missing

EVAL_BATCH = 2048


class Split(torch.utils.data.Dataset):
    """Samples ``rows`` (dataset row numbers) of a genotype store, read a batch at a
    time: (genotypes, labels, rows). With ``packed``, the genotypes are the 2-bit
    bytes, for the GPU to decode."""

    def __init__(self, store, labels, rows, packed=False):
        self.store, self.labels, self.packed = store, labels, packed
        self.rows = np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def __getitems__(self, positions):
        rows = self.rows[np.asarray(positions)]
        source = self.store.packed if self.packed else self.store
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
    ddp = DistributedDataParallel(model, device_ids=[local_rank] if cuda else None)
    optimizers = model.get_optimizers()
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

    # Evaluation has no dropout, so its batch size changes no prediction, only how
    # many batches upstream's eval_step does its per-batch bookkeeping for.
    eval_batch = max(batch_size, EVAL_BATCH)

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

    def eval_loader(rows):
        split = Split(store, labels, rows, packed=packed)
        sampler = BatchSampler(SequentialSampler(split), eval_batch, False)
        loader = _loader(split, sampler, args.workers)
        return split, (_Decoded(loader) if packed else loader)

    valid_set, valid_loader = eval_loader(valid_rows)
    results = Path(args.exp_path) / args.exp_name
    results = results / f"RESULTS_{upstream.exp_identifier(config, fold, args.task, args.model)}"
    best_path, last_path = results / "best_model.pt", results / "last_model.pt"
    if lead:
        results.mkdir(parents=True, exist_ok=True)
    say(f"dietnet: data-parallel training on {world} {'GPUs' if cuda else 'CPU processes'}; "
        f"global batch {batch_size} ({per_rank} per process), {args.inputs} store, "
        f"{args.workers} loader workers")  # fmt: skip

    def validate(epoch):
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
        for batch, (x, y, _) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = task.format_ybatch(y.to(device, non_blocking=True))
            if packed:
                x = table[x.long()].reshape(len(x), -1)[:, :n_snps]
            x = x.float()
            _fill_missing(x, mus)
            if args.normalize:
                x = m.dataset_utils.normalize(x, mus, sigmas)
            for optimizer in optimizers:
                optimizer.zero_grad()
            out = ddp(x, str(results), epoch, batch, "train")
            loss = task.compute_loss(out, y)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
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
                + f" | train pass {train_time:.1f} s")  # fmt: skip
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
        test_set, test_loader = eval_loader(test_rows)
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
