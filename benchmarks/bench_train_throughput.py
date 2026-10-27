"""Training throughput by data path and batch size, with upstream's own model.

Upstream's train.py reads one sample per Python call in a single process
(``num_workers=0``), and fills missing genotypes with a Python loop over the
batch's samples. On a UK Biobank-sized matrix that left an A100 ~87% idle even
with every genotype in RAM. This times the alternatives, per batch size:

1. the loader alone, in samples/s: one read per sample (upstream) or one sorted
   read per batch, with 0 or more worker processes;
2. a full training step -- upstream's DietNetwork, loss and optimisers, run
   unmodified -- fed by either loader, with upstream's per-sample missing-value
   loop or one vectorised ``torch.where`` (the same values).

For timing only, on a benchmark from make_train_bench.py (memory-mapped):

    python benchmarks/bench_train_throughput.py BENCH_DIR [--dataset staged.h5]
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dietnet import genotypes, upstream
from dietnet.dataset import memmap_inputs

CONFIG = {
    "batch_size": 138,
    "epochs": 1,
    "input_dropout": 0.995,
    "dropout_main": 0.1,
    "lr_aux": 0.0006,
    "lr_main": 0.0003,
    "learning_rate_annealing": 0.99,
    "nb_hidden_u_aux": [100, 100],
    "nb_hidden_u_main": [100],
    "patience": 100,
    "seed": 78,
    "uniform_init_limit": 0.02,
}


class PerSample(Dataset):
    """Upstream's access pattern: one row per call."""

    def __init__(self, dataset, rows, labels):
        self.dataset, self.rows, self.labels, self.x = dataset, rows, labels, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.x is None:  # opened in the worker that uses it
            self.x = memmap_inputs(self.dataset)
        row = self.rows[i]
        return torch.from_numpy(np.array(self.x[row])), int(self.labels[row])


class Batched(Dataset):
    """One sorted read per batch: item k is batch k.

    ``kind``: "batched" reads the int8 rows from the dataset's memory map;
    "packed-cpu" reads the 2-bit copy and decodes in the worker; "packed-gpu"
    hands over the packed bytes (a quarter of the transfer) for the GPU to decode.
    """

    def __init__(self, dataset, batches, labels, kind):
        self.dataset, self.batches, self.labels, self.kind = dataset, batches, labels, kind
        self.x = None

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, k):
        if self.x is None:
            if self.kind == "batched":
                self.x = memmap_inputs(self.dataset)
            else:
                self.x = genotypes.PackedStore(self.dataset)
        rows = np.sort(self.batches[k])
        if self.kind == "packed-gpu":
            x = self.x.packed_rows(rows)
        else:
            x = np.asarray(self.x[rows])
        return torch.from_numpy(x), torch.from_numpy(self.labels[rows])


def loader(kind, dataset, rows, labels, batch_size, workers, rng):
    extra = {"num_workers": workers, "pin_memory": True}
    if workers:
        extra.update(prefetch_factor=2, persistent_workers=True)
    if kind == "per-sample":
        return DataLoader(
            PerSample(dataset, rows, labels), batch_size=batch_size, shuffle=True, **extra
        )
    order = rng.permutation(rows)
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    return DataLoader(Batched(dataset, batches, labels, kind), batch_size=None, **extra)


def _forever(data):
    while True:
        yield from data


def timed(data, n, step=None, sync=lambda: None):
    """Samples/s over n batches, after 2 warm-up batches (cycling if the data runs out)."""
    it = _forever(data)
    for _ in range(2):
        batch = next(it)
        if step:
            step(*batch)
    sync()
    start, samples = time.perf_counter(), 0
    for _ in range(n):
        batch = next(it)
        samples += len(batch[0])
        if step:
            step(*batch)
    sync()
    return samples / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bench_dir", type=Path)
    parser.add_argument("--dataset", type=Path, help="default: BENCH_DIR/dataset.h5")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[138, 512, 2048, 8192])
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--batches", type=int, default=20, help="timed batches per row")
    args = parser.parse_args()
    dataset = args.dataset or args.bench_dir / "dataset.h5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    with h5py.File(dataset, "r") as f:
        labels = np.array(f["class_labels"], dtype=np.int64)
    folds = np.load(args.bench_dir / "partition.npz", allow_pickle=True)["folds_indexes"]
    rows = np.asarray(folds[0][0])
    stats = np.load(args.bench_dir / "stats.npz")
    mus = torch.from_numpy(stats["means_by_fold"][0]).float().to(device)
    sds = torch.from_numpy(stats["sd_by_fold"][0]).float().to(device)
    print(f"{len(rows):,} training samples x {mus.numel():,} SNPs on {device}")

    m = upstream.modules()

    def model_for(batch_size):
        torch.manual_seed(0)
        task = m.task_handlers.ClassificationHandler(str(dataset), nn.CrossEntropyLoss())
        handler = m.model_handlers.DietNetworkHandler(
            task, 0, str(args.bench_dir / "embedding.npz"), device, str(dataset),
            {**CONFIG, "batch_size": batch_size}, None,
        )  # fmt: skip
        handler.model.to(device)
        handler.model.train()
        return handler

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    kinds = ["per-sample", "batched"]
    if genotypes.is_packed(dataset):
        kinds += ["packed-cpu", "packed-gpu"]
    decode = torch.from_numpy(genotypes.DECODE_TABLE).to(device)
    n_snps = mus.numel()

    print("table\tbatch_size\tloader\tworkers\tmissing\tsamples_per_s")
    for b in args.batch_sizes:
        for kind in kinds:
            for w in args.workers:
                rate = timed(loader(kind, dataset, rows, labels, b, w, rng), args.batches)
                print(f"loader\t{b}\t{kind}\t{w}\t-\t{rate:.0f}", flush=True)

        handler = model_for(b)
        model, task = handler.model, handler.task_handler

        def make_step(vectorised, packed_on_gpu=False):
            def step(x, y):
                x = x.to(device, non_blocking=True)
                if packed_on_gpu:
                    x = decode[x.long()].reshape(len(x), -1)[:, :n_snps]
                x = x.float()
                y = task.format_ybatch(y.to(device, non_blocking=True))
                if vectorised:
                    x = torch.where(x >= 0, x, mus)
                else:
                    m.dataset_utils.replace_missing_values(x, mus)
                x = m.dataset_utils.normalize(x, mus, sds)
                for opt in model.get_optimizers():
                    opt.zero_grad()
                loss = task.compute_loss(model.forward(x, "", 0, 0, "train"), y)
                loss.backward()
                for opt in model.get_optimizers():
                    opt.step()

            return step

        best = max(args.workers)
        runs = [("per-sample", 0, False)] + [(kind, best, True) for kind in kinds[1:]]
        if b <= 512:  # upstream's loop over samples, with the fast loader: slow at large b
            runs.insert(1, ("batched", best, False))
        for kind, w, vectorised in runs:
            data = loader(kind, dataset, rows, labels, b, w, rng)
            step = make_step(vectorised, packed_on_gpu=kind == "packed-gpu")
            rate = timed(data, args.batches, step, sync)
            missing = "vectorised" if vectorised else "upstream loop"
            print(f"train\t{b}\t{kind}\t{w}\t{missing}\t{rate:.0f}", flush=True)


if __name__ == "__main__":
    main()
