"""Predict a cohort with one or more model packages, at biobank scale.

The arithmetic is upstream's: its ``DietNetwork`` and its ``indep_test_step``
(imputation with the training means, standardisation, the D/(D-|C|) scale,
forward pass, softmax), which is what upstream's test_independent_dataset.py
runs; tests/parity checks the two give identical scores.

What differs is the I/O around it. Upstream's script loads the whole cohort
into memory, matches SNPs with a Python loop, and does both again for every
model. Here the cohort is matched once (allele-aware, :mod:`dietnet.align`) into
an HDF5 file under ``work_dir``, then memory-mapped -- possible because the file
is contiguous -- and read once, a contiguous batch of samples at a time, with
every model scoring each batch before the next is read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from dietnet import align, dataset, upstream
from dietnet.package import Package


def device_for(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def matched_dataset(packages: list[Package], cohort_prefix, work_dir) -> Path:
    """The cohort on the models' SNPs: built once, reused while inputs are unchanged."""
    snps = packages[0].snps()
    for pkg in packages[1:]:
        if not pkg.snps()["name"].equals(snps["name"]):
            raise ValueError(f"{pkg.path} was trained on different SNPs than {packages[0].path}")

    out = (
        Path(work_dir)
        / f"{Path(str(cohort_prefix)).name}.{align.fingerprint(snps, cohort_prefix)}.matched.h5"
    )
    if not out.exists():
        align.write_matched(cohort_prefix, snps, out)
    return out


@dataclass
class Result:
    samples: np.ndarray
    model_names: list[str]
    classes: list[str]
    scores: np.ndarray  # models x samples x classes (classification) or models x samples
    alignment: dict
    scale: float

    def table(self) -> pd.DataFrame:
        """One row per sample: the ensemble's call and how much the models agree."""
        out = pd.DataFrame({"sample": self.samples})
        if not self.classes:
            out["prediction"] = self.scores.mean(axis=0)
            out["sd_across_models"] = self.scores.std(axis=0)
            return out
        mean = self.scores.mean(axis=0)
        call = mean.argmax(axis=1)
        votes = self.scores.argmax(axis=2)
        out["prediction"] = np.asarray(self.classes)[call]
        out["probability"] = mean[np.arange(len(call)), call]
        out["model_agreement"] = (votes == call[None]).mean(axis=0)
        return out


def predict(packages: list[Package], matched: Path, batch_size: int = 1024, device=None) -> Result:
    m = upstream.modules()
    device = device_for(device)
    with h5py.File(matched, "r") as f:
        samples = np.array(f["samples"]).astype(str)
        scale = float(f["scale"][0])
        alignment = json.loads(f.attrs.get("alignment", "{}"))

    tasks = {p.task for p in packages}
    if len(tasks) != 1:
        raise ValueError(f"packages mix tasks: {sorted(tasks)}")
    classes = packages[0].classes
    for pkg in packages[1:]:
        if pkg.classes != classes:
            raise ValueError(f"{pkg.path} predicts different classes than {packages[0].path}")

    models = [(pkg.handler(device), *pkg.stats(device), pkg.normalize) for pkg in packages]
    n = len(samples)
    shape = (len(packages), n, len(classes)) if classes else (len(packages), n)
    scores = np.empty(shape, dtype=np.float64)

    # Each batch of samples is read once, as one contiguous slice of the memory
    # map, and every model scores it with upstream's indep_test_step -- handed a
    # one-batch loader, so the arithmetic is upstream's while the cohort is read
    # once in total rather than once per model, row by row.
    inputs = dataset.memmap_inputs(matched)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = [(torch.arange(start, stop), torch.from_numpy(np.asarray(inputs[start:stop])))]
            for i, (handler, mus, sds, normalize) in enumerate(models):
                results = m.mainloop_utils.indep_test_step(
                    handler, device, range(stop - start), batch, mus, sds, normalize, None, 0, scale
                )
                if not np.array_equal(results["samples"].astype(int), np.arange(start, stop)):
                    raise RuntimeError("upstream's inference step returned samples out of order")
                scores[i, start:stop] = results["scores"] if classes else results["preds"]

    return Result(
        samples=samples,
        model_names=[p.name for p in packages],
        classes=classes,
        scores=scores,
        alignment=alignment,
        scale=scale,
    )
