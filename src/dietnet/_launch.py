"""Run one of upstream's scripts, unmodified, under current torch.

Usage (by :func:`dietnet.upstream.run`): ``python -m dietnet._launch SCRIPT [ARGS...]``.

upstream's train.py saves its validation results (numpy float64 scalars) in its
checkpoint, then reloads it with a bare ``torch.load``. Since torch 2.6 that
call defaults to ``weights_only=True`` and refuses numpy scalars. Rather than
switching the safe loader off, this allowlists exactly the numpy types such a
checkpoint holds -- the remedy torch's own error message gives -- and then runs
the script as ``python SCRIPT`` would, from its own directory.

For train.py, two settings change how it is fed, not what it computes:

- ``DIETNET_INPUTS`` (a :data:`dietnet.genotypes.MODES` store): genotypes come
  from that store instead of train.py's copy of every one in RAM;
  :func:`_store_train_inputs`.
- ``DIETNET_LOADER=fast`` (with ``DIETNET_WORKERS``): each batch is read in one
  call, by loader workers, and missing values are filled in one operation
  instead of a Python loop over the batch's samples; :func:`_fast_loader`.

The batches, their order and every value are upstream's, so the trained model
is bit-identical (tests/integration/test_train_inputs.py).
"""

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

# Which genotype store train.py reads (dietnet.genotypes.MODES); unset: upstream's
# own copy in RAM, untouched.
INPUTS_ENV = "DIETNET_INPUTS"
# "fast": batched reads, loader workers, vectorised missing-value fill.
LOADER_ENV = "DIETNET_LOADER"
WORKERS_ENV = "DIETNET_WORKERS"


def _allow_numpy_scalars() -> None:
    try:
        from numpy._core.multiarray import scalar  # numpy >= 2
    except ImportError:  # pragma: no cover - numpy 1.x
        from numpy.core.multiarray import scalar
    torch.serialization.add_safe_globals(
        [scalar, np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.int64))]
    )


def _store_train_inputs(mode: str) -> None:
    """Make train.py's ``FoldDataset`` read ``inputs`` from a genotype store.

    train.py copies every sample's genotypes into RAM,
    ``FoldDataset.data_x = np.array(FoldDataset.f['inputs'], dtype=np.int8)``,
    and ``FoldDataset.__getitem__`` then reads one row of it per sample. For a
    biobank that copy is hundreds of GB per training job. Here the class train.py
    imports is replaced, before it runs, by a subclass that intercepts the two
    class attributes: ``f`` hands out an empty ``inputs`` (so the copy costs
    nothing) and ``data_x`` becomes the store (:mod:`dietnet.genotypes`), which
    gives the same rows. Rows, values and their order are unchanged; upstream's
    files are untouched.
    """
    import helpers.dataset_utils as du

    from dietnet.genotypes import open_store

    class _File:
        """The opened dataset, except that ``inputs`` is an empty placeholder."""

        def __init__(self, f):
            self._f = f

        def __getitem__(self, key):
            if key == "inputs":
                return np.empty((0,) + self._f[key].shape[1:], dtype=np.int8)
            return self._f[key]

        def __getattr__(self, name):
            return getattr(self._f, name)

    class _Intercept(type):
        def __setattr__(cls, name, value):
            if name == "f":
                value = _File(value)
            elif name == "data_x":
                value = open_store(cls.dataset_file, mode)
            super().__setattr__(name, value)

    class FoldDataset(du.FoldDataset, metaclass=_Intercept):
        pass

    du.FoldDataset = FoldDataset
    print(f"dietnet: train.py reads genotypes from the {mode} store", flush=True)


def _fill_missing(x, per_feature_mean):
    """``du.replace_missing_values``, vectorised: missing (-1) becomes the SNP mean.

    Upstream computes ``mask*x + (~mask)*mean`` row by row in Python, a few GPU
    kernels per sample. One ``where`` gives the same values, in place as upstream.
    """
    x.copy_(torch.where(x >= 0, x, per_feature_mean))


def _fast_loader(workers: int) -> None:
    """Feed train.py's unmodified loops batched reads from loader workers.

    - ``FoldDataset.__getitems__``: PyTorch's DataLoader fetches a whole batch in
      one call when the dataset has it. Rows are read sorted (forward through the
      file) and put back in the sampler's order; the batch is what upstream's
      per-sample reads and default collation give: int8 rows, labels, and the
      samples' row numbers as int64.
    - ``torch.utils.data.DataLoader``, as train.py imports it, becomes a subclass
      that adds ``workers`` processes and pinned memory (train.py fixes
      ``num_workers=0``). Its sampler, shuffling and use of the random generator
      are untouched, so the batches and their order are upstream's.
    - ``du.replace_missing_values`` becomes :func:`_fill_missing`.
    """
    import helpers.dataset_utils as du

    from dietnet.genotypes import read_rows

    class FoldDataset(du.FoldDataset):
        def __getitems__(self, indices):
            rows = np.asarray(self.set_indexes)[np.asarray(indices)]
            x = read_rows(self.data_x, rows)
            y = np.asarray(self.data_y[rows])
            return (
                torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(y),
                torch.from_numpy(rows.astype(np.int64)),
            )

    def _as_is(batch):
        return batch

    class DataLoader(torch.utils.data.DataLoader):
        def __init__(self, dataset, *args, **kwargs):
            if hasattr(dataset, "__getitems__"):
                kwargs.update(collate_fn=_as_is, num_workers=workers)
                kwargs["pin_memory"] = torch.cuda.is_available()
                if workers:
                    # Workers inherit the opened stores instead of being sent them.
                    kwargs["multiprocessing_context"] = "fork"
            super().__init__(dataset, *args, **kwargs)

    du.FoldDataset = FoldDataset
    du.replace_missing_values = _fill_missing
    torch.utils.data.DataLoader = DataLoader
    print(
        f"dietnet: loader reads whole batches with {workers} workers; missing values "
        "filled in one operation",
        flush=True,
    )


def main() -> None:
    script = Path(sys.argv[1]).resolve()
    _allow_numpy_scalars()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    if script.name == "train.py":
        mode = os.environ.get(INPUTS_ENV)
        if mode and mode != "ram":
            _store_train_inputs(mode)
        if os.environ.get(LOADER_ENV) == "fast":
            _fast_loader(int(os.environ.get(WORKERS_ENV, "0")))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
