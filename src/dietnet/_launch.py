"""Run one of upstream's scripts, unmodified, under current torch.

Usage (by :func:`dietnet.upstream.run`): ``python -m dietnet._launch SCRIPT [ARGS...]``.

upstream's train.py saves its validation results (numpy float64 scalars) in its
checkpoint, then reloads it with a bare ``torch.load``. Since torch 2.6 that
call defaults to ``weights_only=True`` and refuses numpy scalars. Rather than
switching the safe loader off, this allowlists exactly the numpy types such a
checkpoint holds -- the remedy torch's own error message gives -- and then runs
the script as ``python SCRIPT`` would, from its own directory.

With ``DIETNET_INPUTS=memmap``, train.py reads genotypes from a memory map of
the dataset instead of a copy in RAM; see :func:`_store_train_inputs`.
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


def main() -> None:
    script = Path(sys.argv[1]).resolve()
    _allow_numpy_scalars()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    mode = os.environ.get(INPUTS_ENV)
    if script.name == "train.py" and mode and mode != "ram":
        _store_train_inputs(mode)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
