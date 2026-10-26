"""Run one of upstream's scripts, unmodified, under current torch.

Usage (by :func:`dietnet.upstream.run`): ``python -m dietnet._launch SCRIPT [ARGS...]``.

upstream's train.py saves its validation results (numpy float64 scalars) in its
checkpoint, then reloads it with a bare ``torch.load``. Since torch 2.6 that
call defaults to ``weights_only=True`` and refuses numpy scalars. Rather than
switching the safe loader off, this allowlists exactly the numpy types such a
checkpoint holds -- the remedy torch's own error message gives -- and then runs
the script as ``python SCRIPT`` would, from its own directory.
"""

import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def _allow_numpy_scalars() -> None:
    try:
        from numpy._core.multiarray import scalar  # numpy >= 2
    except ImportError:  # pragma: no cover - numpy 1.x
        from numpy.core.multiarray import scalar
    torch.serialization.add_safe_globals(
        [scalar, np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.int64))]
    )


def main() -> None:
    script = Path(sys.argv[1]).resolve()
    _allow_numpy_scalars()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
