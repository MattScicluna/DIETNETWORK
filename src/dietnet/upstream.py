"""The bridge to HussinLab/DIETNETWORK's code, which this package runs unmodified.

Upstream's training code is a set of scripts (``train.py``, ``partition_data.py``,
...) that import their helpers as a top-level ``helpers`` package. Two ways in:

* :func:`run` executes a script as upstream's README does, in a subprocess.
  Used for everything that produces a published artefact: partitions,
  embeddings, input statistics, trained weights.
* :func:`modules` imports upstream modules in this process. Used where this
  package needs the model class and inference step without upstream's I/O
  (biobank-scale prediction), so that the arithmetic is still upstream's.

:func:`verify` checks the files against the manifest of the commit they came
from (``upstream_manifest.json``); ``dietnet info`` reports it, and the test
suite fails on any difference.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

_PACKAGE = Path(__file__).resolve().parent
_MANIFEST = _PACKAGE / "upstream_manifest.json"
TRAIN_DIR = "Dietnet_train_and_evaluate"


class UpstreamError(RuntimeError):
    """An upstream script failed; the message carries the tail of its log."""


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads(_MANIFEST.read_text())


def commit() -> str:
    return manifest()["commit"]


@lru_cache(maxsize=1)
def root() -> Path:
    """The directory holding upstream's code.

    Inside an installed wheel it is ``dietnet/_upstream``; in a source checkout
    it is the repository root, where upstream's directories live so that
    ``git merge upstream/main`` keeps working.
    """
    for candidate in (_PACKAGE / "_upstream", _PACKAGE.parents[1]):
        if (candidate / TRAIN_DIR / "train.py").is_file():
            return candidate
    raise FileNotFoundError(f"{TRAIN_DIR}/ not found next to {_PACKAGE}")


def verify() -> list[str]:
    """Upstream files that differ from, or are missing against, the manifest."""
    expected = manifest()["files"]
    problems = []
    for name, digest in expected.items():
        path = root() / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"modified: {name}")
    for directory in manifest()["directories"]:
        for path in sorted((root() / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                name = path.relative_to(root()).as_posix()
                if name not in expected:
                    problems.append(f"added: {name}")
    return problems


def script(name: str) -> Path:
    return root() / TRAIN_DIR / name


def run(name: str, args: list, log: Path | None = None) -> None:
    """Run upstream script ``name`` with ``args``, as its README does.

    The working directory is the script's own, for its ``import helpers``.
    PYTHONHASHSEED is fixed because partition_data.py iterates over ``set(labels)``:
    with Python's default per-process string hashing, the order of samples
    within a fold -- and with it the training trajectory under a fixed seed --
    would change from run to run.
    """
    # Through dietnet._launch, which runs the script unmodified after letting
    # torch's safe loader read the numpy scalars in train.py's checkpoints.
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "dietnet._launch",
        str(script(name)),
        *[str(a) for a in args],
    ]
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    if log is None:
        result = subprocess.run(cmd, cwd=script(name).parent, env=env)
        if result.returncode != 0:
            raise UpstreamError(f"{name} exited with status {result.returncode}")
        return

    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as handle:
        result = subprocess.run(
            cmd, cwd=script(name).parent, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    if result.returncode != 0:
        tail = "".join(log.read_text().splitlines(keepends=True)[-25:])
        raise UpstreamError(f"{name} exited with status {result.returncode}; log {log}:\n{tail}")


_MODULES = (
    "helpers.dataset_utils",
    "helpers.mainloop_utils",
    "helpers.model",
    "helpers.model_handlers",
    "helpers.task_handlers",
)


@contextmanager
def _on_path():
    path = str(root() / TRAIN_DIR)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)


@lru_cache(maxsize=1)
def modules() -> SimpleNamespace:
    """Upstream's training modules, imported in this process.

    ``helpers`` is upstream's top-level package name; nothing in this package
    uses it, so nothing is shadowed. The modules stay imported (and cached)
    after the path entry is removed.
    """
    with _on_path():
        loaded = {name.split(".")[-1]: importlib.import_module(name) for name in _MODULES}
    return SimpleNamespace(**loaded)


def exp_identifier(config: dict, fold: int, task: str) -> str:
    """The results-directory name upstream's train.py uses, from upstream's own method."""
    handler = SimpleNamespace(task_handler=SimpleNamespace(name=task))
    return modules().model_handlers.DietNetworkHandler.get_exp_identifier(handler, config, fold)
