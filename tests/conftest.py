"""Shared fixtures."""

import importlib
import os
import sys
from pathlib import Path

import pytest
from support import DATASETS, Dataset


@pytest.fixture(params=sorted(DATASETS))
def dataset(request) -> Dataset:
    return DATASETS[request.param]


@pytest.fixture
def synthetic() -> Dataset:
    return DATASETS["synthetic"]


@pytest.fixture
def kgp_mini() -> Dataset:
    return DATASETS["kgp_mini"]


@pytest.fixture(scope="session")
def upstream():
    """Import HussinLab/DIETNETWORK's training code from a checkout.

    Pointed at by DIETNET_UPSTREAM_DIR, which CI sets after cloning the pinned
    commit. Imported live rather than compared against stored outputs: dropout
    masks and Adam updates depend on the installed torch, so outputs stored
    under one version would fail under another for reasons that have nothing to
    do with this repository.

    Upstream's scripts import their helpers as a top-level ``helpers`` package,
    so its directory goes on sys.path. Nothing in this repository is called
    ``helpers`` at top level, so nothing is shadowed.
    """
    root = os.environ.get("DIETNET_UPSTREAM_DIR")
    if not root:
        pytest.skip("DIETNET_UPSTREAM_DIR is not set (see tests/README.md)")

    code = Path(root) / "Dietnet_train_and_evaluate"
    if not code.is_dir():
        pytest.fail(f"DIETNET_UPSTREAM_DIR={root} has no Dietnet_train_and_evaluate/")

    sys.path.insert(0, str(code))
    try:
        yield {
            name: importlib.import_module(name)
            for name in (
                "helpers.model",
                "helpers.model_handlers",
                "helpers.task_handlers",
                "helpers.mainloop_utils",
                "helpers.dataset_utils",
                "generate_embedding",
                "compute_input_features_statistics",
            )
        }
    finally:
        sys.path.remove(str(code))
