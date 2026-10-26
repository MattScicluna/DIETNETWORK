"""Shared fixtures."""

import os
from pathlib import Path

import pytest
from support import DATASETS, Dataset, train_pipeline


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
def trained_synthetic(tmp_path_factory) -> Path:
    """Packages for folds 0 and 1 on the synthetic fixture, trained once per session."""
    return train_pipeline(tmp_path_factory.mktemp("trained_synthetic"), DATASETS["synthetic"])


@pytest.fixture(scope="session")
def upstream_checkout() -> Path:
    """A full HussinLab/DIETNETWORK checkout at the pinned commit.

    This repository keeps only the upstream files its API runs. The parity
    tests compare it with upstream scripts it does not keep (create_dataset.py,
    test_independent_dataset.py, Dietnet_inference/), so they need the whole
    tree: DIETNET_UPSTREAM_DIR, which CI sets after cloning UPSTREAM_SHA.
    """
    root = os.environ.get("DIETNET_UPSTREAM_DIR")
    if not root:
        pytest.skip("DIETNET_UPSTREAM_DIR is not set (see tests/README.md)")
    root = Path(root)
    if not (root / "Dietnet_train_and_evaluate" / "test_independent_dataset.py").is_file():
        pytest.fail(f"DIETNET_UPSTREAM_DIR={root} is not a full upstream checkout")
    return root


@pytest.fixture(scope="session")
def published_packages(tmp_path_factory):
    from dietnet import published

    cache = os.environ.get("DIETNET_PUBLISHED_DIR")
    root = Path(cache) if cache else tmp_path_factory.mktemp("published")
    release = published.fetch(root / "download")
    return published.import_training_release(release, root / "packages")
