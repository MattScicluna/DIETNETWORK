"""Edge cases of model packages: building without optional parts, and finding none."""

import io
import json

import h5py
import numpy as np
import pytest

from dietnet import package, published


@pytest.fixture
def header_only(tmp_path):
    """A training dataset with SNP names only: no class labels and no ``snps/`` group."""
    path = tmp_path / "train.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("snp_names", data=np.array([b"rs1_A", b"rs2_C", b"rs3_G"]))
    return path


def _build(out, train_dataset, tmp_path, **overrides):
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    options = dict(
        checkpoint=checkpoint,
        config={"seed": 5},
        embedding=None,
        means=np.zeros(3),
        sds=None,
        train_dataset=train_dataset,
        task="regression",
        normalize=False,
        seed=5,
        fold=2,
        model="Mlp",
    )
    options.update(overrides)
    return package.build(out, **options)


def test_build_without_embedding_sds_or_class_labels(tmp_path, header_only):
    """A regression MLP package keeps only what it has: no embedding, means but no SDs."""
    pkg = _build(tmp_path / "pkg", header_only, tmp_path)

    assert pkg.task == "regression" and pkg.classes == [] and pkg.model == "Mlp"
    assert pkg.metadata["n_snps"] == 3 and pkg.name == "seed5_fold2"
    assert pkg.metadata["format_version"] == package.FORMAT_VERSION
    assert not (pkg.path / "embedding.npz").exists()
    stats = np.load(pkg.path / "input_features_stats.npz")
    assert set(stats.files) == {"means_by_fold"}
    with h5py.File(pkg.path / "train_header.h5", "r") as f:
        assert set(f) == {"snp_names"}
    mus, sds = pkg.stats("cpu")
    assert mus.shape == (3,) and sds is None


def test_build_replaces_an_existing_package(tmp_path, header_only):
    """Building into an existing directory removes what was there first."""
    out = tmp_path / "pkg"
    out.mkdir()
    (out / "stale.txt").write_text("left over")

    _build(out, header_only, tmp_path)

    assert not (out / "stale.txt").exists()
    assert (out / "metadata.json").is_file()


def test_build_rejects_an_embedding_of_the_wrong_size(tmp_path, header_only):
    """An embedding needs one row per SNP of the training dataset."""
    with pytest.raises(ValueError, match="embedding has 2 rows for 3 SNPs"):
        _build(tmp_path / "pkg", header_only, tmp_path, embedding=np.ones((2, 4)))


def test_extra_metadata_is_kept(tmp_path, header_only):
    """Keys passed as ``extra`` land in metadata.json next to the standard ones."""
    pkg = _build(tmp_path / "pkg", header_only, tmp_path, extra={"replicate": 2})
    assert pkg.metadata["replicate"] == 2


def test_a_directory_without_metadata_is_not_a_package(tmp_path):
    with pytest.raises(FileNotFoundError, match="is not a model package"):
        package.Package(tmp_path)


def test_find_with_no_packages_raises(tmp_path):
    (tmp_path / "empty" / "nested").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no model packages under"):
        package.find(tmp_path / "empty")


def test_find_orders_by_seed_then_fold(tmp_path, header_only):
    """Packages come back in (seed, fold) order, whatever their directory names."""
    for name, seed, fold in (("a", 9, 0), ("b", 1, 3), ("c", 1, 0)):
        _build(tmp_path / "pkgs" / name, header_only, tmp_path, seed=seed, fold=fold)

    found = package.find(tmp_path / "pkgs")

    assert [p.name for p in found] == ["seed1_fold0", "seed1_fold3", "seed9_fold0"]


def test_find_accepts_a_single_package(tmp_path, header_only):
    pkg = _build(tmp_path / "pkg", header_only, tmp_path)
    assert package.find(pkg.path) == [pkg]


# The published release, without the network


def _no_network(*args, **kwargs):
    raise AssertionError("no download expected")


def test_fetch_reuses_an_unpacked_release(tmp_path, monkeypatch):
    """A release already unpacked under ``dest`` is returned without a download."""
    (tmp_path / "Dietnet_training" / "supporting_files").mkdir(parents=True)
    monkeypatch.setattr(published.urllib.request, "urlopen", _no_network)
    monkeypatch.setattr(published.urllib.request, "urlretrieve", _no_network)

    assert published.fetch(tmp_path) == tmp_path / "Dietnet_training"


def test_fetch_refuses_an_archive_with_the_wrong_checksum(tmp_path, monkeypatch):
    """A download that does not match Zenodo's checksum is not unpacked."""
    record = {
        "files": [
            {"key": "other.txt", "checksum": "md5:0"},
            {"key": "Dietnet_training.tar.gz", "checksum": "md5:" + "0" * 32},
        ]
    }
    monkeypatch.setattr(
        published.urllib.request, "urlopen", lambda url: io.BytesIO(json.dumps(record).encode())
    )
    monkeypatch.setattr(
        published.urllib.request, "urlretrieve", lambda url, path: path.write_bytes(b"corrupt")
    )

    with pytest.raises(ValueError, match=r"md5 [0-9a-f]{32} != 0{32} from Zenodo"):
        published.fetch(tmp_path / "dest")
    assert not (tmp_path / "dest" / "Dietnet_training").exists()
