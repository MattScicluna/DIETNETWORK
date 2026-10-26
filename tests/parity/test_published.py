"""The published models reproduce their published held-out accuracy.

Each of the 15 models is scored on its own test fold of the published split
(reproduction/1kgp_partition_seed124.npz, verified against the Zenodo
release's per-fold statistics) using the published inputs. The paper's
Figure 1A puts the mean at about 0.92; these are the exact per-model values,
so a change anywhere in the prediction path that moves a single test sample
shows up here.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from dietnet import predict as prediction

PARTITION = Path(__file__).resolve().parents[2] / "reproduction" / "1kgp_partition_seed124.npz"

# (seed, fold) -> held-out accuracy, from reproduction/README.md.
PUBLISHED = {
    (78, 0): 0.9185, (79, 0): 0.9206, (80, 0): 0.9356,
    (78, 1): 0.9224, (79, 1): 0.9073, (80, 1): 0.9289,
    (78, 2): 0.9336, (79, 2): 0.9315, (80, 2): 0.9465,
    (78, 3): 0.9289, (79, 3): 0.9095, (80, 3): 0.9246,
    (78, 4): 0.9089, (79, 4): 0.9002, (80, 4): 0.9046,
}  # fmt: skip


@pytest.mark.slow
@pytest.mark.network
def test_held_out_accuracy_of_the_published_models(tmp_path, published_packages):
    release = Path(published_packages[0].path).parents[2] / "download" / "Dietnet_training"
    source = release / "1KGP_dataset.hdf5"

    # The published inputs in upstream's matched-dataset schema, as they are.
    matched = tmp_path / "published.matched.h5"
    with h5py.File(source) as f, h5py.File(matched, "w") as g:
        for key in ("inputs", "samples", "snp_names"):
            g.create_dataset(key, data=np.array(f[key]))
        g.create_dataset("scale", data=np.array([1.0]))
        labels = np.array(f["class_label_names"]).astype(str)[np.array(f["class_labels"])]

    result = prediction.predict(published_packages, matched, batch_size=512, device="cpu")
    folds = np.load(PARTITION, allow_pickle=True)["folds_indexes"]

    accuracy = {}
    for i, pkg in enumerate(published_packages):
        seed, fold = pkg.metadata["seed"], pkg.metadata["fold"]
        test = np.asarray(folds[fold][2])
        calls = np.asarray(result.classes)[result.scores[i][test].argmax(axis=1)]
        accuracy[(seed, fold)] = (calls == labels[test]).mean()

    # One test sample is ~0.0022; allow a single flip from floating-point noise.
    for key, expected in PUBLISHED.items():
        assert accuracy[key] == pytest.approx(expected, abs=0.0025), key
    assert np.mean(list(accuracy.values())) == pytest.approx(0.9214, abs=0.001)
