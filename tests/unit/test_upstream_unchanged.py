"""The upstream code this package runs is HussinLab's, byte for byte.

"We run the published code" is only true while nobody edits it here. Any change
to a kept file, or a new file in upstream's directory, fails this test; moving
to a newer upstream commit means merging it and regenerating the manifest
(scripts/upstream_manifest.py), never editing in place.
"""

from support import ci_pinned_commit, manifest_commit

from dietnet import upstream


def test_kept_upstream_files_are_identical_to_the_pinned_commit():
    assert upstream.verify() == []


def test_removed_upstream_files_stay_removed():
    present = [name for name in upstream.manifest()["removed"] if (upstream.root() / name).exists()]
    assert present == [], "re-added upstream files: delete them or record them in the manifest"


def test_ci_compares_against_the_same_commit_the_manifest_pins():
    """The parity tests clone upstream in CI; they must clone the commit we run."""
    assert ci_pinned_commit() == manifest_commit()


def test_results_directory_name_comes_from_upstream():
    """training.py finds train.py's output by upstream's own naming method."""
    config = {
        "nb_hidden_u_aux": [100, 100],
        "nb_hidden_u_main": [100],
        "input_dropout": 0.995,
        "dropout_main": 0.1,
        "lr_aux": 0.0006,
        "lr_main": 0.0003,
        "learning_rate_annealing": 0.99,
        "uniform_init_limit": 0.02,
        "epochs": 8000,
        "patience": 2000,
        "seed": 78,
    }
    name = upstream.exp_identifier(config, 2, "classification")
    assert name.startswith("classification_auxu_100_100_mainu_100_100_inpdrop_0.995")
    assert name.endswith("_seed_78_fold2")
