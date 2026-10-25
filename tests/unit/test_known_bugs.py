"""Known bugs, each recorded as a strict xfail named after docs/refactor-plan.md.

These describe the behaviour the code should have. A fix makes its test pass,
and because the markers are strict the same change must delete the marker --
so this file is always an accurate list of what is still broken.

Bugs that only show against upstream (B1, B3) live in tests/parity; the ones
that need a trained model package (B4, B16) in tests/integration.
"""

import numpy as np
import pytest
import torch
import yaml

from Dietnet import cli
from Dietnet.helpers import dataset_utils as du
from Dietnet.helpers import model
from Dietnet.helpers.model_package import ModelPackage, convert_legacy_model
from Dietnet.predict_with_plink import load_model_from_package

CONFIG = {
    "nb_hidden_u_aux": [8, 8],
    "nb_hidden_u_main": [8],
    "uniform_init_limit": 0.02,
    "input_dropout": 0.5,
}


def small_model(n_feats_emb=12, n_targets=4):
    return model.CombinedModel(
        n_feats=n_feats_emb,
        n_hidden_u_aux=CONFIG["nb_hidden_u_aux"],
        n_hidden_u_main=CONFIG["nb_hidden_u_aux"][-1:] + CONFIG["nb_hidden_u_main"],
        n_targets=n_targets,
        param_init=None,
        aux_uniform_init_limit=CONFIG["uniform_init_limit"],
    )


@pytest.mark.xfail(reason="B2: load_embedding always indexes [fold], so a 2-D embedding loses rows")
def test_fold_independent_embedding_loads_whole(tmp_path):
    """A per-SNP embedding shared by all folds -- what a foundation model gives."""
    path = tmp_path / "embedding.npz"
    np.savez(path, emb=np.random.rand(50, 6))

    assert du.load_embedding(path, 0).shape == (50, 6)


@pytest.mark.xfail(reason="B2: _select_embedding_for_fold slices axis 0 whenever it can")
def test_fold_independent_embedding_is_packaged_whole(tmp_path):
    path = tmp_path / "embedding.npz"
    np.savez(path, emb=np.random.rand(50, 6))

    assert cli._select_embedding_for_fold(np.load(path), 0).shape == (50, 6)


def test_per_fold_embedding_selects_the_fold(tmp_path):
    """The case that works today, kept working while B2 is fixed."""
    path = tmp_path / "embedding.npz"
    emb = np.random.rand(5, 50, 6)
    np.savez(path, emb=emb)

    np.testing.assert_array_equal(du.load_embedding(path, 3).numpy(), emb[3])
    np.testing.assert_array_equal(cli._select_embedding_for_fold(np.load(path), 3), emb[3])


@pytest.fixture
def legacy_inputs(tmp_path, synthetic):
    """What a pre-package training run left behind: weights, stats, embedding, config."""
    genotypes, _, _ = synthetic.upstream_arrays()
    n_snps = genotypes.shape[1]

    model_path = tmp_path / "best_model.pt"
    torch.save(small_model(n_targets=4).state_dict(), model_path)

    stats_path = tmp_path / "input_features_means.npz"
    mean, sd = du.compute_norm_values(torch.from_numpy(genotypes.astype(np.float32)))
    # The layout compute_input_features_mean.py writes.
    np.savez(stats_path, means_by_fold=[mean.numpy()] * 2, sd_by_fold=[sd.numpy()] * 2)

    emb_path = tmp_path / "embedding.npz"
    # The layout generate_embedding.py writes: key 'emb', one matrix per fold.
    np.savez(emb_path, emb=np.random.rand(2, n_snps, 12))

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))

    return dict(
        model_path=model_path,
        dataset_file=synthetic.upstream_hdf5,
        embedding_file=emb_path,
        input_stats_file=stats_path,
        config_file=config_path,
        output_dir=tmp_path / "package",
        seed=1,
        fold=1,
    )


@pytest.mark.xfail(
    reason="B5: convert_legacy_model reads 'stds_by_fold'; the file has 'sd_by_fold'"
)
def test_legacy_conversion_keeps_standard_deviations(legacy_inputs):
    package = convert_legacy_model(**legacy_inputs)

    assert "std" in package.input_stats


@pytest.mark.xfail(reason="B5: convert_legacy_model ignores the 'emb' key and packages every fold")
def test_legacy_conversion_packages_one_folds_embedding(legacy_inputs):
    package = convert_legacy_model(**legacy_inputs)

    assert package.embedding.ndim == 2


@pytest.mark.xfail(reason="B10: the genotype cache is only written above 2 GB")
def test_genotype_cache_is_written_for_small_datasets(tmp_path, synthetic):
    cache = tmp_path / "genotypes.npy"
    du.load_plink_genotypes(str(synthetic.plink_prefix), cache_file=cache, verbose=False)

    assert cache.exists()


@pytest.mark.xfail(reason="B12: CombinedModel registers its sub-networks twice")
def test_state_dict_has_no_duplicate_parameters():
    keys = small_model().state_dict().keys()

    assert not [k for k in keys if k.startswith(("feat_emb.", "disc_net."))]


def save_package(path, state):
    package = ModelPackage(path)
    package.save(
        model_state=state,
        snps=[f"snp{i}" for i in range(30)],
        input_stats={"mean": np.ones(30), "std": np.ones(30)},
        embedding=np.random.rand(30, 12),
        label_mapping={f"POP{i}": i for i in range(4)},
        config=CONFIG,
        seed=1,
        fold=0,
    )
    return ModelPackage.load(path)


def test_complete_package_loads(tmp_path):
    reference = small_model()
    loaded = load_model_from_package(save_package(tmp_path / "pkg", reference.state_dict()))

    for key, value in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


@pytest.mark.xfail(reason="B15: load_model_from_package loads with strict=False")
def test_incomplete_package_refuses_to_load(tmp_path):
    """A checkpoint missing a layer must not quietly predict with random weights."""
    state = {
        k: v
        for k, v in small_model().state_dict().items()
        if not k.startswith(("main_net.out.", "disc_net.out."))
    }
    package = save_package(tmp_path / "pkg", state)

    with pytest.raises(RuntimeError):
        load_model_from_package(package)
