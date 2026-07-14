import math

import pytest
import torch
from torch import nn


def test_word_random_forest_oob_random_search_space_is_exact():
    import backend.app.training as training

    candidates = training._traditional_candidate_configs(
        training.TrainConfig(),
        "random_forest",
        1000,
        [0, 1] * 50,
    )

    assert len(candidates) == 10
    assert len({(item.random_forest_max_depth, item.random_forest_min_samples_leaf, item.random_forest_max_features) for item in candidates}) == 10
    assert {item.random_forest_n_estimators for item in candidates} == {200}
    assert {item.random_forest_max_depth for item in candidates} <= {3, 5, 10}
    assert {item.random_forest_min_samples_leaf for item in candidates} <= {2, 5}
    assert {item.random_forest_max_features for item in candidates} <= {"sqrt", "log2", 0.1}
    assert all(item.random_forest_oob_score for item in candidates)


@pytest.mark.parametrize(
    ("model_type", "sample_count"),
    [
        ("cnn1d", 100),
        ("cnn1d_se", 101),
        ("resnet1d", 299),
        ("inception1d", 300),
        ("tcn1d", 100),
        ("cnn_transformer1d", 101),
    ],
)
def test_word_v2_binary_models_emit_single_logit(model_type, sample_count):
    from backend.app.models.registry import build_deep_model
    from backend.app.training import TrainConfig

    model = build_deep_model(
        TrainConfig(model_type=model_type),
        input_length=64,
        class_count=2,
        sample_count=sample_count,
    )

    assert model(torch.randn(3, 1, 64)).shape == (3, 1)


@pytest.mark.parametrize(
    ("feature_count", "kernels", "pools"),
    [
        (1000, [7, 5, 3], [2, 2, 2]),
        (1001, [9, 5, 3], [4, 2, 2]),
        (2999, [9, 5, 3], [4, 2, 2]),
        (3000, [9, 7, 5], [4, 4, 2]),
    ],
)
def test_word_cnn_feature_boundaries_are_exact(feature_count, kernels, pools):
    from backend.app.models.profiles import build_model_profile

    profile = build_model_profile("cnn1d", train_sample_count=100, feature_count=feature_count)

    assert profile.values["kernels"] == kernels
    assert profile.values["pools"] == pools


@pytest.mark.parametrize(
    ("sample_count", "expected"),
    [
        (100, {"conv_channels": [8, 16, 32], "d_model": 32, "heads": 2, "layers": 1, "ffn": 64}),
        (101, {"conv_channels": [16, 32, 64], "d_model": 64, "heads": 4, "layers": 1, "ffn": 128}),
        (299, {"conv_channels": [16, 32, 64], "d_model": 64, "heads": 4, "layers": 1, "ffn": 128}),
        (300, {"conv_channels": [32, 64, 128], "d_model": 128, "heads": 4, "layers": 2, "ffn": 256}),
    ],
)
def test_word_cnn_transformer_sample_boundaries_are_exact(sample_count, expected):
    from backend.app.models.profiles import build_model_profile

    profile = build_model_profile("cnn_transformer1d", train_sample_count=sample_count, feature_count=1000)

    for key, value in expected.items():
        assert profile.values[key] == value


@pytest.mark.parametrize(
    ("feature_count", "stem", "branches", "pools"),
    [
        (1000, 7, [1, 3, 5, 7], [2, 2, 2]),
        (1001, 9, [1, 3, 7, 11], [4, 2, 2]),
        (2999, 9, [1, 3, 7, 11], [4, 2, 2]),
        (3000, 11, [1, 5, 9, 15], [4, 4, 2]),
    ],
)
def test_word_inception_feature_boundaries_are_exact(feature_count, stem, branches, pools):
    from backend.app.models.profiles import build_model_profile

    profile = build_model_profile("inception1d", train_sample_count=100, feature_count=feature_count)

    assert profile.values["stem_kernel"] == stem
    assert profile.values["branch_kernels"] == branches
    assert profile.values["pools"] == pools


@pytest.mark.parametrize(
    ("sample_count", "channels"),
    [(100, 32), (101, 64), (299, 64), (300, 128)],
)
def test_word_tcn_sample_boundaries_are_exact(sample_count, channels):
    from backend.app.models.profiles import build_model_profile

    profile = build_model_profile("tcn1d", train_sample_count=sample_count, feature_count=1000)
    assert profile.values["channels"] == channels


@pytest.mark.parametrize(
    ("feature_count", "stem", "pool", "dilations"),
    [
        (1000, 7, 2, [1, 2, 4]),
        (1001, 9, 4, [1, 2, 4]),
        (2999, 9, 4, [1, 2, 4]),
        (3000, 9, 4, [1, 2, 4, 8]),
    ],
)
def test_word_tcn_feature_boundaries_are_exact(feature_count, stem, pool, dilations):
    from backend.app.models.profiles import build_model_profile

    profile = build_model_profile("tcn1d", train_sample_count=100, feature_count=feature_count)
    assert profile.values["stem_kernel"] == stem
    assert profile.values["stem_pool"] == pool
    assert profile.values["dilations"] == dilations


def test_word_resnet_has_only_three_document_pools_and_plain_projection_shortcuts():
    from backend.app.models.resnet1d_v2 import ResNet1DDocumentV2

    model = ResNet1DDocumentV2(input_length=3000, class_count=3, sample_count=300)

    pools = [module for module in model.modules() if isinstance(module, nn.MaxPool1d)]
    assert len(pools) == 3
    assert model.blocks[-1].pool.__class__ is nn.Identity
    assert isinstance(model.blocks[1].shortcut, nn.Conv1d)


def test_word_dscarnet_profile_and_mode_contract_is_available():
    from backend.app.models import profiles

    assert hasattr(profiles, "build_dscarnet_profile")
    profile = profiles.build_dscarnet_profile(train_sample_count=100, feature_count=3000)
    expected_n = math.ceil((math.sqrt(8 * (0.8**2) * 3000 + 1) - 1) / 2)
    assert profile["pca_components"] == min(expected_n, 99, 3000)
    assert profile["cluster_channels"] == 7
    assert profile["conv1_kernel_size"] == 19
    assert profile["filter_number"] == 16
    assert profile["n_inception"] == 1
    assert profile["dense_layers"] == [32]

    from backend.app.training import TrainConfig

    assert TrainConfig().dscarnet_input_mode == "dual"


@pytest.mark.parametrize("mode", ["sar", "car", "dual"])
def test_word_dscarnet_mapping_accepts_all_three_modes(mode):
    from inspect import signature

    from backend.app.dscarnet_mapping import fit_dscarnet_2d_mapping

    assert "mode" in signature(fit_dscarnet_2d_mapping).parameters
