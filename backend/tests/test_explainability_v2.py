import pytest
import numpy as np


@pytest.mark.parametrize(
    "model_type",
    ["pls_da", "pca_lda", "logistic_regression", "svm", "random_forest", "xgboost"],
)
def test_traditional_models_use_window_permutation(model_type):
    from backend.app.training_explainability import explainability_method

    assert explainability_method(model_type) == "window_permutation"


@pytest.mark.parametrize("model_type", ["pca_mlp", "transformer1d", "cnn_transformer1d", "cnn_mamba1d"])
def test_long_distance_models_use_input_gradient(model_type):
    from backend.app.training_explainability import explainability_method

    assert explainability_method(model_type) == "input_gradient_attribution"


@pytest.mark.parametrize("model_type", ["cnn1d", "cnn1d_se", "resnet1d", "inception1d", "tcn1d"])
def test_convolution_models_use_gradcam(model_type):
    from backend.app.training_explainability import explainability_method

    assert explainability_method(model_type) == "gradcam_1d"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("sar", "dscarnet_sar_2d_gradcam"),
        ("car", "dscarnet_car_2d_gradcam"),
        ("dual", "dscarnet_dual_2d_gradcam"),
    ],
)
def test_dscarnet_modes_use_explicit_2d_routing(mode, expected):
    from backend.app.training_explainability import explainability_method

    assert explainability_method("dscarnet", dscarnet_mode=mode) == expected


def test_unknown_explainability_model_is_rejected():
    from backend.app.training_explainability import explainability_method

    with pytest.raises(ValueError, match="未知模型"):
        explainability_method("unknown_model")


def test_pca_mlp_attribution_keeps_original_feature_axis():
    from backend.app.feature_selection import sample_deep_attribution_importance
    from backend.app.models.registry import build_deep_model
    from backend.app.training import TrainConfig

    x = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 2.0, 3.0, 2.0, 3.0, 2.0],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 1, 0, 1], dtype=np.int64)
    model = build_deep_model(
        TrainConfig(model_type="pca_mlp"),
        input_length=6,
        class_count=2,
        sample_count=2,
        x_train=x[:2],
    )
    result = sample_deep_attribution_importance(
        model,
        x,
        y,
        x_axis=np.arange(6, dtype=np.float32),
        splits={"train": [0, 1], "valid": [], "test": [2, 3]},
        label_names=["A", "B"],
        top_k=2,
        model_type="pca_mlp",
    )

    assert result["method"] == "input_gradient_attribution"
    assert result["importance_metric"] == "absolute_gradient_x_input"
    assert all(len(sample["windows"]) == 6 for sample in result["samples"])
