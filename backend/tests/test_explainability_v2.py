import pytest


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
