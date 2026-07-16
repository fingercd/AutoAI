import pytest
import numpy as np
from torch import nn


@pytest.mark.parametrize(
    "model_type",
    ["pls_da", "pca_lda", "logistic_regression", "svm", "random_forest", "xgboost"],
)
def test_traditional_models_use_window_occlusion_log_loss(model_type):
    from backend.app.training_explainability import explainability_method

    assert explainability_method(model_type) == "window_occlusion_log_loss"


@pytest.mark.parametrize("model_type", ["pca_mlp", "transformer1d", "cnn_transformer1d", "cnn_mamba1d"])
def test_non_convolution_models_use_window_occlusion_log_loss(model_type):
    from backend.app.training_explainability import explainability_method

    assert explainability_method(model_type) == "window_occlusion_log_loss"


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


def test_pca_mlp_log_loss_occlusion_keeps_original_feature_axis():
    from backend.app.models.registry import build_deep_model
    from backend.app.training import TrainConfig, _deep_sample_feature_result

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
    config = TrainConfig(model_type="pca_mlp", feature_window_count=3, feature_top_k=2)
    model = build_deep_model(
        config,
        input_length=6,
        class_count=2,
        sample_count=2,
        x_train=x[:2],
    )
    result = _deep_sample_feature_result(
        config=config,
        model=model,
        x=x,
        y=y,
        x_axis=np.arange(6, dtype=np.float32),
        splits={"train": [0, 1], "valid": [], "test": [2, 3]},
        x_axis_warning={"status": "consistent"},
        label_names=["A", "B"],
        metadata=[{"index": idx, "name": str(idx), "sample_id": str(idx)} for idx in range(4)],
    )

    assert result["method"] == "sample_occlusion_log_loss"
    assert result["importance_metric"] == "masked_true_class_log_loss_minus_original_true_class_log_loss"
    assert all(len(sample["windows"]) == 3 for sample in result["samples"])


def test_document_cnn_uses_explicit_gradcam_target_and_original_axis():
    import torch

    from backend.app.feature_selection import sample_deep_attribution_importance
    from backend.app.models.cnn1d import CNN1DDocumentV2

    model = CNN1DDocumentV2(input_length=32, class_count=2, sample_count=16)
    x = np.random.default_rng(42).normal(size=(4, 32)).astype(np.float32)
    y = np.asarray([0, 1, 0, 1], dtype=np.int64)
    result = sample_deep_attribution_importance(
        model,
        x,
        y,
        x_axis=np.arange(32, dtype=np.float32),
        splits={"train": [0, 1], "valid": [], "test": [2, 3]},
        label_names=["A", "B"],
        model_type="cnn1d",
        top_k=2,
    )

    assert result["status"] == "ready"
    assert result["method"] == "gradcam_1d"
    assert all(len(sample["windows"]) == 32 for sample in result["samples"])
    assert all(sample["sanity_checks"] for sample in result["samples"])
    assert torch.isfinite(torch.as_tensor(result["samples"][0]["curve"])).all()


def test_document_cnn_gradcam_failure_is_explicit():
    from backend.app.feature_selection import sample_deep_attribution_importance

    class BrokenGradCAM(nn.Module):
        def __init__(self):
            super().__init__()
            self.target = nn.Linear(32, 2)

        def forward(self, x):
            return self.target(x.squeeze(1))

        def gradcam_target_layer(self):
            return self.target

    x = np.zeros((3, 32), dtype=np.float32)
    y = np.asarray([0, 1, 0], dtype=np.int64)
    result = sample_deep_attribution_importance(
        BrokenGradCAM(),
        x,
        y,
        x_axis=np.arange(32, dtype=np.float32),
        splits={"train": [0], "valid": [], "test": [1, 2]},
        label_names=["A", "B"],
        model_type="cnn1d",
    )

    assert result["status"] == "failed"
    assert result["method"] == "gradcam_1d"
    assert "Grad-CAM" in result["reason"]
