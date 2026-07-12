import json

import numpy as np
import pytest

from backend.tests.modeling_data_factory import write_grouped_classification_csv


def test_registry_contains_exactly_fifteen_docx_classifiers():
    from backend.app.models.registry import (
        DEEP_MODEL_TYPES,
        TARGET_DEEP_MODEL_TYPES,
        TARGET_TRADITIONAL_MODEL_TYPES,
        TRADITIONAL_MODEL_TYPES,
    )

    assert TARGET_TRADITIONAL_MODEL_TYPES == {
        "pls_da",
        "pca_lda",
        "logistic_regression",
        "svm",
        "random_forest",
        "xgboost",
    }
    assert TARGET_DEEP_MODEL_TYPES == {
        "pca_mlp",
        "cnn1d",
        "cnn1d_se",
        "resnet1d",
        "inception1d",
        "tcn1d",
        "cnn_transformer1d",
        "cnn_mamba1d",
        "dscarnet",
    }
    assert TRADITIONAL_MODEL_TYPES == {"pls_da", "svm", "random_forest", "xgboost"}
    assert DEEP_MODEL_TYPES == {"cnn1d", "transformer1d", "resnet1d", "inception1d", "tcn1d", "dscarnet"}


@pytest.mark.parametrize(
    ("sample_count", "expected_band", "expected_dropout"),
    [(100, "small", 0.5), (101, "medium", 0.4), (300, "medium", 0.4), (301, "large", 0.3)],
)
def test_model_profile_sample_boundaries(sample_count, expected_band, expected_dropout):
    from backend.app.models.profiles import build_model_profile

    profile = build_model_profile(
        "cnn1d",
        train_sample_count=sample_count,
        feature_count=1000,
    )
    assert profile.sample_band == expected_band
    assert profile.dropout == expected_dropout


@pytest.mark.parametrize(
    ("feature_count", "expected_band"),
    [(1000, "short"), (1001, "medium"), (3000, "medium"), (3001, "long")],
)
def test_model_profile_feature_boundaries(feature_count, expected_band):
    from backend.app.models.profiles import build_model_profile

    profile = build_model_profile(
        "cnn1d",
        train_sample_count=100,
        feature_count=feature_count,
    )
    assert profile.feature_band == expected_band


def test_model_range_warnings_are_explicit():
    from backend.app.models.profiles import model_range_warnings

    assert model_range_warnings(train_sample_count=49, feature_count=499) == [
        "训练样本数 N=49 超出文档适用范围 50-1000",
        "特征数 L=499 超出文档适用范围 500-10000",
    ]
    assert model_range_warnings(train_sample_count=50, feature_count=500) == []


def test_unknown_model_profile_is_rejected():
    from backend.app.models.profiles import build_model_profile

    with pytest.raises(ValueError, match="未知模型"):
        build_model_profile("unknown_model", train_sample_count=100, feature_count=1000)


def test_unfinished_model_builder_raises_exact_error():
    from backend.app.models.registry import ModelNotImplementedForVersion, build_deep_model

    class Config:
        model_type = "pca_mlp"

    with pytest.raises(ModelNotImplementedForVersion, match="pca_mlp"):
        build_deep_model(Config(), input_length=12, class_count=2, sample_count=10)


def test_training_writes_architecture_v2_metadata(tmp_path, monkeypatch):
    import backend.app.training as training

    source = write_grouped_classification_csv(
        tmp_path / "grouped.csv",
        groups_per_class=5,
        repeats=2,
        feature_count=12,
    )
    monkeypatch.setattr(training, "RUNS_DIR", tmp_path / "runs")
    result = training.train_model(
        source,
        {
            "model_type": "pls_da",
            "feature_selection_enabled": False,
        },
        run_id="metadata-test",
    )
    run_dir = tmp_path / "runs" / "metadata-test"
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))[0]
    metadata = json.loads((run_dir / "model_metadata.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

    assert result["status"] == "success"
    assert metadata["architecture_version"] == "docx-classification-v2"
    assert metadata["model_type"] == "pls_da"
    assert metadata["N_train"] == len(split["splits"]["train"])
    assert metadata["L"] == 12
    assert metadata["explainability_method"] == "window_permutation"
    assert config["architecture_version"] == metadata["architecture_version"]
    assert status["architecture_version"] == metadata["architecture_version"]
    assert status["model_metadata"] == metadata
