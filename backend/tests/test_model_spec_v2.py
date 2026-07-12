import json

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
    assert TRADITIONAL_MODEL_TYPES == {
        "pls_da",
        "pca_lda",
        "logistic_regression",
        "svm",
        "random_forest",
        "xgboost",
    }
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


def test_traditional_candidate_grids_are_exact():
    import backend.app.training as training

    config = training.TrainConfig()
    y_train = [0, 1] * 50

    pls = training._traditional_candidate_configs(config, "pls_da", 100, y_train)
    assert {candidate.pls_components for candidate in pls} == {1, 2, 3, 4, 5, 6, 8, 10, 12, 15}
    assert {candidate.pls_components for candidate in training._traditional_candidate_configs(config, "pls_da", 3, [0, 1, 0, 1])} == {1, 2, 3}

    pca = training._traditional_candidate_configs(config, "pca_lda", 100, y_train)
    assert {candidate.pca_components for candidate in pca} == {2, 3, 5, 8, 10, 15, 20, 30, 40, 50}
    assert {candidate.pca_components for candidate in training._traditional_candidate_configs(config, "pca_lda", 3, [0, 1, 0, 1])} == {2, 3}

    logistic = training._traditional_candidate_configs(config, "logistic_regression", 100, y_train)
    assert {candidate.logistic_c for candidate in logistic} == {0.1, 1.0, 10.0}

    svm = training._traditional_candidate_configs(config, "svm", 100, y_train)
    assert {candidate.svm_c for candidate in svm} == {0.01, 0.1, 1.0, 10.0, 100.0}
    assert {candidate.svm_kernel for candidate in svm} == {"linear"}

    random_forest = training._traditional_candidate_configs(config, "random_forest", 100, y_train)
    assert len(random_forest) == 18
    assert {candidate.random_forest_n_estimators for candidate in random_forest} == {500}
    assert {candidate.random_forest_max_depth for candidate in random_forest} == {3, 5, 8}
    assert {candidate.random_forest_min_samples_leaf for candidate in random_forest} == {1, 2}
    assert {candidate.random_forest_max_features for candidate in random_forest} == {"sqrt", "log2", 0.1}

    xgboost = training._traditional_candidate_configs(config, "xgboost", 100, y_train)
    assert len(xgboost) == 12
    assert {candidate.xgboost_n_estimators for candidate in xgboost} == {100, 300}
    assert {candidate.xgboost_max_depth for candidate in xgboost} == {2, 3, 5}
    assert {candidate.xgboost_min_child_weight for candidate in xgboost} == {3, 5}
    assert {candidate.xgboost_learning_rate for candidate in xgboost} == {0.1}
    assert {candidate.xgboost_colsample_bytree for candidate in xgboost} == {0.3}
    assert {candidate.xgboost_subsample for candidate in xgboost} == {0.8}
    assert {candidate.xgboost_reg_lambda for candidate in xgboost} == {10.0}


def test_new_traditional_builders_have_documented_types():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from backend.app.models.logistic_regression import build_logistic_regression
    from backend.app.models.pca_lda import build_pca_lda

    pca_lda = build_pca_lda(5)
    assert isinstance(pca_lda, Pipeline)
    assert isinstance(pca_lda.named_steps["lda"], LinearDiscriminantAnalysis)
    logistic = build_logistic_regression(10.0, 42, "balanced")
    assert isinstance(logistic, LogisticRegression)
    assert logistic.C == 10.0
    assert logistic.random_state == 42
    assert logistic.class_weight == "balanced"


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
