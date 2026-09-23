import numpy as np
import pytest
from backend.tests.test_agent_model_sessions import api

from backend.app.processing_policy import (
    EXECUTABLE_MODELS,
    freeze_fixed_processing,
    legal_processing,
    validate_processing,
)
from backend.app.training import _fit_x_normalizer, _transform_x_with_normalizer


def test_finite_support_matrix():
    assert len(EXECUTABLE_MODELS) == 13
    assert sum(len(legal_processing(model, class_count=2)) for model in EXECUTABLE_MODELS) == 96
    assert sum(len(legal_processing(model, class_count=3)) for model in EXECUTABLE_MODELS) == 92
    with pytest.raises(ValueError, match="multiclass_xgboost"):
        validate_processing("xgboost", "area", "class_weight", class_count=3)
    with pytest.raises(ValueError, match="unsupported_for_model"):
        validate_processing("pca_lda", "zscore", "class_weight", class_count=2)


def test_fixed_processing_requires_exact_mapping():
    assert freeze_fixed_processing(["svm"])["svm"] == {
        "normalization": "zscore", "class_balance": "none"
    }
    with pytest.raises(ValueError, match="model_set_mismatch"):
        freeze_fixed_processing(["svm", "cnn1d"], {"svm": {"normalization": "none", "class_balance": "none"}})


def test_area_uses_unit_feature_index_and_finite_float32():
    matrix = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    normalizer = _fit_x_normalizer(matrix, "area")
    assert normalizer == {"mode": "area"}
    actual = _transform_x_with_normalizer(matrix, normalizer)
    np.testing.assert_allclose(actual[:2], matrix[:2] / 4.0)
    np.testing.assert_array_equal(actual[2], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(
        _transform_x_with_normalizer([[3.0]], _fit_x_normalizer(np.array([[3.0]]), "area")),
        np.array([[3e8]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="area must be finite"):
        _transform_x_with_normalizer([[1e308, 1e308, 1e308]], normalizer)
    with pytest.raises(ValueError, match="float32 output must be finite"):
        _transform_x_with_normalizer([[1e100, 1.0]], {"mode": "none"})


def test_normalizer_statistics_use_train_only():
    train = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    valid = np.array([[20.0, 30.0]], dtype=np.float32)
    normalizer = _fit_x_normalizer(train, "minmax")
    np.testing.assert_allclose(_transform_x_with_normalizer(valid, normalizer), [[10.0, 14.5]])
    with pytest.raises(ValueError, match="invalid_normalization"):
        _fit_x_normalizer(train, "bad")


def test_training_spec_rejects_unsupported_weight_before_queuing():
    from backend.app.contracts import TrainingConfigValidationError, TrainingSpec

    with pytest.raises(TrainingConfigValidationError, match="class_weight_unsupported_for_model"):
        TrainingSpec.from_legacy({
            "model_type": "pca_lda", "class_balance": "class_weight",
        }).validated(has_external_test=False)
    with pytest.raises(TrainingConfigValidationError, match="invalid_normalization"):
        TrainingSpec.from_legacy({
            "model_type": "svm", "normalization": "not_a_mode",
        }).validated(has_external_test=False)


def test_old_source_bindings_only_match_verified_default_processing():
    from backend.app.model_config import (
        _DEFAULT_PROCESSING_BINDING_COMPAT, compatible_search_strategy_binding,
        search_strategy_binding,
    )

    assert len(_DEFAULT_PROCESSING_BINDING_COMPAT) == len(EXECUTABLE_MODELS)
    for model_id, (old_digest, new_digest) in _DEFAULT_PROCESSING_BINDING_COMPAT.items():
        assert search_strategy_binding(model_id)["digest"] == new_digest
        assert compatible_search_strategy_binding(model_id, old_digest, "zscore", "none")
        assert not compatible_search_strategy_binding(model_id, old_digest, "area", "none")
        assert not compatible_search_strategy_binding(model_id, old_digest, "zscore", "class_weight")
        assert not compatible_search_strategy_binding(model_id, "0" * 64, "zscore", "none")


def test_multiclass_xgboost_weight_is_rejected_before_run_creation(api, tmp_path):
    from backend.tests.modeling_data_factory import write_grouped_classification_csv
    from backend.app.runs.repository import RunRepository

    client, storage, _ = api
    source = write_grouped_classification_csv(tmp_path / 'three-classes.csv',
        groups_per_class=4,repeats=2,feature_count=16,labels=('A','B','C'))
    with source.open('rb') as handle:
        uploaded = client.post('/api/datasets/upload', files={
            'file': ('three-classes.csv',handle,'text/csv')})
    assert uploaded.status_code == 200
    response = client.post('/api/training/runs',json={
        'dataset_id': uploaded.json()['dataset_id'],
        'config': {'model_type':'xgboost','class_balance':'class_weight'},
    })
    assert response.status_code == 422
    assert RunRepository(storage / 'runs.sqlite3').list() == []
