from __future__ import annotations

import math

from fastapi.testclient import TestClient
import pytest

from backend.app.contracts import TrainingConfigValidationError, TrainingSpec
from backend.app.main import app
from backend.app.routers import runs as runs_router
from backend.app.routers.deps import TrainingDataReference
from backend.app.runs.repository import RunRepository


def test_unknown_training_fields_are_ignored_with_an_explicit_warning() -> None:
    spec = TrainingSpec.from_legacy(
        {
            'model_type': 'pls_da',
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
            'future_typo': 123,
        }
    ).validated(has_external_test=False)

    assert 'future_typo' not in spec.to_legacy_dict()
    assert spec.warnings == ('已忽略未知训练参数：future_typo',)


@pytest.mark.parametrize(
    ('config', 'has_external_test', 'message'),
    [
        ({'model_type': 'not-a-model'}, False, '不支持的分类模型'),
        (
            {
                'model_type': 'pls_da',
                'split_train': 8,
                'split_valid': 1,
                'split_test': 2,
            },
            False,
            '相加为 10',
        ),
        ({'model_type': 'cnn1d', 'learning_rate': math.nan}, False, '有限数值'),
        ({'model_type': 'cnn1d', 'batch_size': 0}, False, '大于 0 的整数'),
        ({'model_type': 'cnn1d', 'epochs': 1.5}, False, '大于 0 的整数'),
        ({'model_type': 'cnn1d', 'dropout': 1}, False, '小于 1'),
        ({'model_type': 'svm', 'svm_gamma': 0}, False, '必须大于 0'),
        ({'model_type': 'xgboost', 'xgboost_subsample': 1.1}, False, '不超过 1'),
    ],
)
def test_invalid_training_configuration_fails_before_queueing(
    config: dict,
    has_external_test: bool,
    message: str,
) -> None:
    with pytest.raises(TrainingConfigValidationError, match=message):
        TrainingSpec.from_legacy(config).validated(has_external_test=has_external_test)


def test_external_test_can_request_leave_one_sample_id_audit() -> None:
    spec = TrainingSpec.from_legacy(
        {
            'model_type': 'cnn1d',
            'split_mode': 'leave_one_sample_id_cv',
            'split_train': 8,
            'split_valid': 2,
            'split_test': 0,
        }
    ).validated(has_external_test=True)

    assert spec.to_legacy_dict()['split_mode'] == 'leave_one_sample_id_cv_with_external_test'


def test_transformer_compatibility_alias_is_normalized_to_catalog_id() -> None:
    spec = TrainingSpec.from_legacy(
        {
            'model_type': 'transformer1d',
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        }
    ).validated(has_external_test=False)

    assert spec.to_legacy_dict()['model_type'] == 'cnn_transformer1d'


def test_current_frontend_payload_is_accepted_and_normalized() -> None:
    payload = {
        'epochs': 200,
        'batch_size': 8,
        'learning_rate': 0.001,
        'normalization': 'zscore',
        'split_mode': 'stratified_holdout',
        'split_train': 8,
        'split_valid': 1,
        'split_test': 1,
        'class_balance': 'none',
        'model_type': 'cnn1d',
        'early_stopping_patience': 20,
        'dropout': None,
        'hidden_size': 64,
        'transformer_heads': 4,
        'dscarnet_input_mode': 'dual',
        'random_forest_n_estimators': 200,
        'random_forest_search_iterations': 10,
        'svm_c': 1,
        'svm_gamma': 'scale',
        'xgboost_n_estimators': 200,
        'xgboost_max_depth': 6,
        'xgboost_learning_rate': 0.05,
        'xgboost_subsample': 0.8,
        'xgboost_colsample_bytree': 0.8,
        'xgboost_reg_lambda': 1,
        'seed': 42,
        'feature_window_count': 100,
    }

    spec = TrainingSpec.from_legacy(payload).validated(has_external_test=False)

    assert spec.warnings == ()
    assert spec.to_legacy_dict() == payload


def test_external_test_payload_is_normalized_to_eight_two_zero() -> None:
    spec = TrainingSpec.from_legacy(
        {
            'model_type': 'logistic_regression',
            'split_mode': 'external_test_holdout',
            'split_train': 8,
            'split_valid': 2,
            'split_test': 0,
        }
    ).validated(has_external_test=True)

    values = spec.to_legacy_dict()
    assert values['split_mode'] == 'external_test_holdout'
    assert (values['split_train'], values['split_valid'], values['split_test']) == (8, 2, 0)


def _patch_create_run_dependencies(monkeypatch, tmp_path) -> RunRepository:
    repository = RunRepository(tmp_path / 'runs.sqlite3')
    repository.initialize()
    reference = TrainingDataReference(
        dataset_id='dataset-1',
        legacy_path=None,
        dataset_name='teacher-data.csv',
        test_dataset_id=None,
        test_legacy_path=None,
        test_dataset_name=None,
    )
    monkeypatch.setattr(runs_router, 'resolve_training_data_reference', lambda payload, principal: reference)
    monkeypatch.setattr(
        runs_router,
        '_dataset_snapshot_for_reference',
        lambda **kwargs: {
            'dataset_id': 'dataset-1',
            'name': 'teacher-data.csv',
            'sha256': 'a' * 64,
        },
    )
    monkeypatch.setattr(runs_router, 'get_run_repository', lambda: repository)
    monkeypatch.setattr(runs_router, 'get_run_dir', lambda run_id: tmp_path / 'runs' / run_id)
    return repository


def test_http_invalid_config_returns_422_without_creating_a_run(tmp_path, monkeypatch) -> None:
    repository = _patch_create_run_dependencies(monkeypatch, tmp_path)

    response = TestClient(app).post(
        '/api/training/runs',
        json={
            'dataset_id': 'dataset-1',
            'config': {
                'model_type': 'not-a-model',
                'split_train': 8,
                'split_valid': 1,
                'split_test': 1,
            },
        },
    )

    assert response.status_code == 422
    assert response.json()['detail']['code'] == 'invalid_training_config'
    assert '不支持的分类模型' in response.json()['detail']['message']
    assert repository.list() == []
    assert not (tmp_path / 'runs').exists()


def test_http_valid_config_returns_202_warning_and_queues_normalized_config(tmp_path, monkeypatch) -> None:
    repository = _patch_create_run_dependencies(monkeypatch, tmp_path)

    response = TestClient(app).post(
        '/api/training/runs',
        json={
            'dataset_id': 'dataset-1',
            'config': {
                'model_type': 'transformer1d',
                'split_mode': 'stratified_holdout',
                'split_train': 8,
                'split_valid': 1,
                'split_test': 1,
                'epochs': 10,
                'future_typo': True,
            },
        },
    )

    assert response.status_code == 202
    assert response.json()['state'] == 'queued'
    assert response.json()['warnings'] == ['已忽略未知训练参数：future_typo']
    records = repository.list()
    assert len(records) == 1
    assert records[0].config['model_type'] == 'cnn_transformer1d'
    assert 'future_typo' not in records[0].config
