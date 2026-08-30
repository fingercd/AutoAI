from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend.app.agent.feedback import build_diagnosis
from backend.app.main import app
from backend.app.datasets.repository import DatasetRepository
from backend.app.paths import DATASETS_DATABASE, RUNS_DATABASE, STORAGE_DIR
from backend.app.runs.repository import RunRepository
from backend.app.training import train_model
from backend.tests.test_agent_sessions import _claim_and_finish
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def test_diagnosis_detects_generalization_gap_and_finite_actions():
    diagnosis = build_diagnosis(
        run_state='succeeded',
        selection_metric='macro_f1',
        training_metrics={'macro_f1': 0.95},
        validation_metrics={'macro_f1': 0.65},
        evidence_card={'statistics': {'risk_codes': ['high_dimension']}},
        restricted_actions_available=True,
    )
    assert diagnosis['status'] == 'ready'
    assert diagnosis['primary_category'] == 'training'
    assert diagnosis['generalization_gap'] == 0.3
    assert 'generalization_gap_high' in diagnosis['symptom_codes']
    assert set(diagnosis['allowed_action_ids']) <= {
        'switch_normalization', 'choose_unused_proposal', 'stop',
    }


def test_failure_diagnosis_uses_codes_not_raw_error_values():
    diagnosis = build_diagnosis(
        run_state='failed',
        selection_metric='macro_f1',
        training_metrics={},
        validation_metrics={},
        error_details={
            'code': 'agent_guard_rejected',
            'message': 'Bearer secret /users/private/model',
            'guard_result': {
                'checks': [
                    {'code': 'split_feasibility', 'status': 'failed'},
                ]
            },
        },
    )
    assert diagnosis['primary_category'] == 'split'
    flat = json.dumps(diagnosis).lower()
    for forbidden in ('bearer', '/users/', 'private/model'):
        assert forbidden not in flat


def test_feedback_module_policy_is_locked_in_session_context(tmp_path):
    client = TestClient(app)
    source = tmp_path / 'diagnosis.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    with source.open('rb') as handle:
        dataset_id = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        ).json()['dataset_id']
    base = {
        'dataset_id': dataset_id,
        'selection_metric': 'macro_f1',
        'allowed_models': ['logistic_regression'],
        'max_runs': 1,
        'seed': 42,
        'evaluation': {
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        },
    }
    missing_guard = client.post(
        '/api/agent/sessions',
        json={
            **base,
            'modules': {'feedback_diagnosis': True},
        },
    )
    assert missing_guard.status_code == 422
    response = client.post(
        '/api/agent/sessions',
        json={
            **base,
            'modules': {
                'fail_fast_guard': True,
                'feedback_diagnosis': True,
            },
        },
    )
    assert response.status_code == 201, response.text
    context = response.json()['context']
    assert context['status'] == 'ready'
    assert context['diagnosis_policy']['scope'] == 'train_validation_only'


def test_real_feedback_includes_train_validation_diagnosis(tmp_path):
    client = TestClient(app)
    source = tmp_path / 'diagnosis-real.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    with source.open('rb') as handle:
        dataset_id = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        ).json()['dataset_id']
    created = client.post(
        '/api/agent/sessions',
        json={
            'dataset_id': dataset_id,
            'selection_metric': 'macro_f1',
            'allowed_models': ['logistic_regression'],
            'max_runs': 1,
            'seed': 42,
            'evaluation': {
                'split_mode': 'stratified_holdout',
                'split_train': 8,
                'split_valid': 1,
                'split_test': 1,
            },
            'modules': {
                'fail_fast_guard': True,
                'feedback_diagnosis': True,
            },
        },
    ).json()
    experiment = client.post(
        f"/api/agent/sessions/{created['session_id']}/experiments",
        json={'model_type': 'logistic_regression', 'rationale': 'baseline'},
    ).json()
    runs = RunRepository(RUNS_DATABASE)
    runs.initialize()
    record = runs.get(experiment['run_id'])
    datasets = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    datasets.initialize()
    dataset = datasets.resolve_system(dataset_id, legacy_path=None)
    config = {**record.config, 'feature_selection_enabled': False}
    train_model(dataset.path, config, run_id=experiment['run_id'])
    _claim_and_finish(runs, experiment['run_id'])
    feedback = client.get(
        f"/api/agent/sessions/{created['session_id']}"
        f"/experiments/{experiment['run_id']}/feedback"
    )
    assert feedback.status_code == 200
    diagnosis = feedback.json()['diagnosis']
    assert diagnosis['status'] == 'ready'
    assert diagnosis['training_metrics']['macro_f1'] >= 0.0
    assert diagnosis['validation_metrics']['macro_f1'] >= 0.0
    assert isinstance(diagnosis['generalization_gap'], float)
    flat = feedback.text.lower()
    for forbidden in ('test_macro_f1', 'artifact', '/users/'):
        assert forbidden not in flat
