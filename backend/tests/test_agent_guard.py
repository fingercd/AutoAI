from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.agent.guard import (
    AgentGuardRejected,
    run_postflight_guard,
    run_preflight_guard,
)
from backend.app.agent.service import _guard_feedback
from backend.app.contracts import TrainingConfigValidationError, TrainingSpec
from backend.app.main import app
from backend.app.paths import RUNS_DATABASE
from backend.app.runs import execution as execution_module
from backend.app.runs.contracts import RunRecord
from backend.app.runs.execution import TrainingExecution
from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import _error_details
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def _config(model_type='logistic_regression'):
    return {
        'model_type': model_type,
        'seed': 42,
        'split_train': 8,
        'split_valid': 1,
        'split_test': 1,
        'hpo_profile': 'tiny',
        'hpo_selection_metric': 'macro_f1',
        'agent_execution': {
            'guard_version': 'agent-guard-v1',
            'fail_fast_guard': True,
            'expected_model_type': model_type,
            'hpo_profile': 'tiny',
            'max_hpo_candidates': 3,
        },
    }


def test_preflight_passes_valid_data_and_rejects_bad_contract(tmp_path):
    source = tmp_path / 'valid.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    passed = run_preflight_guard(source, _config())
    assert passed['status'] == 'passed'
    assert passed['model_fit_count'] == 0

    bad = pd.read_csv(source)
    bad.iloc[0, 4] = float('nan')
    invalid = tmp_path / 'invalid.csv'
    bad.to_csv(invalid, index=False)
    with pytest.raises(AgentGuardRejected) as caught:
        run_preflight_guard(invalid, _config())
    result = caught.value.result
    assert result['status'] == 'rejected'
    assert result['model_fit_count'] == 0
    flat = json.dumps(result).lower()
    assert '/users/' not in flat
    assert str(invalid).lower() not in flat


def test_postflight_requires_model_metrics_and_cost_contract():
    result = {
        'status': 'success',
        'model_type': 'logistic_regression',
        'model_family': 'traditional_ml',
        'fold_count': 1,
        'metrics': {'valid': {'macro_f1': 0.8}},
        'model_metadata': {'model_fit_count': 4},
    }
    passed = run_postflight_guard(result, _config())
    assert passed['status'] == 'passed'
    assert passed['model_fit_count'] == 4
    with pytest.raises(AgentGuardRejected) as caught:
        run_postflight_guard({**result, 'metrics': None}, _config())
    assert caught.value.result['model_fit_count'] == 4
    for invalid in (
        {'test': {'macro_f1': 0.8}},
        {'valid': {'macro_f1': float('nan')}},
        {'valid': {'macro_f1': 1.1}},
    ):
        with pytest.raises(AgentGuardRejected) as invalid_result:
            run_postflight_guard({**result, 'metrics': invalid}, _config())
        assert any(
            item['code'] == 'validation_metric_contract'
            for item in invalid_result.value.result['checks']
        )
    over_budget = {
        **result,
        'model_metadata': {'model_fit_count': 5},
    }
    with pytest.raises(AgentGuardRejected) as budget_result:
        run_postflight_guard(over_budget, _config())
    assert any(
        item['code'] == 'model_fit_budget'
        for item in budget_result.value.result['checks']
    )


def test_execution_rejects_preflight_before_training_or_artifacts(monkeypatch, tmp_path):
    called = False

    class Repository:
        def assert_active(self, *args, **kwargs):
            return None

    class Execution(TrainingExecution):
        def _run_legacy_training(self, *args, **kwargs):
            nonlocal called
            called = True
            return {}

    rejected = {
        'schema_version': 'agent-guard-v1',
        'stage': 'preflight',
        'status': 'rejected',
        'retryable': False,
        'checks': [{'code': 'data_contract', 'status': 'failed'}],
        'model_fit_count': 0,
    }
    monkeypatch.setattr(
        execution_module,
        'run_preflight_guard',
        lambda *args, **kwargs: (_ for _ in ()).throw(AgentGuardRejected(rejected)),
    )
    record = RunRecord(
        run_id='guard-fail', state='running', version=1,
        dataset_id='dataset', legacy_data_path=None, config=_config(), progress={},
        claim_token='claim', worker_id='worker', lease_expires_at=None,
    )
    run_dir = tmp_path / 'guard-fail'
    with pytest.raises(AgentGuardRejected):
        Execution(repository=Repository(), run_dir=run_dir).execute(
            record, data_path=tmp_path / 'missing.csv'
        )
    assert called is False
    assert not run_dir.exists()


def test_worker_error_details_preserve_safe_guard_result():
    result = {
        'schema_version': 'agent-guard-v1',
        'stage': 'preflight',
        'status': 'rejected',
        'retryable': False,
        'checks': [{'code': 'split_feasibility', 'status': 'failed'}],
        'model_fit_count': 0,
    }
    details = _error_details(AgentGuardRejected(result))
    assert details['code'] == 'agent_guard_rejected'
    assert details['guard_result'] == result
    assert details['message'] == 'Agent Fail-Fast 检查未通过'


def test_failed_guard_feedback_merges_failure_over_passed_preflight():
    record = SimpleNamespace(
        progress={'agent_guard': {'preflight': {'status': 'passed'}}},
        error_details={
            'guard_result': {
                'stage': 'postflight',
                'status': 'rejected',
            }
        },
    )
    projected = _guard_feedback(record)
    assert projected['preflight']['status'] == 'passed'
    assert projected['failure']['stage'] == 'postflight'
    assert projected['failure']['status'] == 'rejected'


def test_execution_persists_preflight_and_postflight_progress(monkeypatch, tmp_path):
    class Repository:
        def __init__(self, record):
            self.record = record

        def assert_active(self, *args, **kwargs):
            return None

        def get(self, _run_id):
            return self.record

        def update_progress(self, _run_id, *, progress, **kwargs):
            self.record = replace(self.record, progress=progress)
            return self.record

    class Execution(TrainingExecution):
        def _run_legacy_training(self, record, **kwargs):
            return {
                'run_id': record.run_id,
                'status': 'success',
                'model_type': 'logistic_regression',
                'model_family': 'traditional_ml',
                'fold_count': 1,
                'metrics': {'valid': {'macro_f1': 0.8}},
                'model_metadata': {'model_fit_count': 4},
            }

    preflight = {
        'schema_version': 'agent-guard-v1', 'stage': 'preflight',
        'status': 'passed', 'retryable': False, 'checks': [],
        'model_fit_count': 0,
    }
    postflight = {
        'schema_version': 'agent-guard-v1', 'stage': 'postflight',
        'status': 'passed', 'retryable': False, 'checks': [],
        'model_fit_count': 4,
    }
    monkeypatch.setattr(execution_module, 'run_preflight_guard', lambda *args: preflight)
    monkeypatch.setattr(execution_module, 'run_postflight_guard', lambda *args: postflight)
    record = RunRecord(
        run_id='guard-pass', state='running', version=1,
        dataset_id='dataset', legacy_data_path=None, config=_config(), progress={},
        claim_token='claim', worker_id='worker', lease_expires_at=None,
    )
    repository = Repository(record)
    Execution(repository=repository, run_dir=tmp_path / 'guard-pass').execute(
        record, data_path=tmp_path / 'ignored.csv'
    )
    assert repository.record.progress['agent_guard'] == {
        'preflight': preflight,
        'postflight': postflight,
    }


def test_postflight_rejection_discards_unpublished_artifacts(monkeypatch, tmp_path):
    class Repository:
        def assert_active(self, *args, **kwargs):
            return None

        def update_progress(self, run_id, *, progress, **kwargs):
            return replace(record, progress=progress)

    class Execution(TrainingExecution):
        def _run_legacy_training(self, record, **kwargs):
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / 'model.pkl').write_bytes(b'private-model')
            (self.run_dir / 'metrics.json').write_text('{}', encoding='utf-8')
            return {
                'status': 'success',
                'model_type': 'logistic_regression',
                'model_family': 'traditional_ml',
                'fold_count': 1,
                'metrics': {},
                'model_metadata': {'model_fit_count': 4},
            }

    monkeypatch.setattr(
        execution_module,
        'run_preflight_guard',
        lambda *args: {
            'schema_version': 'agent-guard-v1', 'stage': 'preflight',
            'status': 'passed', 'retryable': False, 'checks': [],
            'model_fit_count': 0,
        },
    )
    record = RunRecord(
        run_id='guard-postflight-fail', state='running', version=1,
        dataset_id='dataset', legacy_data_path=None, config=_config(), progress={},
        claim_token='claim', worker_id='worker', lease_expires_at=None,
    )
    run_dir = tmp_path / 'guard-postflight-fail'
    with pytest.raises(AgentGuardRejected):
        Execution(repository=Repository(), run_dir=run_dir).execute(
            record, data_path=tmp_path / 'ignored.csv'
        )
    assert not run_dir.exists() or list(run_dir.iterdir()) == []


def test_agent_session_locks_guard_policy_and_queues_envelope(tmp_path):
    client = TestClient(app)
    source = tmp_path / 'guard.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    with source.open('rb') as handle:
        dataset_id = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        ).json()['dataset_id']
    body = {
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
        'modules': {'bounded_hpo': True, 'fail_fast_guard': True},
    }
    created = client.post('/api/agent/sessions', json=body)
    assert created.status_code == 201, created.text
    session = created.json()
    assert session['context']['guard_policy']['schema_version'] == 'agent-guard-v1'
    submitted = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={'model_type': 'logistic_regression', 'rationale': 'baseline'},
    )
    assert submitted.status_code == 202, submitted.text
    repository = RunRepository(RUNS_DATABASE)
    repository.initialize()
    record = repository.get(submitted.json()['run_id'])
    assert record.config['agent_execution']['fail_fast_guard'] is True
    assert record.config['agent_execution']['max_hpo_candidates'] == 3

    invalid = dict(body)
    invalid['modules'] = {'constrained_code_evolution': True}
    rejected = client.post('/api/agent/sessions', json=invalid)
    assert rejected.status_code == 422


def test_training_spec_rejects_tampered_guard_envelope():
    config = _config()
    config['hpo_profile'] = 'standard'
    with pytest.raises(TrainingConfigValidationError, match='hpo_profile'):
        TrainingSpec.from_legacy(config).validated(has_external_test=False)


def test_worker_error_redacts_assignment_style_absolute_path():
    details = _error_details(
        RuntimeError('failed path=/users/fotile/private/model.pkl')
    )
    assert '/users/' not in details['message']
    assert '详细路径信息' in details['message']
