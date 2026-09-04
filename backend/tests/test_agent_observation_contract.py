import json

import pytest

from backend.app.agent.observation import build_observation
from backend.app.runs.artifacts import RunArtifactWriter
from backend.app.runs.contracts import RunRecord


def _ready_run(tmp_path):
    run_id = 'run-ready'; run_dir = tmp_path / run_id
    writer = RunArtifactWriter(run_dir)
    writer.write_json('metrics.json', {
        'valid': {'macro_f1': 0.75, 'balanced_accuracy': 0.8, 'nan': float('nan')},
        'test': {'macro_f1': 0.99}, 'predictions': ['secret'], 'artifact': 'secret',
    })
    for name, payload in {
        'cv_metrics.json': {}, 'model_metadata.json': {}, 'label_map.json': {},
        'split.json': {}, 'config.json': {'model_type': 'logistic_regression'},
    }.items(): writer.write_json(name, payload)
    writer.write_bytes('fold_metrics.csv', b'a,b\n1,2\n')
    writer.write_bytes('predictions.csv', b'a,b\n1,2\n')
    writer.finalize(run_id=run_id, metadata={'model_family': 'traditional_ml'})
    return RunRecord(
        run_id=run_id, state='succeeded', version=3, dataset_id='ds-1',
        legacy_data_path=None, config={},
    ), run_dir


def test_observation_only_exposes_manifest_verified_validation(tmp_path):
    record, run_dir = _ready_run(tmp_path)
    result = build_observation(
        session_id='s', session_state='open', selection_metric='macro_f1', run_dir=run_dir,
        record=record, attempt=1, effective_action={'model_type': 'logistic_regression'}, remaining_runs=1,
    )
    assert result['observation_version'] == 'agent-observation-v1'
    assert result['validation'] == {
        'status': 'ready', 'metrics': {'balanced_accuracy': 0.8, 'macro_f1': 0.75}
    }
    flat = json.dumps(result).lower()
    for forbidden in ('test', 'prediction', 'confusion', 'artifact', 'explainability'):
        assert forbidden not in flat


def test_tampered_metrics_never_becomes_ready(tmp_path):
    record, run_dir = _ready_run(tmp_path)
    (run_dir / 'metrics.json').write_text('{"valid":{"macro_f1":1}}', encoding='utf-8')
    result = build_observation(
        session_id='s', session_state='open', selection_metric='macro_f1', run_dir=run_dir,
        record=record, attempt=1, effective_action={}, remaining_runs=0,
    )
    assert result['validation']['status'] == 'unavailable'
    assert 'finalize_ml_session' not in result['allowed_actions']


@pytest.mark.parametrize('state', ['queued', 'running', 'failed', 'cancelled'])
def test_non_success_states_have_server_authoritative_actions(tmp_path, state):
    record = RunRecord(
        run_id=f'run-{state}', state=state, version=1, dataset_id='ds-1',
        legacy_data_path=None, config={}, error='safe failure' if state == 'failed' else None,
    )
    result = build_observation(
        session_id='s', session_state='open', selection_metric='macro_f1',
        run_dir=tmp_path / record.run_id, record=record, attempt=1,
        effective_action={}, remaining_runs=1,
    )
    if state in {'queued', 'running'}:
        assert result['allowed_actions'] == ['observe_ml_experiment', 'inspect_ml_session']
        assert result['retry_after_seconds'] == 2
    else:
        assert 'submit_ml_experiment' in result['allowed_actions']
        assert 'retry_after_seconds' not in result
