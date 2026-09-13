"""Session fingerprints and safe effective configs survive replay and old databases."""

import hashlib
import json
import sqlite3

import pytest

from backend.app.agent.contracts import (
    AGENT_METADATA_VERSION, AgentDomainError, CreateAgentExperimentRequest, CreateAgentSessionRequest,
)
from backend.app.agent.metadata import run_metadata
from backend.app.agent.repository import AgentSessionRepository
from backend.app.runs.contracts import Principal, RunRecord
from backend.tests.test_agent_concurrency import _service


METADATA_KEYS = (
    'metadata_version', 'dataset_sha256', 'dataset_fingerprint_status',
    'effective_config_status', 'effective_config',
)


def _new_session(service, dataset_id):
    request = CreateAgentSessionRequest(
        dataset_id=dataset_id, selection_metric='macro_f1',
        allowed_models=['logistic_regression'], max_runs=1, seed=137,
        client_request_id='new-session-operation',
    )
    return request, service.create_session(request, principal=Principal())


def _submit(service, session_id):
    request = CreateAgentExperimentRequest(
        model_type='logistic_regression', normalization='zscore', class_balance='none',
        rationale='Frozen reason', client_request_id='scientific-experiment-one',
    )
    return request, service.create_experiment(
        session_id=session_id, payload=request, principal=Principal(),
    )


def test_frozen_metadata_matches_run_and_all_public_projections_after_restart(tmp_path):
    service, sessions, runs, legacy = _service(tmp_path)
    session_request, created = _new_session(service, legacy.dataset_id)
    sid = created['session_id']
    digest = service.datasets.resolve(legacy.dataset_id, principal=Principal()).sha256
    assert created['locked_config']['dataset_sha256'] == digest
    assert created['locked_config']['dataset_fingerprint_status'] == 'ready'
    assert created['locked_config']['metadata_version'] == AGENT_METADATA_VERSION
    request, submitted = _submit(service, sid)
    record = runs.get(submitted['run_id'])
    assert record.state == 'queued'
    assert record.dataset_snapshot['sha256'] == digest == submitted['dataset_sha256']
    assert submitted['effective_config'] == {
        'model_type': 'logistic_regression', 'normalization': 'zscore', 'class_balance': 'none',
        'seed': 137, 'feature_selection_enabled': False,
        'evaluation_config': {
            'mode': 'stratified_holdout', 'train_weight': 8, 'validation_weight': 1,
            'heldout_weight': 1,
        },
    }
    service.sessions = AgentSessionRepository(sessions.database_path)
    service.sessions.initialize()
    replay = service.create_experiment(session_id=sid, payload=request, principal=Principal())
    inspected = service.get_session(session_id=sid, principal=Principal())
    observation = service.get_feedback(session_id=sid, run_id=record.run_id, principal=Principal())
    for response in (replay, inspected['experiments'][0], observation):
        assert {key: response[key] for key in METADATA_KEYS} == {
            key: submitted[key] for key in METADATA_KEYS
        }
    assert replay['idempotent_replay'] is True
    assert len(runs.list()) == 1
    assert inspected['locked_config'] == created['locked_config']
    assert 'test' not in json.dumps([submitted, replay, inspected, observation]).lower()
    assert service.create_session(session_request, principal=Principal())['idempotent_replay'] is True


def test_session_creation_replay_uses_original_fingerprint_even_if_file_is_removed(tmp_path):
    service, _sessions, _runs, legacy = _service(tmp_path)
    request, first = _new_session(service, legacy.dataset_id)
    dataset = service.datasets.resolve(legacy.dataset_id, principal=Principal())
    dataset.path.unlink()
    replay = service.create_session(request, principal=Principal())
    assert replay['session_id'] == first['session_id']
    assert replay['locked_config'] == first['locked_config']
    assert replay['idempotent_replay'] is True


def test_current_dataset_snapshot_cannot_silently_replace_frozen_session(tmp_path):
    service, sessions, runs, legacy = _service(tmp_path)
    _, created = _new_session(service, legacy.dataset_id)
    dataset = service.datasets.resolve(legacy.dataset_id, principal=Principal())
    # Simulate a changed trusted Dataset registration, not just a corrupt file.
    dataset.path.write_text('Index,Label,Sample_ID,0\n1,B,S2,3\n', encoding='utf-8')
    replacement_digest = hashlib.sha256(dataset.path.read_bytes()).hexdigest()
    with sqlite3.connect(service.datasets.database_path) as connection:
        connection.execute('UPDATE datasets SET sha256=? WHERE dataset_id=?',
                           (replacement_digest, legacy.dataset_id))
    with pytest.raises(AgentDomainError) as caught:
        _submit(service, created['session_id'])
    assert caught.value.code == 'agent_dataset_fingerprint_mismatch'
    assert runs.list() == []
    assert sessions.count_budget_scoped(session_id=created['session_id'], principal=Principal()) == 0
    assert service.get_session(session_id=created['session_id'], principal=Principal())[
        'locked_config']['dataset_sha256'] == dataset.sha256


@pytest.mark.parametrize('operation', ['replay', 'inspect', 'feedback', 'finalize'])
def test_mismatched_bound_run_is_rejected_from_every_metadata_boundary(tmp_path, operation):
    service, _sessions, runs, legacy = _service(tmp_path)
    _, created = _new_session(service, legacy.dataset_id)
    sid = created['session_id']
    request, submitted = _submit(service, sid)
    rid = submitted['run_id']
    with sqlite3.connect(runs.database_path) as connection:
        connection.execute('UPDATE runs SET dataset_snapshot_json=? WHERE run_id=?',
                           (json.dumps({'dataset_id': legacy.dataset_id, 'sha256': '0' * 64}), rid))
    operations = {
        'replay': lambda: service.create_experiment(session_id=sid, payload=request, principal=Principal()),
        'inspect': lambda: service.get_session(session_id=sid, principal=Principal()),
        'feedback': lambda: service.get_feedback(session_id=sid, run_id=rid, principal=Principal()),
        'finalize': lambda: service.finalize_session(session_id=sid, selected_run_id=rid, principal=Principal()),
    }
    with pytest.raises(AgentDomainError) as caught:
        operations[operation]()
    assert caught.value.code == 'agent_dataset_fingerprint_mismatch'
    assert len(runs.list()) == 1


def test_existing_session_database_migration_does_not_invent_fingerprints(tmp_path):
    service, sessions, _runs, legacy = _service(tmp_path)
    with sqlite3.connect(sessions.database_path) as connection:
        connection.execute('ALTER TABLE agent_sessions_v1 DROP COLUMN dataset_sha256')
        connection.execute('ALTER TABLE agent_sessions_v1 DROP COLUMN metadata_version')
    sessions.initialize()
    sessions.initialize()
    inspected = service.get_session(session_id=legacy.session_id, principal=Principal())
    assert inspected['locked_config']['dataset_sha256'] is None
    assert inspected['locked_config']['dataset_fingerprint_status'] == 'unavailable'
    assert sessions.get_session_scoped(legacy.session_id, principal=Principal()).metadata_version is None


def test_safe_config_projection_rejects_unsafe_values_and_never_exposes_extra_fields():
    raw = {
        'model_type': 'logistic_regression', 'normalization': 'zscore', 'class_balance': 'none',
        'seed': 42, 'feature_selection_enabled': False, 'split_mode': 'stratified_holdout',
        'split_train': 8, 'split_valid': 1, 'split_test': 1,
        'data_path': 'C:/sensitive/SYNTHETIC_SECRET', 'dataset_name': 'SYNTHETIC_SECRET',
        'test_macro_f1': 0.99, 'prediction': ['SYNTHETIC_SECRET'],
    }
    record = RunRecord(run_id='r1', state='queued', version=1, dataset_id='ds1',
                       legacy_data_path=None, config=raw)
    assert run_metadata(record)['effective_config_status'] == 'ready'
    flat = json.dumps(run_metadata(record)).lower()
    for forbidden in ('synthetic_secret', 'test', 'prediction', 'data_path'):
        assert forbidden not in flat
    raw['normalization'] = 'C:/sensitive/SYNTHETIC_SECRET'
    assert run_metadata(record)['effective_config_status'] == 'unavailable'
    assert run_metadata(record)['effective_config'] is None
    assert 'synthetic_secret' not in json.dumps(run_metadata(record)).lower()
