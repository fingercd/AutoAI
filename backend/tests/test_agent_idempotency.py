import pytest

from backend.app.agent.contracts import AGENT_RESERVATION_PROTOCOL_VERSION
from backend.app.agent.repository import AgentIdempotencyConflict, AgentSessionRepository
from backend.app.runs.contracts import Principal


def _repo(tmp_path):
    repo = AgentSessionRepository(tmp_path / 'agent.sqlite3'); repo.initialize(); return repo


def _session(repo, principal=Principal(), request_id='session-1', payload_hash='same'):
    return repo.create_session(
        dataset_id='ds-1', selection_metric='macro_f1', allowed_models=['logistic_regression'],
        max_runs=1, seed=42, evaluation_config={'split_mode': 'stratified_holdout'},
        modules=[], context_policy={'source_role': 'development', 'case_write': False},
        client_request_id=request_id, payload_hash=payload_hash, principal=principal,
    )


def test_session_idempotency_survives_repository_restart(tmp_path):
    first, created = _session(_repo(tmp_path))
    second, replay_created = _session(_repo(tmp_path))
    assert created is True and replay_created is False
    assert first.session_id == second.session_id


def test_same_request_id_with_different_payload_conflicts(tmp_path):
    repo = _repo(tmp_path); _session(repo)
    try:
        _session(repo, payload_hash='different')
    except AgentIdempotencyConflict as exc:
        assert exc.code == 'agent_idempotency_conflict'
    else:
        raise AssertionError('expected idempotency conflict')


def test_principal_scopes_idempotency_keys(tmp_path):
    repo = _repo(tmp_path)
    first, _ = _session(repo, Principal('a', 't'))
    second, _ = _session(repo, Principal('b', 't'))
    assert first.session_id != second.session_id


def test_experiment_request_id_returns_same_reservation_and_conflicts_on_change(tmp_path):
    repo = _repo(tmp_path)
    session, _ = _session(repo, request_id=None)
    kwargs = dict(
        session_id=session.session_id, action_json={'model_type': 'logistic_regression'},
        rationale=None, parent_run_id=None, config_hash='config-a', client_request_id='experiment-1',
        payload_hash='payload-a', active_run_ids=set(), principal=Principal(),
    )
    first, created = repo.reserve_experiment(**kwargs)
    second, replay_created = repo.reserve_experiment(**kwargs)
    assert created is True and replay_created is False
    assert first.experiment_id == second.experiment_id
    try:
        repo.reserve_experiment(**{**kwargs, 'payload_hash': 'payload-b'})
    except AgentIdempotencyConflict:
        pass
    else:
        raise AssertionError('expected idempotency conflict')


def test_released_reservation_is_immutable_for_same_request_replay(tmp_path):
    repo = _repo(tmp_path)
    session, _ = _session(repo, request_id=None)
    kwargs = dict(
        session_id=session.session_id,
        action_json={'model_type': 'logistic_regression'},
        rationale=None,
        parent_run_id=None,
        config_hash='config-a',
        client_request_id='experiment-released',
        payload_hash='payload-a',
        active_run_ids=set(),
        principal=Principal(),
        protocol_version=AGENT_RESERVATION_PROTOCOL_VERSION,
    )
    original, created = repo.reserve_experiment(**kwargs)
    repo.release_reservation(
        original.experiment_id, failure_code='submission_failed', principal=Principal()
    )
    released_before = repo.get_reservation_scoped(
        original.experiment_id, principal=Principal()
    )

    replayed, replay_created = repo.reserve_experiment(**kwargs)
    released_after = repo.get_reservation_scoped(
        original.experiment_id, principal=Principal()
    )

    assert created is True
    assert replay_created is False
    assert replayed.experiment_id == original.experiment_id
    assert replayed.state == 'released'
    assert released_after == released_before
    assert len(repo.list_experiments_scoped(session.session_id, principal=Principal())) == 1
    with pytest.raises(AgentIdempotencyConflict):
        repo.reserve_experiment(**{**kwargs, 'payload_hash': 'different-payload'})


def test_new_request_or_no_request_id_never_reuses_released_reservation(tmp_path):
    repo = _repo(tmp_path)
    session, _ = _session(repo, request_id=None)
    common = dict(
        session_id=session.session_id,
        action_json={'model_type': 'logistic_regression'},
        rationale=None,
        parent_run_id=None,
        config_hash='same-config',
        payload_hash='same-payload',
        active_run_ids=set(),
        principal=Principal(),
        protocol_version=AGENT_RESERVATION_PROTOCOL_VERSION,
    )
    first, _ = repo.reserve_experiment(**common, client_request_id='request-1')
    repo.release_reservation(
        first.experiment_id, failure_code='first-failure', principal=Principal()
    )
    second, _ = repo.reserve_experiment(**common, client_request_id='request-2')
    repo.release_reservation(
        second.experiment_id, failure_code='second-failure', principal=Principal()
    )
    third, _ = repo.reserve_experiment(**common, client_request_id=None)

    assert len({first.experiment_id, second.experiment_id, third.experiment_id}) == 3
    assert (first.attempt, second.attempt, third.attempt) == (1, 2, 3)
    history = repo.list_experiments_scoped(session.session_id, principal=Principal())
    assert [item.state for item in history] == ['released', 'released', 'reserved']
    assert [item.failure_code for item in history[:2]] == ['first-failure', 'second-failure']
