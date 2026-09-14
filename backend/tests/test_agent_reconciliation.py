from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app.agent.contracts import (
    AGENT_RESERVATION_PROTOCOL_VERSION,
    AgentDomainError,
    CreateAgentExperimentRequest,
)
from backend.app.agent.reconciliation import AgentReconciliationService
from backend.app.agent.repository import AgentSessionRepository
from backend.app.agent.service import AgentService
from backend.app.datasets.repository import DatasetRepository
from backend.app.main import app
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import InvalidRunTransition, RunRepository
from backend.app.runs.status_projection import project_status
from backend.app.runs.submission import (
    RunSubmissionError,
    RunSubmissionRequest,
    RunSubmissionService,
)


def _environment(tmp_path, *, principal=Principal()):
    storage = tmp_path / 'storage'
    uploads = storage / 'uploads'
    uploads.mkdir(parents=True)
    source = uploads / 'data.csv'
    source.write_text(
        'Index,Label,Sample_ID,0,1\n1,A,A1,1,2\n2,A,A2,2,3\n3,A,A3,3,4\n4,B,B1,2,1\n5,B,B2,3,2\n6,B,B3,4,3\n',
        encoding='utf-8',
    )
    datasets = DatasetRepository(storage / 'datasets.sqlite3', storage_root=storage)
    datasets.initialize()
    dataset = datasets.register(source, original_name='teacher.csv', principal=principal)
    sessions = AgentSessionRepository(storage / 'agent.sqlite3')
    sessions.initialize()
    runs = RunRepository(storage / 'runs.sqlite3')
    runs.initialize()
    session, _ = sessions.create_session(
        dataset_id=dataset.dataset_id,
        selection_metric='macro_f1',
        allowed_models=['logistic_regression'],
        max_runs=5,
        seed=42,
        evaluation_config={
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        },
        modules=[],
        context_policy={},
        client_request_id=None,
        payload_hash='session',
        principal=principal,
    )
    submissions = RunSubmissionService(
        run_repository=runs,
        dataset_repository=datasets,
        run_dir=lambda run_id: storage / 'runs' / run_id,
        status_projector=project_status,
    )
    reconciler = AgentReconciliationService(
        session_repository=sessions,
        run_repository=runs,
    )
    return storage, sessions, runs, submissions, reconciler, session


def _reserve(sessions, session, *, principal=Principal(), protocol=True, suffix='1'):
    reservation, _ = sessions.reserve_experiment(
        session_id=session.session_id,
        action_json={
            'model_type': 'logistic_regression',
            'normalization': 'zscore',
            'class_balance': 'none',
            'parent_run_id': None,
        },
        rationale=None,
        parent_run_id=None,
        config_hash=f'config-{suffix}',
        client_request_id=f'request-{suffix}',
        payload_hash=f'payload-{suffix}',
        active_run_ids=set(),
        principal=principal,
        protocol_version=(AGENT_RESERVATION_PROTOCOL_VERSION if protocol else None),
    )
    return reservation


def _submit(submissions, session, reservation, *, principal=Principal()):
    return submissions.submit(RunSubmissionRequest(
        dataset_id=session.dataset_id,
        legacy_data_path=None,
        test_dataset_id=None,
        test_legacy_data_path=None,
        raw_config={
            'model_type': 'logistic_regression',
            'normalization': 'zscore',
            'class_balance': 'none',
            'seed': 42,
            'feature_selection_enabled': False,
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        },
        principal=principal,
        submission_source='agent',
        submission_key=reservation.experiment_id,
    ))


def _stale_now(reservation):
    return datetime.fromisoformat(reservation.updated_at) + timedelta(seconds=301)


def test_agent_reservation_audit_migration_is_idempotent_and_preserves_rows(tmp_path):
    database = tmp_path / 'agent.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.execute(
            '''CREATE TABLE agent_experiment_reservations_v1 (
               reservation_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,state TEXT NOT NULL,
               run_id TEXT,attempt INTEGER NOT NULL,parent_run_id TEXT,action_json TEXT NOT NULL,
               rationale TEXT,config_hash TEXT NOT NULL,client_request_id TEXT,payload_hash TEXT,
               failure_code TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
               owner_id TEXT,tenant_id TEXT,UNIQUE(run_id))'''
        )
        connection.execute(
            '''INSERT INTO agent_experiment_reservations_v1(
               reservation_id,session_id,state,attempt,action_json,config_hash,created_at,updated_at
               ) VALUES('legacy-reservation','legacy-session','reserved',1,'{}','hash','now','now')'''
        )

    repository = AgentSessionRepository(database)
    repository.initialize()
    repository.initialize()

    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute(
                'PRAGMA table_info(agent_experiment_reservations_v1)'
            )
        }
        row = connection.execute(
            '''SELECT reservation_id,protocol_version,last_reconciled_at,
               reconcile_attempt_count,resolution_code
               FROM agent_experiment_reservations_v1'''
        ).fetchone()
    assert {
        'protocol_version', 'last_reconciled_at',
        'reconcile_attempt_count', 'resolution_code',
    } <= columns
    assert row == ('legacy-reservation', None, None, 0, None)


def test_fresh_reserved_is_unchanged(tmp_path):
    _, sessions, _, _, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)

    result = reconciler.reconcile_session(
        session_id=session.session_id,
        principal=Principal(),
        now=datetime.fromisoformat(reservation.updated_at),
    )

    current = sessions.get_reservation_scoped(reservation.experiment_id, principal=Principal())
    assert current.state == 'reserved'
    assert current.reconcile_attempt_count == 0
    assert result['status'] == 'unchanged'
    assert result['items'][0]['resolution_code'] == 'reservation_fresh'


def test_stale_durable_reservation_without_mapping_is_released_after_restart(tmp_path):
    storage, sessions, _, _, _, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    restarted_sessions = AgentSessionRepository(storage / 'agent.sqlite3'); restarted_sessions.initialize()
    restarted_runs = RunRepository(storage / 'runs.sqlite3'); restarted_runs.initialize()
    restarted = AgentReconciliationService(
        session_repository=restarted_sessions, run_repository=restarted_runs
    )

    result = restarted.reconcile_session(
        session_id=session.session_id, principal=Principal(), now=_stale_now(reservation)
    )

    current = restarted_sessions.get_reservation_scoped(
        reservation.experiment_id, principal=Principal()
    )
    assert current.state == 'released'
    assert current.resolution_code == 'stale_without_run'
    assert current.reconcile_attempt_count == 1
    assert result['summary']['released'] == 1
    assert restarted_sessions.count_budget_scoped(
        session_id=session.session_id, principal=Principal()
    ) == 0


def test_run_created_before_bind_is_recovered_after_restart(tmp_path):
    storage, sessions, runs, submissions, _, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    submitted = _submit(submissions, session, reservation)
    assert len(runs.list()) == 1
    restarted = AgentReconciliationService(
        session_repository=AgentSessionRepository(storage / 'agent.sqlite3'),
        run_repository=RunRepository(storage / 'runs.sqlite3'),
    )
    restarted.sessions.initialize(); restarted.runs.initialize()

    result = restarted.reconcile_session(
        session_id=session.session_id, principal=Principal(), now=_stale_now(reservation)
    )

    current = restarted.sessions.get_reservation_scoped(
        reservation.experiment_id, principal=Principal()
    )
    assert current.state == 'bound'
    assert current.run_id == submitted.record.run_id
    assert current.resolution_code == 'recovered_binding'
    assert result['summary']['recovered_bound'] == 1
    assert len(restarted.runs.list()) == 1


def test_compensation_queued_unstarted_is_cancelled_and_released(tmp_path):
    _, sessions, runs, submissions, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    run = _submit(submissions, session, reservation).record
    sessions.require_compensation(
        reservation.experiment_id, run_id=run.run_id,
        failure_code='binding_failed', principal=Principal(),
    )

    result = reconciler.reconcile_session(
        session_id=session.session_id, principal=Principal()
    )

    current = sessions.get_reservation_scoped(reservation.experiment_id, principal=Principal())
    assert current.state == 'released'
    assert current.resolution_code == 'cancelled_before_start'
    assert current.reconcile_attempt_count == 1
    assert runs.get(run.run_id).state == 'cancelled'
    assert runs.get(run.run_id).started_at is None
    assert result['summary']['released'] == 1


def test_released_request_cannot_rebind_cancelled_run_and_new_request_gets_new_run(
    tmp_path, monkeypatch,
):
    storage, sessions, runs, submissions, reconciler, session = _environment(tmp_path)
    datasets = DatasetRepository(storage / 'datasets.sqlite3', storage_root=storage)
    datasets.initialize()
    service = AgentService(
        session_repository=sessions,
        run_repository=runs,
        dataset_repository=datasets,
        submission_service=submissions,
        run_root=storage / 'runs',
    )
    original_request = CreateAgentExperimentRequest(
        model_type='logistic_regression',
        normalization='zscore',
        class_balance='none',
        client_request_id='lifecycle-request-1',
    )
    with monkeypatch.context() as patching:
        patching.setattr(
            sessions, 'bind_experiment',
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('bind failed')),
        )
        patching.setattr(
            runs, 'cancel_queued_unstarted_scoped',
            lambda *args, **kwargs: (_ for _ in ()).throw(
                InvalidRunTransition('defer to reconciliation')
            ),
        )
        with pytest.raises(AgentDomainError) as initial_failure:
            service.create_experiment(
                session_id=session.session_id,
                payload=original_request,
                principal=Principal(),
            )
    assert initial_failure.value.code == 'agent_experiment_binding_failed'
    original_reservation = sessions.list_experiments_scoped(
        session.session_id, principal=Principal()
    )[0]
    original_run_id = original_reservation.run_id
    assert original_reservation.state == 'compensation_required'
    assert original_run_id is not None

    reconciler.reconcile_session(session_id=session.session_id, principal=Principal())
    released = sessions.get_reservation_scoped(
        original_reservation.experiment_id, principal=Principal()
    )
    assert released.state == 'released'
    assert released.run_id == original_run_id
    assert runs.get(original_run_id).state == 'cancelled'

    from backend.app.routers import agent as agent_router
    monkeypatch.setattr(agent_router, 'AGENT_DATABASE', storage / 'agent.sqlite3')
    monkeypatch.setattr(agent_router, 'RUNS_DATABASE', storage / 'runs.sqlite3')
    monkeypatch.setattr(agent_router, 'DATASETS_DATABASE', storage / 'datasets.sqlite3')
    monkeypatch.setattr(agent_router, 'STORAGE_DIR', storage)
    monkeypatch.setattr(agent_router, 'RUNS_DIR', storage / 'runs')
    replay_response = TestClient(app).post(
        f'/api/agent/sessions/{session.session_id}/experiments',
        json=original_request.model_dump(mode='json'),
    )
    assert replay_response.status_code == 409
    assert replay_response.json()['detail'] == {
        'code': 'agent_request_released',
        'message': '该请求已经结束；如需重新训练，请使用新的 client_request_id',
        'retryable': False,
        'allowed_actions': ['inspect_ml_session'],
    }
    after_replay = sessions.get_reservation_scoped(
        original_reservation.experiment_id, principal=Principal()
    )
    assert after_replay == released
    assert runs.get(original_run_id).state == 'cancelled'
    assert len(runs.list()) == 1

    new_result = service.create_experiment(
        session_id=session.session_id,
        payload=CreateAgentExperimentRequest(
            model_type='logistic_regression',
            normalization='zscore',
            class_balance='none',
            client_request_id='lifecycle-request-2',
        ),
        principal=Principal(),
    )
    history = sessions.list_experiments_scoped(
        session.session_id, principal=Principal()
    )
    new_reservation = history[1]
    assert new_reservation.experiment_id != original_reservation.experiment_id
    assert new_reservation.attempt == original_reservation.attempt + 1
    assert new_result['run_id'] != original_run_id
    assert new_result['state'] == 'queued'
    assert runs.get(original_run_id).state == 'cancelled'
    assert len(runs.list()) == 2
    old_mapping = runs.lookup_submission_mapping(
        original_reservation.experiment_id, principal=Principal()
    )
    new_mapping = runs.lookup_submission_mapping(
        new_reservation.experiment_id, principal=Principal()
    )
    assert old_mapping.run_id == original_run_id
    assert new_mapping.run_id == new_result['run_id']


def test_released_request_without_mapping_is_not_reexecuted(tmp_path):
    storage, sessions, runs, _submissions, _reconciler, session = _environment(tmp_path)
    datasets = DatasetRepository(storage / 'datasets.sqlite3', storage_root=storage)
    datasets.initialize()

    class AlwaysFails:
        def __init__(self):
            self.calls = 0

        def submit(self, _request):
            self.calls += 1
            raise RunSubmissionError(
                'worker_contract_mismatch', 'worker unavailable', status_code=503
            )

    failing = AlwaysFails()
    service = AgentService(
        session_repository=sessions,
        run_repository=runs,
        dataset_repository=datasets,
        submission_service=failing,
        run_root=storage / 'runs',
    )
    original = CreateAgentExperimentRequest(
        model_type='logistic_regression', client_request_id='failed-before-run-1'
    )
    with pytest.raises(AgentDomainError) as first_failure:
        service.create_experiment(
            session_id=session.session_id, payload=original, principal=Principal()
        )
    assert first_failure.value.code == 'worker_contract_mismatch'
    first = sessions.list_experiments_scoped(session.session_id, principal=Principal())[0]
    assert first.state == 'released'
    assert first.run_id is None
    assert failing.calls == 1

    with pytest.raises(AgentDomainError) as replay_failure:
        service.create_experiment(
            session_id=session.session_id, payload=original, principal=Principal()
        )
    assert replay_failure.value.code == 'agent_request_released'
    assert failing.calls == 1
    assert sessions.get_reservation_scoped(
        first.experiment_id, principal=Principal()
    ) == first

    with pytest.raises(AgentDomainError) as second_failure:
        service.create_experiment(
            session_id=session.session_id,
            payload=CreateAgentExperimentRequest(
                model_type='logistic_regression', client_request_id='failed-before-run-2'
            ),
            principal=Principal(),
        )
    assert second_failure.value.code == 'worker_contract_mismatch'
    history = sessions.list_experiments_scoped(session.session_id, principal=Principal())
    assert failing.calls == 2
    assert len(history) == 2
    assert history[0].experiment_id != history[1].experiment_id
    assert [item.attempt for item in history] == [1, 2]
    assert all(item.state == 'released' for item in history)
    assert len(runs.list()) == 0


def test_agent_db_failure_before_cancel_does_not_change_run(tmp_path, monkeypatch):
    _, sessions, runs, submissions, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    run = _submit(submissions, session, reservation).record
    sessions.require_compensation(
        reservation.experiment_id, run_id=run.run_id,
        failure_code='binding_failed', principal=Principal(),
    )
    monkeypatch.setattr(
        sessions, 'reconcile_transition',
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError('closed')),
    )

    with pytest.raises(AgentDomainError) as failure:
        reconciler.reconcile_session(session_id=session.session_id, principal=Principal())

    assert failure.value.code == 'agent_reconciliation_unavailable'
    assert failure.value.retryable is True
    assert runs.get(run.run_id).state == 'queued'


def test_runs_db_failure_keeps_reservation_and_returns_retryable_503(tmp_path, monkeypatch):
    _, sessions, runs, _, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    monkeypatch.setattr(
        runs, 'lookup_submission_mapping',
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError('locked')),
    )

    with pytest.raises(AgentDomainError) as failure:
        reconciler.reconcile_session(
            session_id=session.session_id,
            principal=Principal(),
            now=_stale_now(reservation),
        )

    current = sessions.get_reservation_scoped(reservation.experiment_id, principal=Principal())
    assert current.state == 'reserved'
    assert current.reconcile_attempt_count == 0
    assert failure.value.code == 'agent_reconciliation_unavailable'
    assert failure.value.retryable is True


def test_compensation_running_stays_occupied(tmp_path):
    _, sessions, runs, submissions, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    run = _submit(submissions, session, reservation).record
    sessions.require_compensation(
        reservation.experiment_id, run_id=run.run_id,
        failure_code='binding_failed', principal=Principal(),
    )
    claimed = runs.claim_next(worker_id='worker-a', now=datetime.now(timezone.utc))
    assert claimed and claimed.run_id == run.run_id

    result = reconciler.reconcile_session(
        session_id=session.session_id, principal=Principal()
    )

    current = sessions.get_reservation_scoped(reservation.experiment_id, principal=Principal())
    assert current.state == 'compensation_required'
    assert current.resolution_code == 'run_still_active'
    assert sessions.count_budget_scoped(session_id=session.session_id, principal=Principal()) == 1
    assert runs.get(run.run_id).state == 'running'
    assert result['summary']['released'] == 0


def test_uncertain_cancellation_keeps_compensation_required(tmp_path, monkeypatch):
    from backend.app.runs.repository import InvalidRunTransition

    _, sessions, runs, submissions, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    run = _submit(submissions, session, reservation).record
    sessions.require_compensation(
        reservation.experiment_id, run_id=run.run_id,
        failure_code='binding_failed', principal=Principal(),
    )
    monkeypatch.setattr(
        runs, 'cancel_queued_unstarted_scoped',
        lambda *args, **kwargs: (_ for _ in ()).throw(InvalidRunTransition('race')),
    )

    result = reconciler.reconcile_session(
        session_id=session.session_id, principal=Principal()
    )

    current = sessions.get_reservation_scoped(reservation.experiment_id, principal=Principal())
    assert current.state == 'compensation_required'
    assert current.resolution_code == 'cancellation_uncertain'
    assert runs.get(run.run_id).state == 'queued'
    assert result['status'] == 'manual_review_required'


@pytest.mark.parametrize('terminal_state', ['succeeded', 'failed'])
def test_compensation_terminal_run_is_bound_without_rewriting_terminal(tmp_path, terminal_state):
    _, sessions, runs, submissions, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    run = _submit(submissions, session, reservation).record
    sessions.require_compensation(
        reservation.experiment_id, run_id=run.run_id,
        failure_code='binding_failed', principal=Principal(),
    )
    claim = runs.claim_next(worker_id='worker-a', now=datetime.now(timezone.utc))
    assert claim is not None
    if terminal_state == 'succeeded':
        runs.finish_success(
            run.run_id, claim_token=claim.claim_token,
            now=datetime.now(timezone.utc),
        )
    else:
        runs.finish_failure(
            run.run_id, claim_token=claim.claim_token,
            now=datetime.now(timezone.utc), error='expected failure',
        )

    reconciler.reconcile_session(session_id=session.session_id, principal=Principal())

    current = sessions.get_reservation_scoped(reservation.experiment_id, principal=Principal())
    assert current.state == 'bound'
    assert current.resolution_code == 'recovered_terminal_binding'
    assert runs.get(run.run_id).state == terminal_state


def test_started_cancelled_run_is_bound_and_budget_is_not_released(tmp_path):
    _, sessions, runs, submissions, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    run = _submit(submissions, session, reservation).record
    sessions.require_compensation(
        reservation.experiment_id, run_id=run.run_id,
        failure_code='binding_failed', principal=Principal(),
    )
    claim = runs.claim_next(worker_id='worker-a', now=datetime.now(timezone.utc))
    assert claim is not None
    runs.cancel_scoped(run.run_id, now=datetime.now(timezone.utc), principal=Principal())

    reconciler.reconcile_session(session_id=session.session_id, principal=Principal())

    current = sessions.get_reservation_scoped(reservation.experiment_id, principal=Principal())
    assert current.state == 'bound'
    assert current.resolution_code == 'recovered_cancelled_binding'
    assert sessions.count_budget_scoped(session_id=session.session_id, principal=Principal()) == 1


def test_legacy_reservation_without_protocol_fails_closed(tmp_path):
    _, sessions, _, _, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session, protocol=False)

    result = reconciler.reconcile_session(
        session_id=session.session_id, principal=Principal(), now=_stale_now(reservation)
    )

    current = sessions.get_reservation_scoped(reservation.experiment_id, principal=Principal())
    assert current.state == 'compensation_required'
    assert current.resolution_code == 'legacy_protocol_unknown'
    assert result['status'] == 'manual_review_required'
    assert result['items'][0]['requires_manual_review'] is True


def test_mapping_to_missing_run_fails_closed(tmp_path):
    _, sessions, runs, submissions, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    run = _submit(submissions, session, reservation).record
    runs.cancel_queued_unstarted_scoped(
        run.run_id, now=datetime.now(timezone.utc), principal=Principal()
    )
    runs.delete_terminal(run.run_id)

    result = reconciler.reconcile_session(
        session_id=session.session_id, principal=Principal(), now=_stale_now(reservation)
    )

    current = sessions.get_reservation_scoped(reservation.experiment_id, principal=Principal())
    assert current.state == 'compensation_required'
    assert current.resolution_code == 'mapped_run_unavailable'
    assert result['status'] == 'manual_review_required'


def test_concurrent_and_repeated_reconciliation_only_transitions_once(tmp_path):
    _, sessions, _, _, reconciler, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    now = _stale_now(reservation)

    def reconcile(_index):
        return reconciler.reconcile_session(
            session_id=session.session_id, principal=Principal(), now=now
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(reconcile, range(8)))
    repeated = reconcile(9)

    current = sessions.get_reservation_scoped(reservation.experiment_id, principal=Principal())
    assert current.state == 'released'
    assert current.reconcile_attempt_count == 1
    assert len(reconciler.runs.list()) == 0
    assert repeated['summary']['recovered_bound'] == 0
    assert repeated['summary']['released'] == 0


def test_cross_principal_endpoint_is_404(tmp_path, monkeypatch):
    from backend.app.http.principal import get_principal
    from backend.app.routers import agent as agent_router

    storage, sessions, _, _, _, session = _environment(
        tmp_path, principal=Principal('owner-a', 'tenant-a')
    )
    monkeypatch.setattr(agent_router, 'AGENT_DATABASE', storage / 'agent.sqlite3')
    monkeypatch.setattr(agent_router, 'RUNS_DATABASE', storage / 'runs.sqlite3')
    app.dependency_overrides[get_principal] = lambda: Principal('owner-b', 'tenant-a')
    try:
        response = TestClient(app).post(
            f'/api/agent/sessions/{session.session_id}/reconcile'
        )
    finally:
        app.dependency_overrides.pop(get_principal, None)

    assert response.status_code == 404
    assert response.json()['detail']['code'] == 'agent_session_not_found'


def test_reconcile_endpoint_has_empty_body_and_safe_response(tmp_path, monkeypatch):
    from backend.app.routers import agent as agent_router

    storage, sessions, _, _, _, session = _environment(tmp_path)
    reservation = _reserve(sessions, session)
    monkeypatch.setattr(agent_router, 'AGENT_DATABASE', storage / 'agent.sqlite3')
    monkeypatch.setattr(agent_router, 'RUNS_DATABASE', storage / 'runs.sqlite3')
    # 让记录在真实路由使用的服务端阈值下变为 stale，不给 HTTP 暴露阈值参数。
    with sqlite3.connect(storage / 'agent.sqlite3') as connection:
        old = (datetime.now(timezone.utc) - timedelta(seconds=301)).isoformat()
        connection.execute(
            'UPDATE agent_experiment_reservations_v1 SET updated_at=? WHERE reservation_id=?',
            (old, reservation.experiment_id),
        )

    client = TestClient(app)
    rejected = client.post(
        f'/api/agent/sessions/{session.session_id}/reconcile',
        json={'stale_after_seconds': 0},
    )
    response = client.post(f'/api/agent/sessions/{session.session_id}/reconcile')

    assert rejected.status_code == 422
    assert response.status_code == 200
    body = response.json()
    assert body['contract_version'] == 'agent-session-v1'
    assert set(body) == {'contract_version', 'session_id', 'status', 'summary', 'items'}
    flat = json.dumps(body, ensure_ascii=False).lower()
    for forbidden in (
        'path', 'traceback', 'test', 'artifact', 'prediction', 'confusion',
        'explainability', '.csv', '.json', 'run_id', 'reservation_id',
    ):
        assert forbidden not in flat
