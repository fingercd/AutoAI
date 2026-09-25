from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from backend.app.runs.contracts import Principal
from backend.app.runs.repository import (
    InvalidRunTransition,
    RunNotFound,
    RunRepository,
    RunDeadlineExceeded,
    RunSubmissionKeyConflict,
)


def test_success_publication_rechecks_supervised_deadline(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    created = repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    claim = repo.claim_next(worker_id='deadline', now=datetime.now(timezone.utc))
    assert claim is not None
    with pytest.raises(RunDeadlineExceeded):
        repo.finish_success(created.run_id, claim_token=claim.claim_token or '',
                            now=datetime.now(timezone.utc),
                            deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert repo.get(created.run_id).state == 'running'
    repo.finish_failure(created.run_id, claim_token=claim.claim_token or '',
                        now=datetime.now(timezone.utc), error='deadline')
    assert repo.get(created.run_id).state == 'failed'


def test_supervised_success_requires_live_lease(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    created = repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    started = datetime.now(timezone.utc)
    claim = repo.claim_next(worker_id='deadline', now=started, lease_seconds=1)
    assert claim is not None
    with pytest.raises(InvalidRunTransition):
        repo.finish_success(created.run_id, claim_token=claim.claim_token or '',
                            now=started + timedelta(seconds=2),
                            deadline_at=started + timedelta(seconds=60))
    assert repo.get(created.run_id).state == 'running'


def test_cancelled_run_cannot_be_reclaimed_or_completed(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    run = repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    claim = repo.claim_next(worker_id='worker-a', now=datetime.now(timezone.utc))
    assert claim is not None and claim.run_id == run.run_id

    cancelled = repo.cancel(run.run_id, now=datetime.now(timezone.utc))
    assert cancelled.state == 'cancelled'
    assert cancelled.progress['stop_status'] == 'stopped'
    assert cancelled.progress['stop_reason'] == 'user_requested'
    assert cancelled.claim_token is None
    assert cancelled.worker_id is None
    assert repo.claim_next(worker_id='worker-b', now=datetime.now(timezone.utc)) is None

    with pytest.raises(InvalidRunTransition):
        repo.finish_success(run.run_id, claim_token=claim.claim_token, now=datetime.now(timezone.utc))


def test_expired_lease_is_stopped_once_and_never_reclaimed(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    now = datetime.now(timezone.utc)
    first = repo.claim_next(worker_id='dead-worker', now=now, lease_seconds=1)
    assert first is not None

    assert repo.stop_expired(now=now + timedelta(seconds=2)) == [first.run_id]
    assert repo.stop_expired(now=now + timedelta(seconds=3)) == []
    stopped = repo.get(first.run_id)
    assert stopped.state == 'cancelled'
    assert stopped.progress['stop_reason'] == 'worker_interrupted'
    assert stopped.progress['stop_status'] == 'stopped'
    assert repo.claim_next(worker_id='live-worker', now=now + timedelta(seconds=3)) is None


@pytest.mark.parametrize('state', ['succeeded', 'failed', 'cancelled'])
def test_terminal_run_can_be_permanently_deleted(tmp_path, state):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    record = repo.import_legacy(
        run_id=f'{state}-run',
        state=state,
        config={'model_type': 'pls_da'},
        dataset_id=None,
        legacy_data_path=None,
    )

    repo.delete_terminal(record.run_id)

    assert not repo.exists(record.run_id)
    with pytest.raises(RunNotFound):
        repo.get(record.run_id)


@pytest.mark.parametrize('state', ['queued', 'running'])
def test_active_run_cannot_be_permanently_deleted(tmp_path, state):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    record = repo.import_legacy(
        run_id=f'{state}-run',
        state=state,
        config={'model_type': 'pls_da'},
        dataset_id=None,
        legacy_data_path=None,
    )

    with pytest.raises(InvalidRunTransition):
        repo.delete_terminal(record.run_id)

    assert repo.exists(record.run_id)


def test_worker_health_requires_all_live_workers_to_match_expected_contract(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    now = datetime.now(timezone.utc)
    repo.record_worker_heartbeat(
        worker_id='current-worker',
        now=now,
        contract_version='run-artifact-manifest-v2',
    )

    current = repo.worker_health(
        now=now,
        stale_seconds=30,
        expected_contract_version='run-artifact-manifest-v2',
    )

    assert current['available'] is True
    assert current['compatible'] is True
    assert current['contract_version'] == 'run-artifact-manifest-v2'

    repo.record_worker_heartbeat(
        worker_id='legacy-worker',
        now=now,
        contract_version=None,
    )
    mixed = repo.worker_health(
        now=now,
        stale_seconds=30,
        expected_contract_version='run-artifact-manifest-v2',
    )

    assert mixed['available'] is True
    assert mixed['compatible'] is False
    assert mixed['contract_version'] is None
    assert {item['compatible'] for item in mixed['workers']} == {False, True}


def test_initialize_migrates_legacy_worker_heartbeat_table_without_dropping_rows(tmp_path):
    database = tmp_path / 'runs.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.execute(
            '''
            CREATE TABLE worker_heartbeats (
                worker_id TEXT PRIMARY KEY,
                last_seen_at TEXT NOT NULL,
                active_run_id TEXT
            )
            '''
        )
        connection.execute(
            'INSERT INTO worker_heartbeats(worker_id, last_seen_at, active_run_id) VALUES (?, ?, ?)',
            ('legacy-worker', datetime.now(timezone.utc).isoformat(), None),
        )

    repo = RunRepository(database)
    repo.initialize()
    repo.initialize()

    health = repo.worker_health(
        now=datetime.now(timezone.utc),
        stale_seconds=30,
        expected_contract_version='run-artifact-manifest-v2',
    )
    assert health['available'] is True
    assert health['compatible'] is False
    assert health['workers'][0]['worker_id'] == 'legacy-worker'
    assert health['workers'][0]['contract_version'] is None


def test_submission_mapping_is_atomic_and_idempotent(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    principal = Principal('owner-a', 'tenant-a')
    kwargs = {
        'dataset_id': 'ds-1',
        'config': {'model_type': 'logistic_regression'},
        'principal': principal,
        'submission_key': 'reservation-1',
        'submission_source': 'agent',
        'submission_payload_hash': 'a' * 64,
    }

    first = repo.create_queued(**kwargs)
    second = repo.create_queued(**kwargs)

    assert second.run_id == first.run_id
    assert len(repo.list()) == 1
    mapping = repo.lookup_submission_mapping('reservation-1', principal=principal)
    assert mapping.status == 'found'
    assert mapping.run_id == first.run_id


def test_submission_mapping_conflicts_fail_closed_without_leaking_run_id(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    first_principal = Principal('owner-a', 'tenant-a')
    run = repo.create_queued(
        dataset_id='ds-1', config={}, principal=first_principal,
        submission_key='reservation-1', submission_source='agent',
        submission_payload_hash='a' * 64,
    )

    with pytest.raises(RunSubmissionKeyConflict):
        repo.create_queued(
            dataset_id='ds-1', config={}, principal=first_principal,
            submission_key='reservation-1', submission_source='agent',
            submission_payload_hash='b' * 64,
        )
    with pytest.raises(RunSubmissionKeyConflict) as cross_scope:
        repo.create_queued(
            dataset_id='ds-1', config={}, principal=Principal('owner-b', 'tenant-a'),
            submission_key='reservation-1', submission_source='agent',
            submission_payload_hash='a' * 64,
        )

    assert run.run_id not in str(cross_scope.value)
    invisible = repo.lookup_submission_mapping(
        'reservation-1', principal=Principal('owner-b', 'tenant-a')
    )
    assert invisible.status == 'scope_mismatch'
    assert invisible.run_id is None
    assert len(repo.list()) == 1


def test_cancel_queued_unstarted_never_cancels_started_run(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    principal = Principal('owner-a', 'tenant-a')
    queued = repo.create_queued(dataset_id='ds-1', config={}, principal=principal)
    cancelled = repo.cancel_queued_unstarted_scoped(
        queued.run_id, now=datetime.now(timezone.utc), principal=principal
    )
    assert cancelled.state == 'cancelled'
    assert cancelled.started_at is None

    started = repo.create_queued(dataset_id='ds-1', config={}, principal=principal)
    claimed = repo.claim_next(worker_id='worker-a', now=datetime.now(timezone.utc))
    assert claimed is not None and claimed.run_id == started.run_id
    with pytest.raises(InvalidRunTransition):
        repo.cancel_queued_unstarted_scoped(
            started.run_id, now=datetime.now(timezone.utc), principal=principal
        )
    assert repo.get(started.run_id).state == 'running'
