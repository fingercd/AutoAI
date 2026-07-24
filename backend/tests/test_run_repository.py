from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from backend.app.runs.repository import InvalidRunTransition, RunNotFound, RunRepository


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
