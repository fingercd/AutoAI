from datetime import datetime, timedelta, timezone

import pytest

from backend.app.runs.repository import InvalidRunTransition, RunRepository


def test_cancelled_run_cannot_be_reclaimed_or_completed(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    run = repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    claim = repo.claim_next(worker_id='worker-a', now=datetime.now(timezone.utc))
    assert claim is not None and claim.run_id == run.run_id

    cancelled = repo.cancel(run.run_id, now=datetime.now(timezone.utc))
    assert cancelled.state == 'cancelled'
    assert repo.claim_next(worker_id='worker-b', now=datetime.now(timezone.utc)) is None

    with pytest.raises(InvalidRunTransition):
        repo.finish_success(run.run_id, claim_token=claim.claim_token, now=datetime.now(timezone.utc))


def test_expired_lease_is_requeued_once_and_claimed_once(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    now = datetime.now(timezone.utc)
    first = repo.claim_next(worker_id='dead-worker', now=now, lease_seconds=1)
    assert first is not None

    repo.requeue_expired(now=now + timedelta(seconds=2))
    second = repo.claim_next(worker_id='live-worker', now=now + timedelta(seconds=2))
    assert second is not None
    assert second.run_id == first.run_id
    assert second.claim_token != first.claim_token
