from datetime import datetime, timedelta, timezone

from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker


def test_expired_worker_claim_is_recovered_without_double_success(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    run = repo.create_queued(dataset_id='ds-1', config={'model_type': 'pls_da'})
    now = datetime.now(timezone.utc)
    first = repo.claim_next(worker_id='dead', now=now, lease_seconds=1)
    repo.requeue_expired(now=now + timedelta(seconds=2))

    worker = RunWorker(repository=repo, worker_id='live', execute=lambda _: {'manifest_name': 'manifest.json'}, now=lambda: now + timedelta(seconds=2))
    assert worker.run_once() is True
    assert repo.get(run.run_id).state == 'succeeded'
    assert repo.get(run.run_id).claim_token != first.claim_token
