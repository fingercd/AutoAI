from datetime import datetime, timedelta, timezone

from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker


def test_expired_worker_claim_becomes_stopped_without_silent_retry(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    run = repo.create_queued(dataset_id='ds-1', config={'model_type': 'pls_da'})
    now = datetime.now(timezone.utc)
    first = repo.claim_next(worker_id='dead', now=now, lease_seconds=1)
    repo.stop_expired(now=now + timedelta(seconds=2))

    worker = RunWorker(repository=repo, worker_id='live', execute=lambda _: {'manifest_name': 'manifest.json'}, now=lambda: now + timedelta(seconds=2))
    assert worker.run_once() is False
    stopped = repo.get(run.run_id)
    assert stopped.state == 'cancelled'
    assert stopped.progress['stop_reason'] == 'worker_interrupted'
    assert stopped.claim_token is None
