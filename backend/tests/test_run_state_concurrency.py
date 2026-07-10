from datetime import datetime, timezone
from threading import Event, Thread

import pytest

from backend.app.runs.repository import InvalidRunTransition, RunRepository


def test_cancel_wins_against_a_stale_success_attempt(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    run = repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    claim = repo.claim_next(worker_id='worker-a', now=datetime.now(timezone.utc))
    cancelled = Event()

    def cancel() -> None:
        repo.cancel(run.run_id, now=datetime.now(timezone.utc))
        cancelled.set()

    thread = Thread(target=cancel)
    thread.start()
    assert cancelled.wait(timeout=5)
    thread.join()
    with pytest.raises(InvalidRunTransition):
        repo.finish_success(run.run_id, claim_token=claim.claim_token, now=datetime.now(timezone.utc))

    assert repo.get(run.run_id).state == 'cancelled'
