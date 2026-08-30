from datetime import datetime, timezone

from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker


def test_run_once_claims_and_finishes_one_queued_run(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    created = repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    calls = []

    def execute(run):
        calls.append(run.run_id)
        return {'manifest_name': 'manifest.json'}

    worker = RunWorker(repository=repo, worker_id='test-worker', execute=execute, now=lambda: datetime.now(timezone.utc))
    assert worker.run_once() is True
    assert calls == [created.run_id]
    assert repo.get(created.run_id).state == 'succeeded'


def test_run_once_leaves_cancelled_run_cancelled(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    run = repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    repo.cancel(run.run_id, now=datetime.now(timezone.utc))
    worker = RunWorker(repository=repo, worker_id='test-worker', execute=lambda _: {'manifest_name': 'manifest.json'}, now=lambda: datetime.now(timezone.utc))
    assert worker.run_once() is False
    assert repo.get(run.run_id).state == 'cancelled'


def test_worker_renews_lease_while_execution_is_running(tmp_path):
    from threading import Event, Thread
    import time

    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    created = repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    started, release = Event(), Event()

    def execute(_):
        started.set()
        assert release.wait(timeout=5)
        return {'manifest_name': 'manifest.json'}

    worker = RunWorker(
        repository=repo,
        worker_id='test-worker',
        execute=execute,
        now=lambda: datetime.now(timezone.utc),
        heartbeat_seconds=0.01,
    )
    thread = Thread(target=worker.run_once)
    thread.start()
    assert started.wait(timeout=5)
    before = repo.get(created.run_id)
    assert before.state == 'running'
    time.sleep(0.05)
    assert repo.get(created.run_id).version > before.version
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert repo.get(created.run_id).state == 'succeeded'


def test_worker_notifies_lease_loss_after_user_stop(tmp_path):
    from threading import Event, Thread

    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    created = repo.create_queued(dataset_id='ds_1', config={'model_type': 'dscarnet'})
    started, release, lost = Event(), Event(), Event()
    lost_run_ids = []

    def execute(_):
        started.set()
        assert release.wait(timeout=5)
        return {'manifest_name': 'manifest.json'}

    def on_lease_lost(run_id):
        lost_run_ids.append(run_id)
        lost.set()

    worker = RunWorker(
        repository=repo,
        worker_id='test-worker',
        execute=execute,
        now=lambda: datetime.now(timezone.utc),
        heartbeat_seconds=0.01,
        on_lease_lost=on_lease_lost,
    )
    thread = Thread(target=worker.run_once)
    thread.start()
    assert started.wait(timeout=5)

    repo.cancel(created.run_id, now=datetime.now(timezone.utc))

    assert lost.wait(timeout=5)
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert lost_run_ids == [created.run_id]
    assert repo.get(created.run_id).state == 'cancelled'


def test_worker_projects_failed_record_after_execution_error(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    created = repo.create_queued(dataset_id='ds_1', config={'model_type': 'pls_da'})
    projected = []
    discarded = []

    def execute(_):
        raise RuntimeError('training exploded')

    worker = RunWorker(
        repository=repo,
        worker_id='test-worker',
        execute=execute,
        now=lambda: datetime.now(timezone.utc),
        project_status=projected.append,
        discard_artifacts=discarded.append,
    )

    assert worker.run_once() is True
    failed = repo.get(created.run_id)
    assert failed.state == 'failed'
    assert failed.error == 'training exploded'
    assert len(projected) == 1
    assert projected[0].state == 'failed'
    assert projected[0].error == 'training exploded'
    assert discarded == [created.run_id]
