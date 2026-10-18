from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier

import pytest

from backend.app.runs.repository import RunRepository
from backend.app.runs.resources import THREAD_VARIABLES, worker_environment, worker_resource_plan


@pytest.mark.parametrize('cpus,expected', [(1, (1, 1)), (2, (1, 1)), (4, (3, 1)), (24, (4, 5))])
def test_default_process_thread_budget(cpus, expected):
    assert worker_resource_plan(cpu_count=cpus) == expected


@pytest.mark.parametrize('workers,threads', [(0, 1), (25, 1), (4, 0), (4, 7)])
def test_invalid_resource_plan_rejected(workers, threads):
    with pytest.raises(ValueError):
        worker_resource_plan(workers, threads, cpu_count=24)


def test_worker_environment_caps_all_native_threadpools():
    base = {'PATH': 'unchanged', 'OMP_NUM_THREADS': '24'}
    env = worker_environment(5, 4, base=base)
    assert all(env[key] == '5' for key in THREAD_VARIABLES)
    assert env['AUTOAI_WORKER_COUNT'] == '4'
    assert env['AUTOAI_WORKER_THREADS'] == '5'
    assert base['OMP_NUM_THREADS'] == '24'
    assert env['PATH'] == 'unchanged'


def test_four_concurrent_claims_are_unique_and_overlap(tmp_path):
    path = tmp_path / 'runs.sqlite3'
    repo = RunRepository(path)
    repo.initialize()
    for _ in range(4):
        repo.create_queued(dataset_id='ds', config={'model_type': 'pls_da'})
    barrier = Barrier(4)

    def claim(index):
        barrier.wait(timeout=10)
        run = RunRepository(path).claim_next(worker_id=f'w{index}', now=datetime.now(timezone.utc))
        assert run is not None
        barrier.wait(timeout=10)  # All four own a running Run at the same time.
        return run

    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(claim, range(4)))
    assert len({run.run_id for run in runs}) == 4
    assert len({run.claim_token for run in runs}) == 4
    assert all(run.state == 'running' for run in repo.list())


def test_gpu_guard_skips_blocked_deep_queue_head_and_releases_on_finish(tmp_path):
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    # Creation times establish FIFO independent of randomized Run IDs.
    first = repo.create_queued(dataset_id='ds', config={'model_type': 'cnn1d'})
    second = repo.create_queued(dataset_id='ds', config={'model_type': 'cnn1d'})
    cpu = repo.create_queued(dataset_id='ds', config={'model_type': 'pls_da'})
    claim = lambda name: repo.claim_next(worker_id=name, now=datetime.now(timezone.utc), exclusive_model_types=('cnn1d',))
    active = claim('gpu')
    assert active.run_id == first.run_id
    assert claim('cpu').run_id == cpu.run_id
    assert claim('idle') is None
    repo.finish_success(first.run_id, claim_token=active.claim_token, now=datetime.now(timezone.utc))
    assert claim('next-gpu').run_id == second.run_id


def test_gpu_exclusive_claim_is_atomic_between_workers(tmp_path):
    path = tmp_path / 'runs.sqlite3'
    repo = RunRepository(path)
    repo.initialize()
    for _ in range(4):
        repo.create_queued(dataset_id='ds', config={'model_type': 'cnn1d'})
    barrier = Barrier(4)

    def claim(index):
        barrier.wait(timeout=10)
        return RunRepository(path).claim_next(worker_id=str(index), now=datetime.now(timezone.utc), exclusive_model_types=('cnn1d',))

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(run is not None for run in pool.map(claim, range(4))) == 1


def test_tree_models_obey_worker_budget(monkeypatch):
    from backend.app.models.random_forest import build_random_forest
    from backend.app.models.xgboost import build_xgboost
    monkeypatch.setenv('AUTOAI_WORKER_THREADS', '5')
    assert build_random_forest().n_jobs == 5
    import numpy as np
    model = build_xgboost(np.array([0, 1]), 2, 'none', 42)
    assert model.get_params()['n_jobs'] == 5


def test_launcher_starts_and_stops_all_workers_even_on_start_failure(monkeypatch):
    import run
    import uvicorn
    events = []

    class Supervisor:
        def __init__(self, command, *, environment):
            self.index = len([e for e in events if e[0] == 'new'])
            assert environment['AUTOAI_WORKER_THREADS'] == '1'
            events.append(('new', self.index))

        def start(self):
            events.append(('start', self.index))

        def stop(self):
            events.append(('stop', self.index))

    monkeypatch.setattr(run, 'WorkerSupervisor', Supervisor)
    monkeypatch.setattr(uvicorn, 'run', lambda *a, **k: events.append(('web', 0)))
    monkeypatch.setattr(run.sys, 'argv', ['run.py', '--no-browser', '--workers', '2', '--threads-per-worker', '1'])
    run.main()
    assert events == [('new', 0), ('start', 0), ('new', 1), ('start', 1), ('web', 0), ('stop', 0), ('stop', 1)]
    events.clear()

    def fail(self):
        events.append(('start', self.index))
        if self.index == 1:
            raise OSError('spawn failed')

    monkeypatch.setattr(Supervisor, 'start', fail)
    with pytest.raises(OSError, match='spawn failed'):
        run.main()
    assert events[-2:] == [('stop', 0), ('stop', 1)]


def test_supervisor_passes_budget_before_child_imports(monkeypatch):
    import run
    seen = []
    monkeypatch.setattr(run.subprocess, 'Popen', lambda command, **kw: seen.append((command, kw)))
    env = worker_environment(2, 2, base={})
    supervisor = run.WorkerSupervisor(['python', '-m', 'backend.app.runs.worker'], environment=env)
    supervisor._spawn()
    assert seen[0][1]['env'] == env


def test_supervisor_shutdown_cannot_leave_a_concurrently_respawned_child(monkeypatch):
    from threading import Event, Thread
    import run
    from backend.tests.test_run_launcher import _FakeProcess
    entered, release = Event(), Event()
    process = _FakeProcess()

    def spawn(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return process

    monkeypatch.setattr(run.subprocess, 'Popen', spawn)
    supervisor = run.WorkerSupervisor(['worker'])
    spawning = Thread(target=supervisor._spawn)
    spawning.start()
    assert entered.wait(timeout=5)
    stopping = Thread(target=supervisor.stop)
    stopping.start()
    assert supervisor._stop.wait(timeout=5)
    release.set()
    spawning.join(timeout=5)
    stopping.join(timeout=5)
    assert not spawning.is_alive() and not stopping.is_alive()
    assert process.terminated
    supervisor._spawn()
    assert supervisor._process is process


@pytest.mark.parametrize('mode,total', [('stratified_holdout', 1), ('leave_one_sample_id_cv', 24)])
def test_single_run_exposes_expected_experiment_count_while_queued(tmp_path, monkeypatch, mode, total):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.routers import runs
    from backend.app.routers.deps import TrainingDataReference
    repo = RunRepository(tmp_path / 'runs.sqlite3')
    repo.initialize()
    monkeypatch.setattr(runs, 'get_run_repository', lambda: repo)
    monkeypatch.setattr(runs, 'get_run_dir', lambda rid: tmp_path / rid)
    monkeypatch.setattr(runs, 'resolve_training_data_reference', lambda payload, **kw: TrainingDataReference(
        dataset_id='d', legacy_path=None, dataset_name='fixture.csv',
        test_dataset_id=None, test_legacy_path=None, test_dataset_name=None))

    def snapshot(**kwargs):
        assert kwargs['include_summary'] is True
        return {'sample_id_count': 24, 'sha256': 'a' * 64}

    monkeypatch.setattr(runs, '_dataset_snapshot_for_reference', snapshot)
    client = TestClient(app)
    response = client.post('/api/train', json={'dataset_id': 'd', 'config': {
        'model_type': 'pls_da', 'experiment_version': 'word-0904', 'feature_scheme': 'bin_3', 'split_mode': mode}})
    assert response.status_code == 202, response.text
    rid = response.json()['run_id']
    status = client.get(f'/api/training/runs/{rid}').json()
    assert status['state'] == 'queued'
    assert status['config']['expected_outer_folds'] == total
