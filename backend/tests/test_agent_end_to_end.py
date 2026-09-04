"""Agent API → queued Run → 真实 Worker → Manifest → Finalize 闭环。"""

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.runs.artifacts import RunArtifactWriter
from backend.app.runs.execution import execute_claimed_run
from backend.app.runs.repository import RunRepository
from backend.app.runs.status_projection import project_status
from backend.app.runs.worker import RunWorker
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def _isolate_storage(monkeypatch, tmp_path):
    import backend.app.paths as paths
    import backend.app.routers.agent as agent_router
    import backend.app.routers.datasets as datasets_router
    import backend.app.routers.deps as deps_router
    import backend.app.routers.runs as runs_router
    import backend.app.training as training

    storage = tmp_path / 'storage'; uploads = storage / 'uploads'; runs_dir = storage / 'runs'
    uploads.mkdir(parents=True); runs_dir.mkdir(parents=True)
    values = {
        'STORAGE_DIR': storage, 'UPLOADS_DIR': uploads, 'RUNS_DIR': runs_dir,
        'DATASETS_DATABASE': storage / 'datasets.sqlite3',
        'RUNS_DATABASE': storage / 'runs.sqlite3', 'AGENT_DATABASE': storage / 'agent.sqlite3',
    }
    for name, value in values.items():
        if hasattr(paths, name): monkeypatch.setattr(paths, name, value)
    for module in (agent_router, datasets_router, deps_router, runs_router):
        for name, value in values.items():
            if hasattr(module, name): monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(training, 'RUNS_DIR', runs_dir)
    return storage, uploads, runs_dir


def test_real_worker_agent_round_trip(monkeypatch, tmp_path):
    storage, _uploads, runs_dir = _isolate_storage(monkeypatch, tmp_path)
    source = tmp_path / 'grouped.csv'
    write_grouped_classification_csv(source, groups_per_class=6, repeats=2, feature_count=6)
    client = TestClient(app)
    with source.open('rb') as handle:
        uploaded = client.post('/api/datasets/upload', files={'file': ('grouped.csv', handle, 'text/csv')})
    assert uploaded.status_code == 200
    dataset_id = uploaded.json()['dataset_id']
    session = client.post('/api/agent/sessions', json={
        'dataset_id': dataset_id, 'selection_metric': 'macro_f1',
        'allowed_models': ['logistic_regression'], 'max_runs': 1, 'seed': 42,
        'client_request_id': 'e2e-session-1',
    })
    assert session.status_code == 201
    session_id = session.json()['session_id']
    experiment_body = {
        'model_type': 'logistic_regression', 'normalization': 'zscore',
        'class_balance': 'none', 'client_request_id': 'e2e-experiment-1',
    }
    experiment = client.post(f'/api/agent/sessions/{session_id}/experiments', json=experiment_body)
    assert experiment.status_code == 202
    run_id = experiment.json()['run_id']
    assert experiment.json()['state'] == 'queued'
    assert client.post(f'/api/agent/sessions/{session_id}/experiments', json=experiment_body).json()['run_id'] == run_id

    repository = RunRepository(storage / 'runs.sqlite3'); repository.initialize()
    observed_states = ['queued']

    def execute(record):
        observed_states.append(repository.get(record.run_id).state)
        return execute_claimed_run(record, repository=repository)

    worker = RunWorker(
        repository=repository, worker_id='e2e-worker', execute=execute,
        now=lambda: datetime.now(timezone.utc), heartbeat_seconds=60,
        project_status=lambda record: project_status(runs_dir / record.run_id, record),
    )
    assert worker.run_once() is True
    observed_states.append(repository.get(run_id).state)
    assert observed_states == ['queued', 'running', 'succeeded']

    writer = RunArtifactWriter(runs_dir / run_id)
    manifest, descriptors = writer.descriptors(run_id=run_id)
    assert manifest['run_id'] == run_id
    assert not any(
        item['integrity'] in {'missing', 'corrupt'} and (item['required'] or item['applicable'])
        for item in descriptors
    )
    writer.resolve_download('metrics.json')

    feedback = client.get(f'/api/agent/sessions/{session_id}/experiments/{run_id}/feedback')
    assert feedback.status_code == 200
    assert feedback.json()['validation']['status'] == 'ready'
    assert 'test' not in str(feedback.json()).lower()

    finalized = client.post(
        f'/api/agent/sessions/{session_id}/finalize', json={'selected_run_id': run_id}
    )
    assert finalized.status_code == 200
    blocked = client.post(
        f'/api/agent/sessions/{session_id}/experiments',
        json={'model_type': 'logistic_regression'},
    )
    assert blocked.status_code == 409
    assert blocked.json()['detail']['code'] == 'agent_session_finalized'

    human_result = client.get(f'/api/training/runs/{run_id}/result')
    assert human_result.status_code == 200
    assert human_result.json()['schema_version'] == 'run-result-v1'
    assert 'test' not in str(client.get(f'/api/agent/sessions/{session_id}').json()).lower()
