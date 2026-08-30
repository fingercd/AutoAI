"""Agent Adapter 的存储覆盖、Principal scope 和 reservation 并发回归。"""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from backend.app.agent.contracts import CreateAgentExperimentRequest, CreateAgentSessionRequest
from backend.app.agent.repository import AgentConfigCollision, AgentSessionRepository
from backend.app.agent.service import AgentService
from backend.app.contracts import TrainingConfigValidationError
from backend.app.datasets.repository import DatasetRepository
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunRepository
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def _build_service(tmp_path: Path, principal: Principal) -> tuple[
    AgentService, str, AgentSessionRepository, RunRepository
]:
    storage = tmp_path / 'storage'
    upload_dir = storage / 'uploads'
    upload_dir.mkdir(parents=True)
    dataset_path = upload_dir / 'agent.csv'
    write_grouped_classification_csv(
        dataset_path, groups_per_class=6, repeats=2, feature_count=8
    )

    datasets = DatasetRepository(storage / 'datasets.sqlite3', storage_root=storage)
    datasets.initialize()
    dataset = datasets.register(
        dataset_path, original_name=dataset_path.name, principal=principal
    )
    runs = RunRepository(storage / 'runs.sqlite3')
    runs.initialize()
    sessions = AgentSessionRepository(
        storage / 'agent.sqlite3',
        runs_database_path=storage / 'runs.sqlite3',
    )
    sessions.initialize()
    service = AgentService(
        session_repository=sessions,
        run_repository=runs,
        dataset_repository=datasets,
    )
    return service, dataset.dataset_id, sessions, runs


def _session(service: AgentService, dataset_id: str, principal: Principal, *, max_runs: int = 1):
    return service.create_session(
        CreateAgentSessionRequest(
            dataset_id=dataset_id,
            selection_metric='macro_f1',
            allowed_models=['logistic_regression', 'svm', 'random_forest'],
            max_runs=max_runs,
            seed=42,
        ),
        principal=principal,
    )


def _submit(
    service: AgentService,
    session_id: str,
    action: dict[str, str],
    principal: Principal,
) -> tuple[str, object]:
    try:
        result = service.create_experiment(
            session_id=session_id,
            payload=CreateAgentExperimentRequest(**action),
            principal=principal,
        )
        return 'ok', result
    except Exception as exc:  # assertions below check the concrete domain error
        return 'error', exc


@pytest.mark.parametrize('same_action', [True, False])
def test_concurrent_reservation_allows_at_most_one_experiment(tmp_path, same_action):
    principal = Principal(owner_id='owner-a', tenant_id='tenant-a')
    service, dataset_id, sessions, runs = _build_service(tmp_path, principal)
    session = _session(service, dataset_id, principal)
    action_a = {
        'model_type': 'svm',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    action_b = dict(action_a)
    if not same_action:
        action_b['model_type'] = 'logistic_regression'

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda action: _submit(service, session['session_id'], action, principal),
                [action_a, action_b],
            )
        )

    assert sum(kind == 'ok' for kind, _ in results) == 1
    failures = [value for kind, value in results if kind == 'error']
    assert len(failures) == 1
    assert isinstance(failures[0], AgentConfigCollision)
    assert len(sessions.list_experiments_scoped(session['session_id'], principal=principal)) == 1
    assert sessions.count_reservations_scoped(
        session_id=session['session_id'], principal=principal
    ) == 0
    scoped_runs = runs.list_scoped(principal=principal)
    assert len(scoped_runs) == 1
    assert scoped_runs[0].state == 'queued'
    assert scoped_runs[0].dataset_snapshot['dataset_id'] == dataset_id
    assert scoped_runs[0].dataset_snapshot['sha256']


def test_reservation_atomically_rechecks_bound_active_run(tmp_path):
    principal = Principal(owner_id='owner-a', tenant_id='tenant-a')
    service, dataset_id, sessions, _runs = _build_service(tmp_path, principal)
    session = _session(service, dataset_id, principal, max_runs=2)
    first = service.create_experiment(
        session_id=session['session_id'],
        payload=CreateAgentExperimentRequest(model_type='logistic_regression'),
        principal=principal,
    )
    assert first['state'] == 'queued'
    with pytest.raises(AgentConfigCollision, match='非终态 Run'):
        sessions.reserve_experiment(
            session_id=session['session_id'],
            config_hash='another-config',
            action_json={'model_type': 'svm'},
            principal=principal,
        )


def test_dataset_scope_is_checked_when_creating_session(tmp_path):
    owner_a = Principal(owner_id='owner-a', tenant_id='tenant-a')
    owner_b = Principal(owner_id='owner-b', tenant_id='tenant-b')
    service, dataset_id, _, _ = _build_service(tmp_path, owner_a)

    with pytest.raises(TrainingConfigValidationError, match='dataset_id 不存在'):
        _session(service, dataset_id, owner_b)


def test_bind_failure_releases_reservation_and_cancels_queued_run(tmp_path, monkeypatch):
    principal = Principal(owner_id='owner-a', tenant_id='tenant-a')
    service, dataset_id, sessions, runs = _build_service(tmp_path, principal)
    session = _session(service, dataset_id, principal)

    def fail_bind(**_kwargs):
        raise RuntimeError('simulated agent DB bind failure')

    monkeypatch.setattr(service.sessions, 'bind_reservation', fail_bind)
    with pytest.raises(RuntimeError, match='simulated agent DB bind failure'):
        service.create_experiment(
            session_id=session['session_id'],
            payload=CreateAgentExperimentRequest(
                model_type='svm', normalization='zscore', class_balance='none'
            ),
            principal=principal,
        )

    assert sessions.count_reservations_scoped(
        session_id=session['session_id'], principal=principal
    ) == 0
    scoped_runs = runs.list_scoped(principal=principal)
    assert len(scoped_runs) == 1
    assert scoped_runs[0].state == 'cancelled'
    assert scoped_runs[0].progress['stop_reason'] == 'agent_bind_failed'


def test_storage_override_requires_absolute_path(monkeypatch, tmp_path):
    import backend.app.paths as paths

    original_storage = paths.STORAGE_DIR
    monkeypatch.setenv('AUTOAI_STORAGE_DIR', str(tmp_path / 'shared-storage'))
    overridden = importlib.reload(paths)
    assert overridden.STORAGE_DIR == (tmp_path / 'shared-storage').resolve()
    assert overridden.RUNS_DATABASE == overridden.STORAGE_DIR / 'runs.sqlite3'
    assert overridden.AGENT_DATABASE == overridden.STORAGE_DIR / 'agent.sqlite3'

    monkeypatch.setenv('AUTOAI_STORAGE_DIR', 'relative-storage')
    with pytest.raises(RuntimeError, match='绝对路径'):
        importlib.reload(paths)

    monkeypatch.delenv('AUTOAI_STORAGE_DIR', raising=False)
    restored = importlib.reload(paths)
    assert restored.STORAGE_DIR == original_storage
