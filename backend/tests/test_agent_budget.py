from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.app.agent.budget import (
    AgentBudgetEstimateError,
    estimate_model_fit_upper_bound,
)
from backend.app.agent.contracts import (
    CreateAgentExperimentRequest,
    CreateAgentSessionRequest,
)
from backend.app.agent.repository import (
    AgentBudgetAccountingError,
    AgentBudgetExceeded,
    AgentSessionRepository,
)
from backend.app.agent.service import AgentService
from backend.app.datasets.repository import DatasetRepository
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunRepository
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def _limits(*, max_model_fits: int = 8) -> dict[str, int]:
    return {
        'max_model_fits': max_model_fits,
        'max_llm_calls': 10,
        'max_api_calls': 100,
        'max_wall_clock_seconds': 600,
        'max_retry_attempts': 0,
    }


def _request(**overrides) -> CreateAgentSessionRequest:
    values = {
        'dataset_id': 'dataset',
        'selection_metric': 'macro_f1',
        'allowed_models': ['logistic_regression', 'svm', 'random_forest'],
        'max_runs': 3,
        'modules': {
            'bounded_hpo': True,
            'fail_fast_guard': True,
            'budget_control': True,
        },
        'budget': _limits(),
    }
    values.update(overrides)
    return CreateAgentSessionRequest(**values)


def _budget_context(max_model_fits: int) -> dict[str, object]:
    return {
        'schema_version': 'agent-context-v1',
        'status': 'ready',
        'source_role': 'development',
        'budget_policy': {
            'schema_version': 'agent-budget-policy-v1',
            'limits': _limits(max_model_fits=max_model_fits),
            'model_fit_accounting': 'reservation_then_observation',
        },
    }


def _repository_session(
    tmp_path,
    *,
    max_model_fits: int = 8,
) -> tuple[AgentSessionRepository, Principal, str]:
    repository = AgentSessionRepository(tmp_path / 'agent.sqlite3')
    repository.initialize()
    principal = Principal(owner_id='owner-a', tenant_id='tenant-a')
    session = repository.create_session(
        dataset_id='dataset',
        selection_metric='macro_f1',
        allowed_models=['logistic_regression'],
        max_runs=4,
        seed=42,
        evaluation_config={'split_mode': 'stratified_holdout'},
        principal=principal,
        module_flags={
            'bounded_hpo': True,
            'fail_fast_guard': True,
            'budget_control': True,
        },
        context=_budget_context(max_model_fits),
    )
    return repository, principal, session.session_id


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('max_model_fits', True),
        ('max_llm_calls', '10'),
        ('max_api_calls', 0),
        ('max_wall_clock_seconds', 86401),
        ('max_retry_attempts', -1),
    ],
)
def test_budget_contract_is_strict(field, value):
    budget = _limits()
    budget[field] = value
    with pytest.raises(ValidationError):
        _request(budget=budget)


def test_budget_contract_requires_all_fields_and_forbids_extra():
    missing = _limits()
    missing.pop('max_llm_calls')
    with pytest.raises(ValidationError):
        _request(budget=missing)
    extra = {**_limits(), 'client_reported_model_fits': 1}
    with pytest.raises(ValidationError):
        _request(budget=extra)


@pytest.mark.parametrize(
    ('modules', 'budget', 'message'),
    [
        ({'budget_control': False}, _limits(), '禁止提供'),
        ({'budget_control': True}, None, '需要提供'),
        (
            {'budget_control': True, 'fail_fast_guard': True},
            _limits(),
            'bounded_hpo',
        ),
        (
            {'budget_control': True, 'bounded_hpo': True},
            _limits(),
            'fail_fast_guard',
        ),
    ],
)
def test_budget_business_dependencies_are_locked(modules, budget, message):
    request = _request(modules=modules, budget=budget)
    with pytest.raises(ValueError, match=message):
        request.validate()


@pytest.mark.parametrize(
    ('profile', 'model_type', 'expected'),
    [
        ('off', 'svm', 2),
        ('tiny', 'logistic_regression', 4),
        ('tiny', 'svm', 4),
        ('tiny', 'random_forest', 4),
        ('standard', 'logistic_regression', 4),
        ('standard', 'svm', 6),
        ('standard', 'random_forest', 11),
    ],
)
def test_model_fit_estimate_uses_real_locked_search_space(
    profile, model_type, expected
):
    config = {
        'model_type': model_type,
        'split_mode': 'stratified_holdout',
        'hpo_profile': profile,
        'model_fit_count': 1,  # untrusted caller field must be ignored
        'agent_execution': {'max_hpo_candidates': 18},
    }
    assert estimate_model_fit_upper_bound(config) == expected


def test_model_fit_estimate_rejects_missing_guard_envelope():
    with pytest.raises(AgentBudgetEstimateError):
        estimate_model_fit_upper_bound({
            'model_type': 'svm',
            'split_mode': 'stratified_holdout',
            'hpo_profile': 'tiny',
        })


def test_reservation_bind_and_settlement_are_idempotent(tmp_path):
    repository, principal, session_id = _repository_session(
        tmp_path, max_model_fits=8
    )
    reservation = repository.reserve_experiment(
        session_id=session_id,
        config_hash='config-a',
        action_json={'model_type': 'logistic_regression'},
        principal=principal,
        reserved_model_fits=4,
    )
    assert repository.get_budget_usage_scoped(
        session_id=session_id, principal=principal
    )['model_fits'] == {
        'limit': 8, 'actual': 0, 'reserved': 4, 'charged': 4, 'remaining': 4,
    }
    repository.bind_reservation(
        reservation_id=reservation.reservation_id,
        run_id='run-a',
        parent_run_id=None,
        rationale=None,
        principal=principal,
    )

    def settle():
        return repository.settle_experiment_model_fits_scoped(
            session_id=session_id,
            run_id='run-a',
            actual_model_fits=3,
            principal=principal,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        settled = list(executor.map(lambda _: settle(), range(2)))
    assert all(item.settled and item.actual_model_fits == 3 for item in settled)
    assert repository.get_budget_usage_scoped(
        session_id=session_id, principal=principal
    )['model_fits'] == {
        'limit': 8, 'actual': 3, 'reserved': 0, 'charged': 3, 'remaining': 5,
    }
    with pytest.raises(AgentBudgetAccountingError, match='conflicts'):
        repository.settle_experiment_model_fits_scoped(
            session_id=session_id,
            run_id='run-a',
            actual_model_fits=2,
            principal=principal,
        )


def test_over_reservation_and_over_settlement_leave_accounting_unchanged(tmp_path):
    repository, principal, session_id = _repository_session(
        tmp_path, max_model_fits=3
    )
    with pytest.raises(AgentBudgetExceeded):
        repository.reserve_experiment(
            session_id=session_id,
            config_hash='config-too-large',
            action_json={'model_type': 'logistic_regression'},
            principal=principal,
            reserved_model_fits=4,
        )
    assert repository.count_reservations_scoped(
        session_id=session_id, principal=principal
    ) == 0

    repository2, principal2, session_id2 = _repository_session(
        tmp_path / 'other', max_model_fits=4
    )
    reservation = repository2.reserve_experiment(
        session_id=session_id2,
        config_hash='config-a',
        action_json={'model_type': 'logistic_regression'},
        principal=principal2,
        reserved_model_fits=4,
    )
    repository2.bind_reservation(
        reservation_id=reservation.reservation_id,
        run_id='run-a',
        parent_run_id=None,
        rationale=None,
        principal=principal2,
    )
    with pytest.raises(AgentBudgetAccountingError, match='exceeds'):
        repository2.settle_experiment_model_fits_scoped(
            session_id=session_id2,
            run_id='run-a',
            actual_model_fits=5,
            principal=principal2,
        )
    assert repository2.get_budget_usage_scoped(
        session_id=session_id2, principal=principal2
    )['model_fits']['reserved'] == 4


def test_initialize_migrates_legacy_budget_columns_idempotently(tmp_path):
    database = tmp_path / 'agent.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.executescript(
            '''
            CREATE TABLE agent_sessions (
                session_id TEXT PRIMARY KEY, state TEXT NOT NULL,
                dataset_id TEXT NOT NULL, selection_metric TEXT NOT NULL,
                allowed_models_json TEXT NOT NULL, max_runs INTEGER NOT NULL,
                seed INTEGER NOT NULL, evaluation_config_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE agent_experiments (
                experiment_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                run_id TEXT NOT NULL UNIQUE, attempt INTEGER NOT NULL,
                parent_run_id TEXT, action_json TEXT NOT NULL, rationale TEXT,
                config_hash TEXT NOT NULL, created_at TEXT NOT NULL,
                owner_id TEXT, tenant_id TEXT,
                UNIQUE(session_id, config_hash)
            );
            CREATE TABLE agent_experiment_reservations (
                reservation_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                attempt INTEGER NOT NULL, config_hash TEXT NOT NULL,
                action_json TEXT NOT NULL, created_at TEXT NOT NULL,
                owner_id TEXT, tenant_id TEXT,
                UNIQUE(session_id, config_hash), UNIQUE(session_id, attempt)
            );
            INSERT INTO agent_sessions VALUES (
                'legacy', 'open', 'dataset', 'macro_f1',
                '["logistic_regression"]', 2, 42, '{}', '2026-01-01T00:00:00Z'
            );
            INSERT INTO agent_experiments VALUES (
                'experiment', 'legacy', 'run', 1, NULL, '{}', NULL,
                'config-a', '2026-01-01T00:00:00Z', NULL, NULL
            );
            INSERT INTO agent_experiment_reservations VALUES (
                'reservation', 'legacy', 2, 'config-b', '{}',
                '2026-01-01T00:00:00Z', NULL, NULL
            );
            '''
        )
    repository = AgentSessionRepository(database)
    repository.initialize()
    repository.initialize()
    with sqlite3.connect(database) as connection:
        experiment_columns = {
            row[1] for row in connection.execute(
                'PRAGMA table_info(agent_experiments)'
            )
        }
        reservation_columns = {
            row[1] for row in connection.execute(
                'PRAGMA table_info(agent_experiment_reservations)'
            )
        }
        experiment = connection.execute(
            'SELECT reserved_model_fits, actual_model_fits, settled '
            'FROM agent_experiments WHERE experiment_id = ?',
            ('experiment',),
        ).fetchone()
        reservation = connection.execute(
            'SELECT reserved_model_fits FROM agent_experiment_reservations '
            'WHERE reservation_id = ?',
            ('reservation',),
        ).fetchone()
    assert {'reserved_model_fits', 'actual_model_fits', 'settled'} <= experiment_columns
    assert 'reserved_model_fits' in reservation_columns
    assert experiment == (0, None, 0)
    assert reservation == (0,)


def _build_service(tmp_path):
    principal = Principal(owner_id='owner-a', tenant_id='tenant-a')
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
    return service, sessions, runs, principal, dataset.dataset_id


def test_service_budget_rejection_happens_before_run_creation(tmp_path):
    service, sessions, runs, principal, dataset_id = _build_service(tmp_path)
    request = _request(
        dataset_id=dataset_id,
        budget=_limits(max_model_fits=3),
    )
    session = service.create_session(request, principal=principal)
    with pytest.raises(AgentBudgetExceeded):
        service.create_experiment(
            session_id=session['session_id'],
            payload=CreateAgentExperimentRequest(
                model_type='logistic_regression',
                normalization='zscore',
                class_balance='none',
            ),
            principal=principal,
        )
    assert runs.list_scoped(principal=principal) == []
    assert sessions.list_experiments_scoped(
        session['session_id'], principal=principal
    ) == []
    assert sessions.count_reservations_scoped(
        session_id=session['session_id'], principal=principal
    ) == 0
    usage = sessions.get_budget_usage_scoped(
        session_id=session['session_id'], principal=principal
    )
    assert usage['model_fits']['charged'] == 0


def test_bind_and_cancel_failure_keeps_reservation_charged(tmp_path, monkeypatch):
    service, sessions, runs, principal, dataset_id = _build_service(tmp_path)
    session = service.create_session(
        _request(dataset_id=dataset_id, budget=_limits(max_model_fits=4)),
        principal=principal,
    )

    def fail_bind(**_kwargs):
        raise RuntimeError('bind failed')

    def fail_cancel(*_args, **_kwargs):
        raise RuntimeError('cancel failed')

    monkeypatch.setattr(sessions, 'bind_reservation', fail_bind)
    monkeypatch.setattr(runs, 'cancel_scoped', fail_cancel)
    with pytest.raises(RuntimeError, match='bind failed'):
        service.create_experiment(
            session_id=session['session_id'],
            payload=CreateAgentExperimentRequest(
                model_type='logistic_regression',
                normalization='zscore',
                class_balance='none',
            ),
            principal=principal,
        )
    assert len(runs.list_scoped(principal=principal)) == 1
    assert sessions.count_reservations_scoped(
        session_id=session['session_id'], principal=principal
    ) == 1
    assert sessions.get_budget_usage_scoped(
        session_id=session['session_id'], principal=principal
    )['model_fits']['charged'] == 4


def test_started_cancelled_run_keeps_reservation_charged(tmp_path, monkeypatch):
    service, sessions, runs, principal, dataset_id = _build_service(tmp_path)
    session = service.create_session(
        _request(dataset_id=dataset_id, budget=_limits(max_model_fits=4)),
        principal=principal,
    )

    def fail_bind(**_kwargs):
        raise RuntimeError('bind failed')

    monkeypatch.setattr(sessions, 'bind_reservation', fail_bind)
    monkeypatch.setattr(
        runs,
        'cancel_scoped',
        lambda *_args, **_kwargs: SimpleNamespace(
            state='cancelled',
            started_at='2026-08-31T00:00:00+00:00',
        ),
    )
    with pytest.raises(RuntimeError, match='bind failed'):
        service.create_experiment(
            session_id=session['session_id'],
            payload=CreateAgentExperimentRequest(
                model_type='logistic_regression',
                normalization='zscore',
                class_balance='none',
            ),
            principal=principal,
        )
    assert sessions.count_reservations_scoped(
        session_id=session['session_id'], principal=principal
    ) == 1
    assert sessions.get_budget_usage_scoped(
        session_id=session['session_id'], principal=principal
    )['model_fits']['reserved'] == 4
