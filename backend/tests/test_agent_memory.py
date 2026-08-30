from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.app.agent.contracts import (
    AgentContextPolicy,
    AgentModuleFlags,
    CreateAgentSessionRequest,
)
from backend.app.agent.memory import (
    MAX_MEMORY_CONTEXT_ITEMS,
    build_memory_record,
    build_memory_signatures,
    build_terminal_outcome,
    freeze_memory_context,
)
from backend.app.agent.repository import (
    AgentMemoryCollision,
    AgentMemoryWriteDenied,
    AgentSessionClosed,
    AgentSessionRepository,
)
from backend.app.runs.contracts import Principal


def _flags() -> dict[str, bool]:
    return {
        'evidence_card': True,
        'fail_fast_guard': True,
        'feedback_diagnosis': True,
        'case_memory': True,
    }


def _create_repository_session(
    repository: AgentSessionRepository,
    principal: Principal,
    *,
    source_role: str = 'domain',
    case_write: bool = True,
    module_flags: dict[str, bool] | None = None,
):
    return repository.create_session(
        dataset_id='internal-dataset-id',
        selection_metric='macro_f1',
        allowed_models=['logistic_regression'],
        max_runs=3,
        seed=42,
        evaluation_config={
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        },
        principal=principal,
        module_flags=module_flags if module_flags is not None else _flags(),
        context_policy={
            'source_role': source_role,
            'case_write': case_write,
        },
        context={'schema_version': 'agent-context-v1', 'status': 'ready'},
    )


def _create_repository_experiment(
    repository: AgentSessionRepository,
    *,
    session_id: str,
    run_id: str,
    principal: Principal,
    attempt: int = 1,
):
    return repository.create_experiment(
        session_id=session_id,
        run_id=run_id,
        attempt=attempt,
        parent_run_id=None,
        action_json={
            'model_type': 'logistic_regression',
            'normalization': 'zscore',
            'class_balance': 'none',
        },
        rationale='raw rationale /users/private/data.csv must never persist',
        config_hash=f'config-{attempt}',
        principal=principal,
    )


def _signatures() -> tuple[str, str]:
    return build_memory_signatures(
        selection_metric='macro_f1',
        allowed_models=['logistic_regression'],
        evaluation_config={
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        },
        evidence_card={
            'scope': 'train_only',
            'data_format': 'wide-feature-v2',
            'statistics': {
                'observation_count': 48,
                'predictor_count': 12,
                'class_distribution': {'class_0': {}, 'class_1': {}},
                'repeat_measurements': {'uniform': True},
                'risk_codes': ['very_small_train_partition'],
            },
        },
    )


def _failed_outcome() -> dict:
    payload = build_terminal_outcome(
        state='failed',
        selection_metric='macro_f1',
        action={
            'model_type': 'logistic_regression',
            'normalization': 'zscore',
            'class_balance': 'none',
            'rationale': 'Bearer secret /users/private/data.csv',
        },
        diagnosis={
            'status': 'ready',
            'primary_category': 'training',
            'symptom_codes': ['failed_run'],
            'allowed_action_ids': ['choose_unused_proposal', 'stop'],
        },
        duration_seconds=1.25,
        model_fit_count=2,
    )
    # Repository normalization must discard unknown keys even if a caller is
    # compromised after the service builder returned.
    payload['raw_error'] = 'Bearer secret /users/private/data.csv'
    payload['run_id'] = 'historical-run-id'
    return payload


def test_case_memory_contract_dependencies_are_strict():
    base = {
        'dataset_id': 'dataset',
        'selection_metric': 'macro_f1',
        'allowed_models': ['logistic_regression'],
        'max_runs': 1,
    }
    with pytest.raises(ValueError, match='case_memory'):
        CreateAgentSessionRequest(
            **base,
            modules=AgentModuleFlags(case_memory=True),
        ).validate()
    with pytest.raises(ValueError, match='case_write'):
        CreateAgentSessionRequest(
            **base,
            context_policy=AgentContextPolicy(case_write=True),
        ).validate()
    with pytest.raises(ValueError, match='benchmark'):
        CreateAgentSessionRequest(
            **base,
            modules=AgentModuleFlags(
                evidence_card=True,
                fail_fast_guard=True,
                feedback_diagnosis=True,
                case_memory=True,
            ),
            context_policy=AgentContextPolicy(
                source_role='benchmark', case_write=True
            ),
        ).validate()


def test_initialize_adds_memory_tables_to_legacy_database(tmp_path):
    database = tmp_path / 'agent.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.execute(
            '''
            CREATE TABLE agent_sessions (
                session_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                selection_metric TEXT NOT NULL,
                allowed_models_json TEXT NOT NULL,
                max_runs INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                evaluation_config_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            '''
        )
    repository = AgentSessionRepository(database)
    repository.initialize()
    repository.initialize()
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {'agent_terminal_outcomes', 'agent_memory_records'} <= tables


@pytest.mark.parametrize(
    'source_role,case_write,module_flags',
    [
        ('benchmark', True, _flags()),
        ('domain', False, _flags()),
        (
            'domain',
            True,
            {
                'evidence_card': True,
                'feedback_diagnosis': True,
                'case_memory': True,
            },
        ),
    ],
)
def test_repository_rechecks_memory_write_policy(
    tmp_path, source_role, case_write, module_flags
):
    repository = AgentSessionRepository(tmp_path / 'agent.sqlite3')
    repository.initialize()
    principal = Principal(owner_id='owner', tenant_id='tenant')
    session = _create_repository_session(
        repository,
        principal,
        source_role=source_role,
        case_write=case_write,
        module_flags=module_flags,
    )
    _create_repository_experiment(
        repository,
        session_id=session.session_id,
        run_id='run-denied',
        principal=principal,
    )
    task_signature, evidence_signature = _signatures()
    with pytest.raises(AgentMemoryWriteDenied):
        repository.record_terminal_outcome_scoped(
            session_id=session.session_id,
            run_id='run-denied',
            state='failed',
            task_signature=task_signature,
            evidence_signature=evidence_signature,
            payload=_failed_outcome(),
            failure_memory_payload=build_memory_record(
                'failure', _failed_outcome()
            ),
            principal=principal,
        )


def test_concurrent_terminal_feedback_is_idempotent_and_principal_scoped(tmp_path):
    database = tmp_path / 'agent.sqlite3'
    repository = AgentSessionRepository(database)
    repository.initialize()
    principal = Principal(owner_id='owner-a', tenant_id='tenant-a')
    session = _create_repository_session(repository, principal)
    _create_repository_experiment(
        repository,
        session_id=session.session_id,
        run_id='run-failed',
        principal=principal,
    )
    task_signature, evidence_signature = _signatures()
    outcome = _failed_outcome()
    failure = build_memory_record('failure', outcome)

    def write_once(_index: int):
        return repository.record_terminal_outcome_scoped(
            session_id=session.session_id,
            run_id='run-failed',
            state='failed',
            task_signature=task_signature,
            evidence_signature=evidence_signature,
            payload=outcome,
            failure_memory_payload=failure,
            principal=principal,
        ).outcome_id

    with ThreadPoolExecutor(max_workers=6) as executor:
        outcome_ids = list(executor.map(write_once, range(12)))
    assert len(set(outcome_ids)) == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            'SELECT COUNT(*) FROM agent_terminal_outcomes'
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM agent_memory_records WHERE record_type = 'failure'"
        ).fetchone()[0] == 1
        encoded = connection.execute(
            'SELECT payload_json FROM agent_terminal_outcomes'
        ).fetchone()[0].lower()
    for forbidden in (
        'raw_error', 'rationale', 'historical-run-id', '/users/', 'bearer secret'
    ):
        assert forbidden not in encoded

    visible = repository.list_memory_payloads_scoped(
        task_signature=task_signature,
        evidence_signature=evidence_signature,
        principal=principal,
    )
    assert len(visible) == 1
    assert visible[0]['record_type'] == 'failure'
    assert repository.list_memory_payloads_scoped(
        task_signature=task_signature,
        evidence_signature=evidence_signature,
        principal=Principal(owner_id='owner-b', tenant_id='tenant-a'),
    ) == []


def test_existing_terminal_outcome_rejects_canonical_payload_change(tmp_path):
    repository = AgentSessionRepository(tmp_path / 'agent.sqlite3')
    repository.initialize()
    principal = Principal(owner_id='owner', tenant_id='tenant')
    session = _create_repository_session(repository, principal)
    experiment = _create_repository_experiment(
        repository,
        session_id=session.session_id,
        run_id='run-collision',
        principal=principal,
    )
    task_signature, evidence_signature = _signatures()

    def payload(score: float) -> dict:
        return build_terminal_outcome(
            state='succeeded',
            selection_metric='macro_f1',
            action=experiment.action_json,
            validation_score=score,
            diagnosis={'status': 'ready', 'primary_category': 'none'},
        )

    repository.record_terminal_outcome_scoped(
        session_id=session.session_id,
        run_id='run-collision',
        state='succeeded',
        task_signature=task_signature,
        evidence_signature=evidence_signature,
        payload=payload(0.91),
        principal=principal,
    )
    with pytest.raises(AgentMemoryCollision):
        repository.record_terminal_outcome_scoped(
            session_id=session.session_id,
            run_id='run-collision',
            state='succeeded',
            task_signature=task_signature,
            evidence_signature=evidence_signature,
            payload=payload(0.72),
            principal=principal,
        )


def test_missing_validation_score_cannot_persist_or_finalize_memory_session(tmp_path):
    database = tmp_path / 'agent.sqlite3'
    repository = AgentSessionRepository(database)
    repository.initialize()
    principal = Principal(owner_id='owner', tenant_id='tenant')
    session = _create_repository_session(repository, principal)
    experiment = _create_repository_experiment(
        repository,
        session_id=session.session_id,
        run_id='run-no-validation',
        principal=principal,
    )
    task_signature, evidence_signature = _signatures()
    no_score = build_terminal_outcome(
        state='succeeded',
        selection_metric='macro_f1',
        action=experiment.action_json,
        diagnosis={'status': 'insufficient_evidence', 'primary_category': 'none'},
    )
    with pytest.raises(AgentMemoryWriteDenied, match='validation_score'):
        repository.record_terminal_outcome_scoped(
            session_id=session.session_id,
            run_id='run-no-validation',
            state='succeeded',
            task_signature=task_signature,
            evidence_signature=evidence_signature,
            payload=no_score,
            principal=principal,
        )
    with pytest.raises(AgentMemoryWriteDenied):
        repository.finalize_session(
            session_id=session.session_id,
            selected_run_id='run-no-validation',
            principal=principal,
        )
    assert repository.get_session_scoped(
        session.session_id, principal=principal
    ).state == 'open'
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            'SELECT COUNT(*) FROM agent_terminal_outcomes'
        ).fetchone()[0] == 0
        assert connection.execute(
            'SELECT COUNT(*) FROM agent_memory_records'
        ).fetchone()[0] == 0


def test_memory_disabled_finalize_keeps_legacy_behavior(tmp_path):
    repository = AgentSessionRepository(tmp_path / 'agent.sqlite3')
    repository.initialize()
    principal = Principal(owner_id='owner', tenant_id='tenant')
    session = _create_repository_session(
        repository,
        principal,
        source_role='development',
        case_write=False,
        module_flags={},
    )
    _create_repository_experiment(
        repository,
        session_id=session.session_id,
        run_id='legacy-run-without-outcome',
        principal=principal,
    )
    finalized = repository.finalize_session(
        session_id=session.session_id,
        selected_run_id='legacy-run-without-outcome',
        principal=principal,
    )
    assert finalized.state == 'finalized'
    assert finalized.selected_run_id == 'legacy-run-without-outcome'


def test_atomic_finalize_rejects_tampered_outcome_without_publishing_case(tmp_path):
    database = tmp_path / 'agent.sqlite3'
    repository = AgentSessionRepository(database)
    repository.initialize()
    principal = Principal(owner_id='owner', tenant_id='tenant')
    session = _create_repository_session(repository, principal)
    experiment = _create_repository_experiment(
        repository,
        session_id=session.session_id,
        run_id='run-tampered',
        principal=principal,
    )
    task_signature, evidence_signature = _signatures()
    outcome = build_terminal_outcome(
        state='succeeded',
        selection_metric='macro_f1',
        action=experiment.action_json,
        validation_score=0.9,
        diagnosis={'status': 'ready', 'primary_category': 'none'},
    )
    repository.record_terminal_outcome_scoped(
        session_id=session.session_id,
        run_id='run-tampered',
        state='succeeded',
        task_signature=task_signature,
        evidence_signature=evidence_signature,
        payload=outcome,
        principal=principal,
    )
    with sqlite3.connect(database) as connection:
        stored = json.loads(connection.execute(
            'SELECT payload_json FROM agent_terminal_outcomes'
        ).fetchone()[0])
        stored['raw_error'] = 'Bearer secret /users/private/data.csv'
        connection.execute(
            'UPDATE agent_terminal_outcomes SET payload_json = ?',
            (json.dumps(stored),),
        )
    with pytest.raises(AgentMemoryCollision):
        repository.finalize_session(
            session_id=session.session_id,
            selected_run_id='run-tampered',
            principal=principal,
        )
    assert repository.get_session_scoped(
        session.session_id, principal=principal
    ).state == 'open'
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM agent_memory_records WHERE record_type = 'case'"
        ).fetchone()[0] == 0


def test_finalize_persists_only_selected_case_and_is_idempotent(tmp_path):
    database = tmp_path / 'agent.sqlite3'
    repository = AgentSessionRepository(database)
    repository.initialize()
    principal = Principal(owner_id='owner', tenant_id='tenant')
    session = _create_repository_session(repository, principal)
    task_signature, evidence_signature = _signatures()
    for attempt, run_id, score in (
        (1, 'run-selected', 0.91),
        (2, 'run-not-selected', 0.89),
    ):
        experiment = _create_repository_experiment(
            repository,
            session_id=session.session_id,
            run_id=run_id,
            principal=principal,
            attempt=attempt,
        )
        payload = build_terminal_outcome(
            state='succeeded',
            selection_metric='macro_f1',
            action=experiment.action_json,
            validation_score=score,
            diagnosis={
                'status': 'ready',
                'primary_category': 'none',
                'symptom_codes': [],
                'allowed_action_ids': ['finalize'],
                'generalization_gap': 0.02,
            },
        )
        repository.record_terminal_outcome_scoped(
            session_id=session.session_id,
            run_id=run_id,
            state='succeeded',
            task_signature=task_signature,
            evidence_signature=evidence_signature,
            payload=payload,
            principal=principal,
        )
    first = repository.finalize_session(
        session_id=session.session_id,
        selected_run_id='run-selected',
        principal=principal,
    )
    second = repository.finalize_session(
        session_id=session.session_id,
        selected_run_id='run-selected',
        principal=principal,
    )
    repaired = repository.record_selected_case_scoped(
        session_id=session.session_id,
        run_id='run-selected',
        principal=principal,
    )
    assert first.state == second.state == 'finalized'
    assert repaired.record_type == 'case'
    with pytest.raises(AgentMemoryWriteDenied):
        repository.record_selected_case_scoped(
            session_id=session.session_id,
            run_id='run-not-selected',
            principal=principal,
        )
    with pytest.raises(AgentSessionClosed):
        repository.finalize_session(
            session_id=session.session_id,
            selected_run_id='run-not-selected',
            principal=principal,
        )
    with sqlite3.connect(database) as connection:
        cases = connection.execute(
            "SELECT COUNT(*) FROM agent_memory_records WHERE record_type = 'case'"
        ).fetchone()[0]
        selected = connection.execute(
            'SELECT SUM(selected) FROM agent_terminal_outcomes'
        ).fetchone()[0]
    assert cases == 1
    assert selected == 1


def test_frozen_memory_context_is_bounded_and_contains_no_historical_ids():
    records = []
    for score in (0.91, 0.90, 0.89, 0.88, 0.87):
        outcome = build_terminal_outcome(
            state='succeeded',
            selection_metric='macro_f1',
            action={'model_type': 'logistic_regression'},
            validation_score=score,
            diagnosis={'status': 'ready', 'primary_category': 'none'},
        )
        record = build_memory_record('case', outcome)
        record['dataset_id'] = 'historical-dataset'
        record['session_id'] = 'historical-session'
        record['run_id'] = 'historical-run'
        records.append(record)
    context = freeze_memory_context(records, max_items=99)
    assert len(context['entries']) == MAX_MEMORY_CONTEXT_ITEMS
    flat = json.dumps(context, ensure_ascii=False).lower()
    for forbidden in (
        'dataset_id', 'session_id', 'run_id', 'historical-', '/users/', 'rationale'
    ):
        assert forbidden not in flat


def test_service_freezes_selected_case_without_historical_ids(tmp_path):
    """HTTP regression: feedback/finalize write once; next Session freezes it."""
    from fastapi.testclient import TestClient

    from backend.app.main import app
    from backend.app.paths import AGENT_DATABASE, RUNS_DATABASE, RUNS_DIR
    from backend.app.runs.repository import RunRepository
    from backend.tests.modeling_data_factory import write_grouped_classification_csv
    from backend.tests.test_agent_sessions import _claim_and_finish

    client = TestClient(app)
    source = tmp_path / 'memory.csv'
    write_grouped_classification_csv(
        source, groups_per_class=8, repeats=2, feature_count=10
    )
    with source.open('rb') as handle:
        upload = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        )
    assert upload.status_code == 200, upload.text
    dataset_id = upload.json()['dataset_id']
    modules = {
        'evidence_card': True,
        'fail_fast_guard': True,
        'feedback_diagnosis': True,
        'case_memory': True,
    }
    session_body = {
        'dataset_id': dataset_id,
        'selection_metric': 'macro_f1',
        'allowed_models': ['logistic_regression'],
        'max_runs': 1,
        'seed': 42,
        'evaluation': {
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        },
        'modules': modules,
        'context_policy': {'source_role': 'domain', 'case_write': True},
    }
    writer = client.post('/api/agent/sessions', json=session_body)
    assert writer.status_code == 201, writer.text
    writer_id = writer.json()['session_id']
    experiment = client.post(
        f'/api/agent/sessions/{writer_id}/experiments',
        json={'model_type': 'logistic_regression', 'rationale': '/users/private'},
    )
    assert experiment.status_code == 202, experiment.text
    run_id = experiment.json()['run_id']
    runs = RunRepository(RUNS_DATABASE)
    runs.initialize()
    _claim_and_finish(runs, run_id)

    feedback_url = (
        f'/api/agent/sessions/{writer_id}/experiments/{run_id}/feedback'
    )
    assert client.get(feedback_url).status_code == 200
    with sqlite3.connect(AGENT_DATABASE) as connection:
        assert connection.execute(
            'SELECT COUNT(*) FROM agent_terminal_outcomes WHERE session_id = ?',
            (writer_id,),
        ).fetchone()[0] == 0
    finalize_url = f'/api/agent/sessions/{writer_id}/finalize'
    rejected = client.post(finalize_url, json={'selected_run_id': run_id})
    assert rejected.status_code == 422, rejected.text
    assert client.get(f'/api/agent/sessions/{writer_id}').json()['state'] == 'open'

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'metrics.json').write_text(
        json.dumps({
            'train': {'macro_f1': 0.94},
            'valid': {'macro_f1': 0.91},
        }),
        encoding='utf-8',
    )
    assert client.get(feedback_url).status_code == 200
    assert client.post(
        finalize_url, json={'selected_run_id': run_id}
    ).status_code == 200
    assert client.post(
        finalize_url, json={'selected_run_id': run_id}
    ).status_code == 200

    failed_writer = client.post('/api/agent/sessions', json=session_body)
    assert failed_writer.status_code == 201, failed_writer.text
    failed_writer_id = failed_writer.json()['session_id']
    failed_experiment = client.post(
        f'/api/agent/sessions/{failed_writer_id}/experiments',
        json={'model_type': 'logistic_regression', 'rationale': 'Bearer secret'},
    )
    assert failed_experiment.status_code == 202, failed_experiment.text
    failed_run_id = failed_experiment.json()['run_id']
    with runs._connection() as connection:  # type: ignore[attr-defined]
        connection.execute('BEGIN IMMEDIATE')
        connection.execute(
            '''
            UPDATE runs
            SET state = 'failed', version = version + 1,
                error = ?, error_json = ?, finished_at = updated_at
            WHERE run_id = ? AND state = 'queued'
            ''',
            (
                'failure at /users/private/data.csv',
                json.dumps({
                    'code': 'training_failed',
                    'message': 'Bearer secret /users/private/data.csv',
                }),
                failed_run_id,
            ),
        )
        connection.commit()
    failed_feedback_url = (
        f'/api/agent/sessions/{failed_writer_id}/experiments/'
        f'{failed_run_id}/feedback'
    )
    assert client.get(failed_feedback_url).status_code == 200
    assert client.get(failed_feedback_url).status_code == 200

    reader_body = {
        **session_body,
        'context_policy': {'source_role': 'domain', 'case_write': False},
    }
    reader = client.post('/api/agent/sessions', json=reader_body)
    assert reader.status_code == 201, reader.text
    memory_context = reader.json()['context']['memory_context']
    assert 2 <= len(memory_context['entries']) <= MAX_MEMORY_CONTEXT_ITEMS
    assert {'case', 'failure'} <= {
        entry['record_type'] for entry in memory_context['entries']
    }
    assert any(
        entry['record_type'] == 'case'
        and entry['effective_action']['model_type'] == 'logistic_regression'
        for entry in memory_context['entries']
    )
    flat = json.dumps(memory_context, ensure_ascii=False).lower()
    for forbidden in (
        dataset_id.lower(), writer_id.lower(), run_id.lower(),
        failed_writer_id.lower(), failed_run_id.lower(),
        'dataset_id', 'session_id', 'run_id', '/users/', 'rationale', 'raw_error',
    ):
        assert forbidden not in flat
