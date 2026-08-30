from __future__ import annotations

import sqlite3

import pytest

from backend.app.agent.repository import AgentSessionNotFound, AgentSessionRepository
from backend.app.runs.contracts import Principal


def test_initialize_migrates_legacy_session_table(tmp_path):
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
        connection.execute(
            '''
            INSERT INTO agent_sessions (
                session_id, state, dataset_id, selection_metric,
                allowed_models_json, max_runs, seed,
                evaluation_config_json, created_at
            ) VALUES (
                'legacy-session', 'open', 'dataset-legacy', 'macro_f1',
                '["logistic_regression"]', 1, 42, '{}', '2026-01-01T00:00:00Z'
            )
            '''
        )
    repository = AgentSessionRepository(database)
    repository.initialize()
    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute('PRAGMA table_info(agent_sessions)').fetchall()
        }
    assert {'module_flags_json', 'context_policy_json', 'context_json'} <= columns
    legacy = repository.get_session_scoped(
        'legacy-session', principal=Principal()
    )
    assert not any(legacy.module_flags.values())
    assert legacy.context_policy == {
        'source_role': 'development',
        'case_write': False,
    }
    assert legacy.context == {
        'schema_version': 'agent-context-v1',
        'status': 'disabled',
        'source_role': 'development',
    }


def test_context_snapshot_update_is_principal_scoped(tmp_path):
    repository = AgentSessionRepository(tmp_path / 'agent.sqlite3')
    repository.initialize()
    principal = Principal(owner_id='owner-a', tenant_id='tenant-a')
    session = repository.create_session(
        dataset_id='dataset-1',
        selection_metric='macro_f1',
        allowed_models=['logistic_regression'],
        max_runs=2,
        seed=42,
        evaluation_config={'split_mode': 'stratified_holdout'},
        principal=principal,
        module_flags={'evidence_card': True},
        context_policy={'source_role': 'domain', 'case_write': True},
        context={'schema_version': 'agent-context-v1', 'status': 'pending'},
    )
    updated = repository.update_context_scoped(
        session.session_id,
        context={'schema_version': 'agent-context-v1', 'status': 'ready'},
        principal=principal,
    )
    assert updated.context['status'] == 'ready'
    assert updated.module_flags['evidence_card'] is True
    assert sum(updated.module_flags.values()) == 1
    with pytest.raises(AgentSessionNotFound):
        repository.update_context_scoped(
            session.session_id,
            context={'schema_version': 'agent-context-v1', 'status': 'tampered'},
            principal=Principal(owner_id='owner-b', tenant_id='tenant-a'),
        )
