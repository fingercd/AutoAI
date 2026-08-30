from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app.agent.repository import AgentConfigCollision, AgentSessionRepository
from backend.app.main import app
from backend.app.paths import AGENT_DATABASE, RUNS_DATABASE
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunRepository, _timestamp
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def _set_terminal(repo, run_id, state, *, error_details=None):
    now = _timestamp(datetime.now(timezone.utc))
    with repo._connection() as connection:  # type: ignore[attr-defined]
        connection.execute('BEGIN IMMEDIATE')
        connection.execute(
            '''
            UPDATE runs SET state = ?, version = version + 1,
                error = ?, error_json = ?, updated_at = ?, finished_at = ?
            WHERE run_id = ?
            ''',
            (
                state,
                'controlled failure' if state == 'failed' else None,
                json.dumps(error_details or {}),
                now,
                now,
                run_id,
            ),
        )
        connection.commit()


def test_backend_replan_requires_failed_parent_diagnosis_and_unused_recipe(tmp_path):
    client = TestClient(app)
    source = tmp_path / 'replan.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    with source.open('rb') as handle:
        dataset_id = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        ).json()['dataset_id']
    created = client.post(
        '/api/agent/sessions',
        json={
            'dataset_id': dataset_id,
            'selection_metric': 'macro_f1',
            'allowed_models': [
                'logistic_regression', 'svm', 'random_forest',
            ],
            'max_runs': 3,
            'seed': 42,
            'evaluation': {
                'split_mode': 'stratified_holdout',
                'split_train': 8,
                'split_valid': 1,
                'split_test': 1,
            },
            'modules': {
                'restricted_strategy_pool': True,
                'fail_fast_guard': True,
                'feedback_diagnosis': True,
                'limited_replanning': True,
            },
        },
    )
    assert created.status_code == 201, created.text
    session = created.json()
    assert session['context']['status'] == 'ready'
    assert session['context']['replanning_policy']['max_retries_per_failure'] == 1
    assert session['context']['replanning_policy']['max_replan_depth'] == 1
    proposals = session['context']['proposal_catalog']['proposals']
    first = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={**proposals[0], 'rationale': 'baseline'},
    )
    assert first.status_code == 202, first.text
    repo = RunRepository(RUNS_DATABASE)
    repo.initialize()
    _set_terminal(
        repo,
        first.json()['run_id'],
        'failed',
        error_details={'code': 'training_failed', 'stage': 'training'},
    )

    sessions = AgentSessionRepository(
        AGENT_DATABASE,
        runs_database_path=RUNS_DATABASE,
    )
    sessions.initialize()
    with pytest.raises(AgentConfigCollision, match='未处理失败'):
        sessions.reserve_experiment(
            session_id=session['session_id'],
            config_hash='stale-actionless-request',
            action_json={'model_type': 'svm'},
            principal=Principal(),
            require_replan_for_failed=True,
        )

    bypass = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={**proposals[1], 'rationale': 'bypass'},
    )
    assert bypass.status_code == 422

    def submit(recipe):
        with TestClient(app) as concurrent_client:
            response = concurrent_client.post(
                f"/api/agent/sessions/{session['session_id']}/experiments",
                json={
                    **recipe,
                    'parent_run_id': first.json()['run_id'],
                    'action_id': 'choose_unused_proposal',
                    'rationale': 'retry',
                },
            )
            return response.status_code, response.json()

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = list(executor.map(submit, proposals[1:3]))
    assert sorted(status for status, _ in attempts) == [202, 409]
    replanned_payload = next(payload for status, payload in attempts if status == 202)
    accepted_proposal_id = replanned_payload['effective_action']['proposal_id']
    unused_recipe = next(
        item
        for item in proposals[1:3]
        if item['proposal_id'] != accepted_proposal_id
    )

    missing_parent = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={
            **unused_recipe,
            'action_id': 'choose_unused_proposal',
            'rationale': 'retry',
        },
    )
    assert missing_parent.status_code == 422

    _set_terminal(
        repo,
        replanned_payload['run_id'],
        'failed',
        error_details={'code': 'training_failed', 'stage': 'training'},
    )
    too_deep = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={
            **unused_recipe,
            'parent_run_id': replanned_payload['run_id'],
            'action_id': 'choose_unused_proposal',
            'rationale': 'retry',
        },
    )
    assert too_deep.status_code == 422

    _set_terminal(repo, replanned_payload['run_id'], 'succeeded')
    successful_parent = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={
            **unused_recipe,
            'parent_run_id': replanned_payload['run_id'],
            'action_id': 'choose_unused_proposal',
            'rationale': 'retry',
        },
    )
    assert successful_parent.status_code == 422


def test_legacy_parent_lineage_remains_accepted_without_replanning_module(tmp_path):
    client = TestClient(app)
    source = tmp_path / 'legacy-lineage.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    with source.open('rb') as handle:
        dataset_id = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        ).json()['dataset_id']
    session = client.post(
        '/api/agent/sessions',
        json={
            'dataset_id': dataset_id,
            'selection_metric': 'macro_f1',
            'allowed_models': ['logistic_regression', 'svm'],
            'max_runs': 2,
            'seed': 42,
            'evaluation': {
                'split_mode': 'stratified_holdout',
                'split_train': 8,
                'split_valid': 1,
                'split_test': 1,
            },
        },
    ).json()
    first = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={'model_type': 'logistic_regression'},
    ).json()
    repo = RunRepository(RUNS_DATABASE)
    repo.initialize()
    _set_terminal(repo, first['run_id'], 'failed')
    child = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={'model_type': 'svm', 'parent_run_id': first['run_id']},
    )
    assert child.status_code == 202, child.text
