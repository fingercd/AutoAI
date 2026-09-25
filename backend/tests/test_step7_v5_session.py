"""The new recipe revision freezes governance with the Session, including off mode."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone

import pytest

from backend.app.agent.budget import BudgetPolicy, DIMENSIONS
from backend.app.runs.repository import RunRepository
from backend.app.model_catalog import MODEL_DECLARATIONS, model_availability
from backend.tests.test_agent_model_sessions import api
from backend.tests.test_finite_search import search_session_request


HEADERS = {'X-AutoAI-Agent-Revision': 'agent-recipes-revision-v5'}


def _request(dataset_id, *, awareness='off'):
    body = search_session_request(dataset_id, ['logistic_regression'])
    body['protocol_revision'] = 'agent-recipes-revision-v5'
    body['client_request_id'] = 'session-budget-v5'
    now = time.time()
    limits = {name: 100 for name in DIMENSIONS}
    limits.update(experiments=1, model_fits=2, training_epochs=0,
                  llm_calls=6, api_calls=12, input_tokens=None,
                  cached_tokens=None, output_tokens=6144)
    body['budget_policy'] = BudgetPolicy(task_id='session-budget-v5',
        started_at=now - 1, work_deadline_at=now + 120,
        deadline_at=now + 180, limits=limits).as_dict()
    body['budget_awareness'] = awareness
    return body


def test_v5_session_freezes_policy_and_off_still_reserves_training(api):
    client, storage, dataset_id = api
    request = _request(dataset_id)
    created = client.post('/api/agent/v2/sessions', headers=HEADERS, json=request)
    assert created.status_code == 201, created.text
    locked = created.json()['locked_config']
    assert locked['budget_awareness'] == 'off'
    digest = BudgetPolicy.from_dict(request['budget_policy']).digest
    assert locked['budget_policy_digest'] == digest
    replay = client.post('/api/agent/v2/sessions', headers=HEADERS, json=request)
    assert replay.status_code == 201, replay.text
    assert replay.json()['session_id'] == created.json()['session_id']
    recipe = locked['preparation']['catalog']['recipes'][0]
    submitted = client.post(
        f"/api/agent/v2/sessions/{created.json()['session_id']}/experiments",
        headers=HEADERS, json={'recipe_id': recipe['recipe_id'],
            'recipe_digest': recipe['recipe_digest'],
            'catalog_digest': locked['preparation']['catalog']['catalog_digest'],
            'knowledge_refs': [], 'client_request_id': 'v5-first-run'})
    assert submitted.status_code == 202, submitted.text
    assert RunRepository(storage / 'runs.sqlite3').get(
        submitted.json()['run_id']).config['execution_budget_policy_digest'] == digest
    repo = RunRepository(storage / 'runs.sqlite3')
    now_utc = datetime.now(timezone.utc)
    repo.record_worker_heartbeat(worker_id='old-search-worker', now=now_utc,
        contract_version='training-worker-search-v1')
    with pytest.raises(sqlite3.IntegrityError, match='budget_worker_contract_required'):
        repo.claim_next(worker_id='old-search-worker', now=now_utc)
    assert repo.get(submitted.json()['run_id']).state == 'queued'
    blocked = client.post(
        f"/api/agent/v2/sessions/{created.json()['session_id']}/terminate",
        headers=HEADERS, json={'client_request_id': 'stop-active',
                               'reason': 'operator_stop'})
    assert blocked.status_code == 409
    with sqlite3.connect(storage / 'agent.sqlite3') as db:
        row = db.execute('''SELECT budget_task_id,budget_policy_digest FROM
            agent_sessions_v1 WHERE session_id=?''',
            (created.json()['session_id'],)).fetchone()
        assert row == ('session-budget-v5', digest)
        amounts = dict(db.execute('''SELECT dimension,amount FROM
            task_budget_reservations_v1 WHERE task_id=? AND owner='backend' ''',
            ('session-budget-v5',)).fetchall())
        assert amounts == {'experiments': 1, 'model_fits': 2, 'training_epochs': 0}


def test_v5_terminate_is_idempotent_and_excludes_future_experiments(api):
    client, _, dataset_id = api
    request = _request(dataset_id, awareness='on')
    created = client.post('/api/agent/v2/sessions', headers=HEADERS, json=request)
    assert created.status_code == 201, created.text
    session_id = created.json()['session_id']
    path = f'/api/agent/v2/sessions/{session_id}/terminate'
    payload = {'client_request_id': 'stop-v5', 'reason': 'no_candidates'}
    first = client.post(path, headers=HEADERS, json=payload)
    assert first.status_code == 200, first.text
    assert first.json()['state'] == 'terminated'
    assert first.json()['selected_run_id'] is None
    replay = client.post(path, headers=HEADERS, json=payload)
    assert replay.status_code == 200, replay.text
    assert replay.json() == first.json()
    conflict = client.post(path, headers=HEADERS,
        json={**payload, 'reason': 'budget_exhausted'})
    assert conflict.status_code == 409
    locked = created.json()['locked_config']
    recipe = locked['preparation']['catalog']['recipes'][0]
    submitted = client.post(f'/api/agent/v2/sessions/{session_id}/experiments',
        headers=HEADERS, json={'recipe_id': recipe['recipe_id'],
            'recipe_digest': recipe['recipe_digest'],
            'catalog_digest': locked['preparation']['catalog']['catalog_digest'],
            'knowledge_refs': [], 'client_request_id': 'after-stop'})
    assert submitted.status_code == 409


def test_v5_all_executable_models_have_finite_training_upper_bound(api):
    from backend.app.agent.budget import training_upper_bound
    client, _, dataset = api
    models = [item.id for item in MODEL_DECLARATIONS
              if model_availability(item.id)[0]]
    assert len(models) == 13
    request = _request(dataset)
    request['allowed_models'] = models
    created = client.post('/api/agent/v2/sessions', headers=HEADERS, json=request)
    assert created.status_code == 201, created.text
    locked = created.json()['locked_config']
    preparation = locked['preparation']
    recipes = preparation['catalog']['recipes']
    assert {item['model_id'] for item in recipes} == set(models)
    for recipe in recipes:
        plan = preparation['search_plans'][recipe['search_plan_digest']]
        config = locked['capability_snapshot']['model_configs'][recipe['model_id']]
        cost = training_upper_bound(plan,
            epochs=config.get('epochs') if
                plan['fit_strategy']=='train_best_epoch_no_refit' else None)
        assert cost['model_fits'] >= 1
        assert cost['training_epochs'] >= 0
