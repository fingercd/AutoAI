"""Durable budget invariants, including real SQLite process contention."""
from __future__ import annotations

import multiprocessing
from pathlib import Path
import sqlite3
import time

import pytest

from backend.app.agent.budget import (
    BudgetError, BudgetPolicy, DIMENSIONS, dimension_summary, freeze_policy,
    initialize_ledger, reserve, training_upper_bound, transition,
)
from backend.app.agent.repository import AgentSessionRepository
from backend.app.runs.contracts import Principal


def policy(*, task_id='task-1', fits=2, calls=12):
    limits = {name: 100 for name in DIMENSIONS}
    limits.update(experiments=1, model_fits=fits, training_epochs=12,
                  llm_calls=6, api_calls=calls, input_tokens=None,
                  cached_tokens=None)
    return BudgetPolicy(task_id, 100.0, 200.0, 140.0, limits)


def open_ledger(path: Path, policy_value: BudgetPolicy, owner='backend'):
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    db.execute('PRAGMA busy_timeout=10000')
    db.execute('BEGIN IMMEDIATE')
    initialize_ledger(db)
    freeze_policy(db, policy_value, owner=owner)
    db.commit()
    return db


def reserve_fit(db, operation, amount=1):
    db.execute('BEGIN IMMEDIATE')
    try:
        reserve(db, task_id='task-1', owner='backend', operation_id=operation,
                attempt_id='0', amounts={'model_fits': amount},
                payload={'plan': 'fixed'}, now=101.0)
        db.commit()
        return 'reserved'
    except BudgetError as error:
        db.rollback()
        return error.code


def contender(path: str, start, result, operation: str):
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    db.execute('PRAGMA busy_timeout=10000')
    start.wait(10)
    result.put(reserve_fit(db, operation))
    db.close()


def test_exact_limit_idempotence_conflict_and_unknown_hold(tmp_path):
    db = open_ledger(tmp_path / 'budget.sqlite', policy())
    assert reserve_fit(db, 'first', 2) == 'reserved'
    assert reserve_fit(db, 'first', 2) == 'reserved'
    assert reserve_fit(db, 'second') == 'budget_insufficient'
    assert dimension_summary(db, task_id='task-1', owner='backend', dimension='model_fits')['remaining'] == 0
    db.execute('BEGIN IMMEDIATE')
    with pytest.raises(BudgetError, match='budget_reservation_conflict'):
        reserve(db, task_id='task-1', owner='backend', operation_id='first',
                attempt_id='0', amounts={'model_fits': 1}, payload={'plan': 'fixed'})
    db.rollback()
    db.execute('BEGIN IMMEDIATE')
    transition(db, task_id='task-1', owner='backend', operation_id='first', attempt_id='0',
               dimension='model_fits', status='dispatched')
    transition(db, task_id='task-1', owner='backend', operation_id='first', attempt_id='0',
               dimension='model_fits', status='unknown_pending', actual=1, held=1)
    db.commit()
    summary = dimension_summary(db, task_id='task-1', owner='backend', dimension='model_fits')
    assert (summary['known_actual'], summary['held_unknown'], summary['remaining']) == (1, 1, 0)
    assert reserve_fit(db, 'second') == 'budget_insufficient'
    db.execute('BEGIN IMMEDIATE')
    transition(db, task_id='task-1', owner='backend', operation_id='first', attempt_id='0',
               dimension='model_fits', status='settled', actual=1, source_ref='worker-event-1')
    db.commit()
    assert reserve_fit(db, 'second') == 'reserved'
    db.close()


def test_two_spawned_processes_compete_for_last_fit(tmp_path):
    path = tmp_path / 'budget.sqlite'
    open_ledger(path, policy(fits=1)).close()
    ctx = multiprocessing.get_context('spawn')
    start, result = ctx.Event(), ctx.Queue()
    processes = [ctx.Process(target=contender, args=(str(path), start, result, f'op-{i}'))
                 for i in range(2)]
    for process in processes:
        process.start()
    start.set()
    outcomes = sorted(result.get(timeout=15) for _ in processes)
    for process in processes:
        process.join(15)
        assert process.exitcode == 0
    assert outcomes == ['budget_insufficient', 'reserved']


def test_policy_validation_tampering_owner_and_tail(tmp_path):
    with pytest.raises(BudgetError, match='budget_invalid_integer'):
        policy(fits=True)
    with pytest.raises(BudgetError, match='budget_invalid_deadline'):
        BudgetPolicy('task-1', 100.0, float('nan'), 140.0, policy().limits)
    with pytest.raises(BudgetError, match='budget_finalization_insufficient'):
        policy(calls=8)
    db = open_ledger(tmp_path / 'budget.sqlite', policy())
    db.execute('BEGIN IMMEDIATE')
    with pytest.raises(BudgetError, match='budget_dimension_owner_mismatch'):
        reserve(db, task_id='task-1', owner='backend', operation_id='x', attempt_id='0',
                amounts={'llm_calls': 1}, payload='x')
    db.rollback()
    db.execute("UPDATE task_budget_policies_v1 SET policy_json='{}' WHERE task_id='task-1'")
    with pytest.raises(BudgetError, match='budget_policy_conflict'):
        dimension_summary(db, task_id='task-1', owner='backend', dimension='model_fits')
    db.close()


def test_training_bounds_do_not_hide_refit_or_early_stop():
    plan = {'schema_version': 'search-plan-v1', 'effective_trials': 3,
            'candidates': [{}, {}, {}], 'fit_strategy': 'train_then_train_valid_refit'}
    assert training_upper_bound(plan) == {'model_fits': 4, 'training_epochs': 0}
    plan['fit_strategy'] = 'train_best_epoch_no_refit'
    assert training_upper_bound(plan, epochs=5) == {'model_fits': 3, 'training_epochs': 15}
    with pytest.raises(BudgetError, match='budget_invalid_integer'):
        training_upper_bound(plan, epochs=True)


def test_protected_tail_and_deadline(tmp_path):
    db = open_ledger(tmp_path / 'budget.sqlite', policy(), owner='journal')
    db.execute('BEGIN IMMEDIATE')
    reserve(db, task_id='task-1', owner='journal', operation_id='ordinary', attempt_id='0',
            amounts={'api_calls': 3}, payload='ordinary', now=101.0)
    db.commit()
    db.execute('BEGIN IMMEDIATE')
    with pytest.raises(BudgetError, match='budget_insufficient'):
        reserve(db, task_id='task-1', owner='journal', operation_id='ordinary-2', attempt_id='0',
                amounts={'api_calls': 1}, payload='ordinary-2', now=101.0)
    db.rollback()
    db.execute('BEGIN IMMEDIATE')
    reserve(db, task_id='task-1', owner='journal', operation_id='final', attempt_id='0',
            amounts={'api_calls': 9}, payload='final', phase='finalization', now=199.0)
    db.commit()
    db.execute('BEGIN IMMEDIATE')
    with pytest.raises(BudgetError, match='budget_deadline_exceeded'):
        reserve(db, task_id='task-1', owner='journal', operation_id='late', attempt_id='0',
                amounts={'llm_calls': 1}, payload='late', phase='finalization', now=200.0)
    db.rollback()
    db.close()


def test_session_and_complete_search_reserve_in_same_transaction(tmp_path):
    path = tmp_path / 'agent.sqlite'
    repo = AgentSessionRepository(path)
    repo.initialize()
    now = time.time()
    base = policy(fits=4)
    frozen = BudgetPolicy('task-1', now, now+100, now+40, base.limits)
    session, _ = repo.create_session(dataset_id='dataset', selection_metric='macro_f1',
        allowed_models=['logistic_regression'], max_runs=1, seed=42,
        evaluation_config={}, modules=[], context_policy={}, client_request_id='create-1',
        payload_hash='hash-1', principal=Principal(), budget_policy=frozen)
    plan = {'schema_version': 'search-plan-v1', 'effective_trials': 3,
            'candidates': [{}, {}, {}], 'fit_strategy': 'train_then_train_valid_refit',
            'plan_digest': 'plan-1'}
    reservation, created = repo.reserve_experiment(session_id=session.session_id,
        action_json={'model_type': 'logistic_regression'}, rationale=None,
        parent_run_id=None, config_hash='config-1', client_request_id='experiment-1',
        payload_hash='payload-1', active_run_ids=set(), principal=Principal(),
        compiled_config={'execution_search_plan': plan})
    assert created
    db = sqlite3.connect(path)
    assert dimension_summary(db, task_id='task-1', owner='backend', dimension='model_fits')['held_reserved'] == 4
    replay, created = repo.reserve_experiment(session_id=session.session_id,
        action_json={'model_type': 'logistic_regression'}, rationale=None,
        parent_run_id=None, config_hash='config-1', client_request_id='experiment-1',
        payload_hash='payload-1', active_run_ids=set(), principal=Principal(),
        compiled_config={'execution_search_plan': plan})
    assert not created and replay.experiment_id == reservation.experiment_id
    repo.release_reservation(reservation.experiment_id, failure_code='no_run', principal=Principal())
    assert dimension_summary(db, task_id='task-1', owner='backend', dimension='model_fits')['remaining'] == 4
    db.close()


def test_created_run_costs_one_even_when_queued_run_is_cancelled(tmp_path):
    path = tmp_path / 'agent.sqlite'
    repo = AgentSessionRepository(path)
    repo.initialize()
    now = time.time()
    frozen = BudgetPolicy('task-1', now, now+100, now+40, policy(fits=2).limits)
    session, _ = repo.create_session(dataset_id='dataset', selection_metric='macro_f1',
        allowed_models=['logistic_regression'], max_runs=1, seed=42,
        evaluation_config={}, modules=[], context_policy={}, client_request_id='create-1',
        payload_hash='hash-1', principal=Principal(), budget_policy=frozen)
    plan = {'schema_version': 'search-plan-v1', 'effective_trials': 1,
            'candidates': [{}], 'fit_strategy': 'train_then_train_valid_refit',
            'plan_digest': 'plan-1'}
    reservation, _ = repo.reserve_experiment(session_id=session.session_id,
        action_json={'model_type': 'logistic_regression'}, rationale=None,
        parent_run_id=None, config_hash='config-1', client_request_id='experiment-1',
        payload_hash='payload-1', active_run_ids=set(), principal=Principal(),
        compiled_config={'execution_search_plan': plan})
    repo.release_reservation(reservation.experiment_id, failure_code='binding_failed',
                             principal=Principal(), created_run_id='run-created')
    db = sqlite3.connect(path)
    runs = dimension_summary(db, task_id='task-1', owner='backend', dimension='experiments')
    fits = dimension_summary(db, task_id='task-1', owner='backend', dimension='model_fits')
    assert (runs['known_actual'], runs['remaining'], fits['remaining']) == (1, 0, 2)
    assert repo.count_budget_scoped(session_id=session.session_id, principal=Principal()) == 1
    db.close()
