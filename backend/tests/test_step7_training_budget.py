"""Actual training entry/settlement persists outside disposable Run artifacts."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from backend.app.agent.budget import BudgetError, BudgetPolicy, DIMENSIONS, dimension_summary
from backend.app.agent.repository import AgentSessionRepository
from backend.app.agent.training_budget import (TrainingBudget, load_training_binding,
                                               reconcile_orphaned_run)
from backend.app.runs.contracts import Principal
from backend.app.runs.supervisor import SupervisionStopped, supervise


def _setup(tmp_path, *, deep=False):
    path = tmp_path / 'agent.sqlite3'
    repository = AgentSessionRepository(path)
    repository.initialize()
    now = time.time()
    limits = {name: 100 for name in DIMENSIONS}
    limits.update(experiments=1, model_fits=2, training_epochs=2,
                  llm_calls=6, api_calls=12, input_tokens=None,
                  cached_tokens=None, output_tokens=6144)
    policy = BudgetPolicy('task-fit', now - 1, now + 30, now + 15, limits)
    session, _ = repository.create_session(
        dataset_id='dataset', selection_metric='macro_f1',
        allowed_models=['cnn1d' if deep else 'logistic_regression'],
        max_runs=1, seed=42, evaluation_config={}, modules=[], context_policy={},
        client_request_id='session', payload_hash='hash', principal=Principal(),
        budget_policy=policy)
    plan = {'schema_version': 'search-plan-v1', 'effective_trials': 1,
            'candidates': [{}], 'plan_digest': 'plan',
            'fit_strategy': ('train_best_epoch_no_refit' if deep else
                             'train_then_train_valid_refit')}
    compiled = {'execution_search_plan': plan,
                'execution_budget_task_id': policy.task_id,
                'execution_budget_policy_digest': policy.digest}
    if deep:
        compiled['epochs'] = 2
    reservation, _ = repository.reserve_experiment(
        session_id=session.session_id, action_json={'model_type': 'cnn1d' if deep else
            'logistic_regression'}, rationale=None, parent_run_id=None,
        config_hash='config', client_request_id='experiment', payload_hash='payload',
        active_run_ids=set(), principal=Principal(), compiled_config=compiled)
    repository.bind_experiment(reservation.experiment_id, run_id='run-1',
                               principal=Principal())
    ledger = TrainingBudget(path, task_id=policy.task_id,
                            policy_digest=policy.digest,
                            reservation_id=reservation.experiment_id,
                            run_id='run-1', claim_token='claim-1',
                            claim_check=lambda: None)
    return path, policy, reservation, ledger


def _summary(path, dimension):
    with sqlite3.connect(path) as connection:
        return dimension_summary(connection, task_id='task-fit', owner='backend',
                                 dimension=dimension)


def test_fit_entry_survives_interruption_and_releases_only_after_exit(tmp_path):
    path, _, reservation, ledger = _setup(tmp_path)
    ledger.bind()
    ledger.enter('fold-1-trial-0-fit', dimension='model_fits', kind='trial')
    ledger.enter('fold-1-trial-0-fit', dimension='model_fits', kind='trial')
    summary = _summary(path, 'model_fits')
    assert (summary['known_actual'], summary['held_reserved'], summary['remaining']) == (1, 1, 0)
    with pytest.raises(BudgetError, match='budget_training_event_conflict'):
        ledger.enter('fold-1-trial-0-fit', dimension='model_fits', kind='refit')
    ledger.settle(exit_confirmed=False)
    summary = _summary(path, 'model_fits')
    assert (summary['known_actual'], summary['held_unknown'], summary['remaining']) == (1, 1, 0)
    with pytest.raises(BudgetError, match='budget_training_execution_closed'):
        ledger.enter('fold-1-refit', dimension='model_fits', kind='refit')
    ledger.settle(exit_confirmed=True)
    ledger.settle(exit_confirmed=True)
    summary = _summary(path, 'model_fits')
    assert (summary['known_actual'], summary['held_unknown'], summary['remaining']) == (1, 0, 1)
    with sqlite3.connect(path) as connection:
        row = connection.execute('''SELECT status,actual,held FROM task_budget_reservations_v1
            WHERE operation_id=? AND dimension='model_fits' ''',
            (reservation.experiment_id,)).fetchone()
        assert row == ('settled', 1, 0)


def test_epoch_entered_and_completed_are_separate_facts(tmp_path):
    path, _, _, ledger = _setup(tmp_path, deep=True)
    ledger.bind()
    ledger.enter('fit-1', dimension='model_fits', kind='trial')
    ledger.enter('epoch-1', dimension='training_epochs', kind='trial')
    ledger.complete('epoch-1')
    ledger.complete('epoch-1')
    ledger.enter('epoch-2', dimension='training_epochs', kind='trial')
    with pytest.raises(BudgetError, match='budget_training_bound_exhausted'):
        ledger.enter('epoch-3', dimension='training_epochs', kind='trial')
    ledger.settle(exit_confirmed=True)
    assert _summary(path, 'training_epochs')['known_actual'] == 2
    with sqlite3.connect(path) as connection:
        assert connection.execute('''SELECT status FROM task_training_events_v1
            WHERE event_id='epoch-1' ''').fetchone()[0] == 'completed'
        assert connection.execute('''SELECT status FROM task_training_events_v1
            WHERE event_id='epoch-2' ''').fetchone()[0] == 'entered'


def test_wrong_claim_and_deadline_cannot_consume_more_fits(tmp_path):
    path, policy, reservation, ledger = _setup(tmp_path)
    ledger.bind()
    wrong = TrainingBudget(path, task_id=policy.task_id,
                           policy_digest=policy.digest,
                           reservation_id=reservation.experiment_id,
                           run_id='run-1', claim_token='other-claim',
                           claim_check=lambda: None)
    with pytest.raises(BudgetError, match='budget_training_claim_conflict'):
        wrong.enter('fit-1', dimension='model_fits', kind='trial')
    with pytest.raises(BudgetError, match='budget_deadline_exceeded'):
        ledger.enter('fit-1', dimension='model_fits', kind='trial',
                     now=policy.work_deadline_at)
    assert _summary(path, 'model_fits')['known_actual'] == 0
    with pytest.raises(BudgetError, match='budget_training_binding_invalid'):
        TrainingBudget(path, task_id=policy.task_id,
                       policy_digest=policy.digest,
                       reservation_id=reservation.experiment_id,
                       run_id='foreign-run', claim_token='claim-1',
                       claim_check=lambda: None).bind()


def test_run_binding_requires_scoped_mapping_and_exact_plan(tmp_path):
    path, policy, reservation, _ = _setup(tmp_path)
    with sqlite3.connect(path) as connection:
        prepared = connection.execute('''SELECT compiled_config_json FROM
            agent_experiment_reservations_v1 WHERE reservation_id=?''',
            (reservation.experiment_id,)).fetchone()[0]
    config = json.loads(prepared)
    config['submission_source'] = 'agent'
    record = SimpleNamespace(run_id='run-1', config=config,
                             owner_id=None, tenant_id=None)
    loaded = load_training_binding(path, record)
    assert loaded is not None
    assert loaded.task_id == policy.task_id
    assert loaded.reservation_id == reservation.experiment_id
    record.config = {**config, 'execution_search_plan': {'plan_digest': 'foreign'}}
    with pytest.raises(BudgetError, match='budget_training_binding_invalid'):
        load_training_binding(path, record)
    record.config = config
    record.owner_id = 'foreign'
    with pytest.raises(BudgetError, match='budget_training_binding_invalid'):
        load_training_binding(path, record)
    record.owner_id = None
    record.config = {key: value for key, value in config.items()
                     if key not in ('execution_budget_task_id',
                                    'execution_budget_policy_digest')}
    with pytest.raises(BudgetError, match='budget_training_binding_missing'):
        load_training_binding(path, record)


def test_expired_queued_run_settles_zero_without_fit(tmp_path):
    path, policy, _, ledger = _setup(tmp_path)
    with pytest.raises(BudgetError, match='budget_deadline_exceeded'):
        ledger.bind(now=policy.work_deadline_at)
    ledger.settle_unstarted()
    ledger.settle_unstarted()
    summary = _summary(path, 'model_fits')
    assert (summary['known_actual'], summary['remaining']) == (0, 2)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM task_training_events_v1').fetchone()[0] == 0


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows Job Object contract')
def test_blocked_child_termination_preserves_fit_and_latency(tmp_path):
    path, _, reservation, ledger = _setup(tmp_path)
    ledger.bind()
    ledger.enter('fit-blocked', dimension='model_fits', kind='trial')
    deadline = datetime.now(timezone.utc) + timedelta(seconds=0.5)
    with pytest.raises(SupervisionStopped) as caught:
        supervise({'probe': 'block'}, deadline_at=deadline, active=lambda: True,
                  poll_seconds=0.05, on_stop=ledger.record_termination)
    assert caught.value.evidence.reason == 'deadline'
    ledger.settle(exit_confirmed=True)
    with sqlite3.connect(path) as connection:
        row = connection.execute('''SELECT reason,latency_seconds,exit_code
            FROM task_training_terminations_v1 WHERE reservation_id=?''',
            (reservation.experiment_id,)).fetchone()
        assert row[0] == 'deadline' and row[1] is not None and row[1] < 2
        assert row[2] is not None
    summary = _summary(path, 'model_fits')
    assert (summary['known_actual'], summary['held_reserved']) == (1, 0)


def test_parent_loss_keeps_unconfirmed_fit_hold(tmp_path):
    path, _, reservation, ledger = _setup(tmp_path)
    ledger.bind()
    ledger.enter('fit-started', dimension='model_fits', kind='trial')
    with sqlite3.connect(path) as connection:
        config = json.loads(connection.execute('''SELECT compiled_config_json FROM
            agent_experiment_reservations_v1 WHERE reservation_id=?''',
            (reservation.experiment_id,)).fetchone()[0])
    config['submission_source'] = 'agent'
    record = SimpleNamespace(run_id='run-1', config=config,
                             owner_id=None, tenant_id=None)
    runs_database = tmp_path / 'runs.sqlite3'
    with sqlite3.connect(runs_database) as connection:
        connection.execute('''CREATE TABLE run_submission_keys_v1 (
            submission_key TEXT,run_id TEXT,submission_source TEXT)''')
        connection.execute('INSERT INTO run_submission_keys_v1 VALUES(?,?,?)',
                           (reservation.experiment_id, 'run-1', 'agent'))
    assert reconcile_orphaned_run(path, record, runs_database=runs_database)
    summary = _summary(path, 'model_fits')
    assert (summary['known_actual'], summary['held_unknown'], summary['remaining']) == (1, 1, 0)
    assert reconcile_orphaned_run(path, record, runs_database=runs_database)
    ledger.settle(exit_confirmed=True)
    assert _summary(path, 'model_fits')['remaining'] == 1
