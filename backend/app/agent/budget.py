"""Frozen task budget rules and transactional, single-owner reservations.

The caller owns the SQLite connection and its BEGIN IMMEDIATE transaction.  A
backend reservation can therefore be committed with its training allocation;
the orchestration journal uses the same rules in its own database for calls.
Neither database purports to be a transaction spanning both owners.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sqlite3
from typing import Mapping
from types import MappingProxyType


POLICY_VERSION = 'budget-policy-v1'
LEDGER_VERSION = 'task-budget-ledger-v1'
DIMENSIONS = {
    'experiments': ('runs', 'backend'),
    'model_fits': ('fits', 'backend'),
    'training_epochs': ('epochs', 'backend'),
    'llm_calls': ('calls', 'journal'),
    'api_calls': ('calls', 'journal'),
    'input_tokens': ('tokens', 'journal'),
    'output_tokens': ('tokens', 'journal'),
    'cached_tokens': ('tokens', 'journal'),
    'output_repairs': ('calls', 'journal'),
    'network_retries': ('calls', 'journal'),
}


class BudgetError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise BudgetError('budget_invalid_integer')
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class BudgetPolicy:
    task_id: str
    started_at: float
    deadline_at: float
    work_deadline_at: float
    limits: Mapping[str, int | None]
    finalization_llm_calls: int = 3
    finalization_api_calls: int = 9
    max_operation_attempts: int = 3
    max_repair_attempts: int = 2
    max_output_tokens_per_call: int = 1024
    input_tokens_per_call_bound: int | None = None
    input_bound_source: str | None = None
    source_version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise BudgetError('budget_invalid_task')
        if self.source_version != POLICY_VERSION:
            raise BudgetError('budget_policy_version_unsupported')
        times = (self.started_at, self.work_deadline_at, self.deadline_at)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in times):
            raise BudgetError('budget_invalid_deadline')
        if not self.started_at < self.work_deadline_at < self.deadline_at:
            raise BudgetError('budget_invalid_deadline')
        if set(self.limits) != set(DIMENSIONS):
            raise BudgetError('budget_dimension_mismatch')
        for dimension, value in self.limits.items():
            if value is not None:
                _nonnegative_int(value)
            elif dimension not in ('input_tokens', 'cached_tokens'):
                raise BudgetError('budget_hard_limit_missing')
        for value in (self.finalization_llm_calls, self.finalization_api_calls,
                      self.max_operation_attempts, self.max_repair_attempts,
                      self.max_output_tokens_per_call):
            _nonnegative_int(value)
        if self.max_operation_attempts < 1 or self.max_output_tokens_per_call < 1:
            raise BudgetError('budget_invalid_integer')
        if self.input_tokens_per_call_bound is not None:
            _nonnegative_int(self.input_tokens_per_call_bound)
        if (self.limits['input_tokens'] is not None or self.limits['cached_tokens'] is not None):
            if (self.input_tokens_per_call_bound is None or
                    self.input_bound_source != 'provider_verified'):
                raise BudgetError('budget_input_bound_unverified')
        if self.limits['output_tokens'] < self.finalization_llm_calls * self.max_output_tokens_per_call:
            raise BudgetError('budget_finalization_insufficient')
        for dimension in ('input_tokens', 'cached_tokens'):
            if (self.limits[dimension] is not None and
                    self.limits[dimension] < self.finalization_llm_calls * self.input_tokens_per_call_bound):
                raise BudgetError('budget_finalization_insufficient')
        if self.limits['llm_calls'] < self.finalization_llm_calls or self.limits['api_calls'] < self.finalization_api_calls:
            raise BudgetError('budget_finalization_insufficient')
        object.__setattr__(self, 'limits', MappingProxyType(dict(self.limits)))

    def as_dict(self) -> dict:
        return dict(version=self.source_version, task_id=self.task_id,
                    started_at=self.started_at, deadline_at=self.deadline_at,
                    work_deadline_at=self.work_deadline_at,
                    dimensions={key: dict(unit=unit, owner=owner, limit=self.limits[key],
                                          enforcement='hard' if self.limits[key] is not None else 'observation_only')
                                for key, (unit, owner) in DIMENSIONS.items()},
                    finalization=dict(llm_calls=self.finalization_llm_calls,
                                      api_calls=self.finalization_api_calls,
                                      output_tokens=self.finalization_llm_calls * self.max_output_tokens_per_call,
                                      input_tokens=(self.finalization_llm_calls * self.input_tokens_per_call_bound
                                          if self.limits['input_tokens'] is not None else 0),
                                      cached_tokens=(self.finalization_llm_calls * self.input_tokens_per_call_bound
                                          if self.limits['cached_tokens'] is not None else 0)),
                    max_operation_attempts=self.max_operation_attempts,
                    max_repair_attempts=self.max_repair_attempts,
                    max_output_tokens_per_call=self.max_output_tokens_per_call,
                    input_tokens_per_call_bound=self.input_tokens_per_call_bound,
                    input_bound_source=self.input_bound_source)

    @property
    def digest(self) -> str:
        return _digest(self.as_dict())

    @classmethod
    def from_dict(cls, body: object) -> 'BudgetPolicy':
        """Accept only the exact versioned wire image of a validated policy."""
        if type(body) is not dict:
            raise BudgetError('budget_policy_invalid')
        try:
            limits = {name: body['dimensions'][name]['limit'] for name in DIMENSIONS}
            policy = cls(task_id=body['task_id'], started_at=body['started_at'],
                deadline_at=body['deadline_at'], work_deadline_at=body['work_deadline_at'],
                limits=limits,
                finalization_llm_calls=body['finalization']['llm_calls'],
                finalization_api_calls=body['finalization']['api_calls'],
                max_operation_attempts=body['max_operation_attempts'],
                max_repair_attempts=body['max_repair_attempts'],
                max_output_tokens_per_call=body['max_output_tokens_per_call'],
                input_tokens_per_call_bound=body['input_tokens_per_call_bound'],
                input_bound_source=body['input_bound_source'],
                source_version=body['version'])
            if policy.as_dict() != body:
                raise BudgetError('budget_policy_invalid')
            return policy
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, BudgetError):
                raise
            raise BudgetError('budget_policy_invalid') from exc


def training_upper_bound(plan: Mapping[str, object], *, epochs: int | None = None) -> dict[str, int]:
    """Count base model fits; preprocessing and normalizer fits are separate spans."""
    if plan.get('schema_version') != 'search-plan-v1':
        raise BudgetError('budget_search_plan_invalid')
    trials = _nonnegative_int(plan.get('effective_trials'))
    if trials < 1 or trials > len(plan.get('candidates', ())):
        raise BudgetError('budget_search_plan_invalid')
    strategy = plan.get('fit_strategy')
    if strategy == 'train_then_train_valid_refit':
        if epochs is not None:
            raise BudgetError('budget_epoch_not_applicable')
        return {'model_fits': trials + 1, 'training_epochs': 0}
    if strategy == 'train_best_epoch_no_refit':
        count = _nonnegative_int(epochs)
        if count < 1:
            raise BudgetError('budget_epochs_invalid')
        return {'model_fits': trials, 'training_epochs': trials * count}
    raise BudgetError('budget_search_plan_invalid')


def initialize_ledger(connection: sqlite3.Connection) -> None:
    connection.execute('''CREATE TABLE IF NOT EXISTS task_budget_policies_v1 (
        task_id TEXT PRIMARY KEY, owner TEXT NOT NULL, policy_json TEXT NOT NULL,
        policy_digest TEXT NOT NULL)''')
    connection.execute('''CREATE TABLE IF NOT EXISTS task_budget_reservations_v1 (
        task_id TEXT NOT NULL, owner TEXT NOT NULL, operation_id TEXT NOT NULL,
        attempt_id TEXT NOT NULL, dimension TEXT NOT NULL, payload_digest TEXT NOT NULL,
        amount INTEGER NOT NULL, actual INTEGER, held INTEGER NOT NULL,
        status TEXT NOT NULL, source_ref TEXT,
        PRIMARY KEY(task_id, owner, operation_id, attempt_id, dimension))''')


def freeze_policy(connection: sqlite3.Connection, policy: BudgetPolicy, *, owner: str) -> None:
    if owner not in ('backend', 'journal'):
        raise BudgetError('budget_owner_invalid')
    body = _canonical(policy.as_dict())
    row = connection.execute('SELECT owner,policy_digest FROM task_budget_policies_v1 WHERE task_id=?',
                             (policy.task_id,)).fetchone()
    if row is not None:
        if tuple(row) != (owner, policy.digest):
            raise BudgetError('budget_policy_conflict')
        return
    connection.execute('INSERT INTO task_budget_policies_v1 VALUES(?,?,?,?)',
                       (policy.task_id, owner, body, policy.digest))


def _policy(connection: sqlite3.Connection, task_id: str, owner: str) -> dict:
    row = connection.execute('SELECT policy_json,policy_digest FROM task_budget_policies_v1 WHERE task_id=? AND owner=?',
                             (task_id, owner)).fetchone()
    if row is None:
        raise BudgetError('budget_policy_missing')
    policy = json.loads(row[0])
    if _digest(policy) != row[1] or policy.get('task_id') != task_id:
        raise BudgetError('budget_policy_conflict')
    return policy


def dimension_summary(connection: sqlite3.Connection, *, task_id: str, owner: str,
                      dimension: str) -> dict:
    policy = _policy(connection, task_id, owner)
    spec = policy['dimensions'].get(dimension)
    if spec is None or spec['owner'] != owner:
        raise BudgetError('budget_dimension_owner_mismatch')
    rows = connection.execute('''SELECT actual,held,status FROM task_budget_reservations_v1
        WHERE task_id=? AND owner=? AND dimension=?''', (task_id, owner, dimension)).fetchall()
    known = sum(row[0] or 0 for row in rows)
    held_reserved = sum(row[1] for row in rows if row[2] == 'reserved')
    held_unknown = sum(row[1] for row in rows if row[2] in ('dispatched', 'unknown_pending'))
    limit = spec['limit']
    return dict(unit=spec['unit'], owner=owner, limit=limit, known_actual=known,
                held_reserved=held_reserved, held_unknown=held_unknown,
                remaining=None if limit is None else max(0, limit-known-held_reserved-held_unknown),
                budget_breach=limit is not None and known+held_reserved+held_unknown > limit,
                known_count=sum(row[0] is not None for row in rows),
                unknown_count=sum(row[2] in ('dispatched', 'unknown_pending') for row in rows))


def reserve(connection: sqlite3.Connection, *, task_id: str, owner: str,
            operation_id: str, attempt_id: str, amounts: Mapping[str, int],
            payload: object, phase: str = 'work', now: float | None = None) -> None:
    """Call only inside BEGIN IMMEDIATE; all dimensions reserve atomically."""
    policy = _policy(connection, task_id, owner)
    if phase not in ('work', 'finalization'):
        raise BudgetError('budget_phase_invalid')
    if now is not None:
        if type(now) not in (int, float) or not math.isfinite(now):
            raise BudgetError('budget_invalid_deadline')
        deadline = policy['work_deadline_at'] if phase == 'work' else policy['deadline_at']
        if now >= deadline:
            raise BudgetError('budget_deadline_exceeded')
    if not operation_id or not attempt_id or not amounts:
        raise BudgetError('budget_reservation_invalid')
    digest = _digest({'payload': payload, 'phase': phase})
    for dimension, amount in amounts.items():
        _nonnegative_int(amount)
        spec = policy['dimensions'].get(dimension)
        if spec is None or spec['owner'] != owner:
            raise BudgetError('budget_dimension_owner_mismatch')
        row = connection.execute('''SELECT amount,payload_digest FROM task_budget_reservations_v1
            WHERE task_id=? AND owner=? AND operation_id=? AND attempt_id=? AND dimension=?''',
            (task_id, owner, operation_id, attempt_id, dimension)).fetchone()
        if row is not None:
            if tuple(row) != (amount, digest):
                raise BudgetError('budget_reservation_conflict')
            continue
        summary = dimension_summary(connection, task_id=task_id, owner=owner, dimension=dimension)
        remaining = summary['remaining']
        if remaining is not None:
            protected = 0
            if phase == 'work':
                protected = policy['finalization'].get(dimension, 0)
            if amount > remaining - protected:
                raise BudgetError('budget_insufficient')
        connection.execute('''INSERT INTO task_budget_reservations_v1
            (task_id,owner,operation_id,attempt_id,dimension,payload_digest,amount,actual,held,status)
            VALUES(?,?,?,?,?,?,?,NULL,?,'reserved')''',
            (task_id, owner, operation_id, attempt_id, dimension, digest, amount, amount))


def transition(connection: sqlite3.Connection, *, task_id: str, owner: str,
               operation_id: str, attempt_id: str, dimension: str, status: str,
               actual: int | None = None, held: int | None = None,
               source_ref: str | None = None) -> None:
    """A terminal fact is immutable; unknown holds remain until reconciled."""
    key = (task_id, owner, operation_id, attempt_id, dimension)
    row = connection.execute('''SELECT amount,actual,held,status,source_ref
        FROM task_budget_reservations_v1 WHERE task_id=? AND owner=? AND operation_id=?
        AND attempt_id=? AND dimension=?''', key).fetchone()
    if row is None:
        raise BudgetError('budget_reservation_missing')
    amount, old_actual, old_held, old_status, old_source = row
    if status not in ('dispatched', 'settled', 'released', 'unknown_pending'):
        raise BudgetError('budget_status_invalid')
    if actual is not None:
        _nonnegative_int(actual)
    if held is not None:
        _nonnegative_int(held)
    if old_status == status and (old_actual, old_held, old_source) == (
            actual, old_held if held is None else held, source_ref):
        return
    if old_status in ('settled', 'released'):
        raise BudgetError('budget_transition_conflict')
    if status == 'dispatched':
        if old_status != 'reserved' or actual is not None or held is not None:
            raise BudgetError('budget_transition_invalid')
        next_held = old_held
    elif status == 'released':
        if old_status != 'reserved' or actual is not None or held is not None:
            raise BudgetError('budget_transition_invalid')
        next_held = 0
    elif status == 'settled':
        if old_status not in ('reserved', 'dispatched', 'unknown_pending') or actual is None:
            raise BudgetError('budget_transition_invalid')
        next_held = 0 if held is None else held
        if next_held:
            raise BudgetError('budget_transition_invalid')
    else:
        if old_status not in ('reserved', 'dispatched', 'unknown_pending'):
            raise BudgetError('budget_transition_invalid')
        next_held = old_held if held is None else held
        if next_held > amount:
            raise BudgetError('budget_transition_invalid')
    if old_actual is not None and (actual is None or actual < old_actual):
        raise BudgetError('budget_transition_conflict')
    connection.execute('''UPDATE task_budget_reservations_v1
        SET status=?,actual=?,held=?,source_ref=?
        WHERE task_id=? AND owner=? AND operation_id=? AND attempt_id=? AND dimension=?''',
        (status, actual, next_held, source_ref, *key))
