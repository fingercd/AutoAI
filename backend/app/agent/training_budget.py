"""Durable fit/epoch entry facts inside the existing Agent budget reservation.

The parent reserves the entire frozen plan.  An entry converts one held unit to
known actual in the same SQLite transaction as its stable event ID.  Entry is
recorded immediately before the underlying fit or epoch, so a killed child
cannot erase already entered work.  Only a confirmed process exit permits the
unused portion of the parent reservation to be released.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Callable, Iterator

from .budget import BudgetError, dimension_summary, transition


def initialize_training_budget(connection: sqlite3.Connection) -> None:
    connection.execute('''CREATE TABLE IF NOT EXISTS task_training_executions_v1 (
        reservation_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
        run_id TEXT NOT NULL, claim_digest TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('active','unknown_pending','settled')),
        started_at TEXT NOT NULL, finished_at TEXT)''')
    connection.execute('''CREATE TABLE IF NOT EXISTS task_training_events_v1 (
        reservation_id TEXT NOT NULL, event_id TEXT NOT NULL,
        payload_digest TEXT NOT NULL, dimension TEXT NOT NULL,
        kind TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('entered','completed')),
        entered_at TEXT NOT NULL, completed_at TEXT,
        PRIMARY KEY(reservation_id,event_id))''')
    connection.execute('''CREATE TABLE IF NOT EXISTS task_training_terminations_v1 (
        reservation_id TEXT PRIMARY KEY, reason TEXT NOT NULL,
        requested_at TEXT NOT NULL, exited_at TEXT,
        latency_seconds REAL, exit_code INTEGER)''')


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TrainingBinding:
    agent_database: Path
    task_id: str
    policy_digest: str
    reservation_id: str
    work_deadline_at: float

    def child_payload(self) -> dict[str, str]:
        return dict(agent_database=str(self.agent_database), task_id=self.task_id,
                    policy_digest=self.policy_digest,
                    reservation_id=self.reservation_id)


def load_training_binding(agent_database: Path, record: object, *,
                          runs_database: Path | None = None) -> TrainingBinding | None:
    """Resolve a server-owned Run marker to its scoped, bound reservation."""
    config = record.config
    task_id = config.get('execution_budget_task_id')
    digest = config.get('execution_budget_policy_digest')
    marker_absent = task_id is None and digest is None
    if marker_absent and config.get('submission_source') != 'agent':
        return None
    if not marker_absent and (
            not task_id or not digest or config.get('submission_source') != 'agent'):
        raise BudgetError('budget_training_binding_invalid')
    path = Path(agent_database).resolve()
    if not path.is_file():
        if marker_absent:
            return None
        raise BudgetError('budget_training_binding_invalid')
    submission_key = None
    if runs_database is not None:
        runs_uri = 'file:' + Path(runs_database).resolve().as_posix() + '?mode=ro'
        with closing(sqlite3.connect(runs_uri, uri=True)) as runs:
            mapping = runs.execute('''SELECT submission_key FROM run_submission_keys_v1
                WHERE run_id=? AND submission_source='agent' ''',
                (record.run_id,)).fetchone()
            submission_key = mapping[0] if mapping is not None else None
        if submission_key is None and not marker_absent:
            raise BudgetError('budget_training_binding_invalid')
    uri = 'file:' + path.as_posix() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        try:
            selector = 'r.reservation_id' if submission_key is not None else 'r.run_id'
            row = connection.execute(f'''SELECT r.reservation_id,r.state,r.run_id,
            r.compiled_config_json,
            r.owner_id AS reservation_owner,r.tenant_id AS reservation_tenant,
            s.budget_task_id,s.budget_policy_digest,
            s.owner_id AS session_owner,s.tenant_id AS session_tenant
            FROM agent_experiment_reservations_v1 r
            JOIN agent_sessions_v1 s ON s.session_id=r.session_id
            WHERE {selector}=?''',
            (submission_key if submission_key is not None else record.run_id,)).fetchone()
        except sqlite3.OperationalError:
            columns = {row[1] for row in connection.execute(
                'PRAGMA table_info(agent_sessions_v1)').fetchall()}
            if marker_absent and 'budget_task_id' not in columns:
                # Pre-budget Agent databases have no budget columns.
                return None
            raise BudgetError('budget_training_binding_invalid') from None
        if marker_absent:
            if row is None and submission_key is not None:
                raise BudgetError('budget_training_binding_invalid')
            if row is not None and row['budget_task_id'] is not None:
                raise BudgetError('budget_training_binding_missing')
            return None
        if (row is None or row['state'] != 'bound' or row['run_id'] != record.run_id or
                row['budget_task_id'] != task_id or row['budget_policy_digest'] != digest or
                row['session_owner'] != record.owner_id or
                row['session_tenant'] != record.tenant_id or
                row['reservation_owner'] != record.owner_id or
                row['reservation_tenant'] != record.tenant_id):
            raise BudgetError('budget_training_binding_invalid')
        prepared = json.loads(row['compiled_config_json'])
        if (prepared.get('execution_search_plan') != config.get('execution_search_plan') or
                prepared.get('execution_budget_task_id') != task_id or
                prepared.get('execution_budget_policy_digest') != digest):
            raise BudgetError('budget_training_binding_invalid')
        policy = connection.execute('''SELECT policy_json,policy_digest FROM
            task_budget_policies_v1 WHERE task_id=? AND owner='backend' ''',
            (task_id,)).fetchone()
        if policy is None or policy['policy_digest'] != digest:
            raise BudgetError('budget_policy_conflict')
        dimension_summary(connection, task_id=task_id, owner='backend',
                          dimension='model_fits')
        body = json.loads(policy['policy_json'])
        deadline = body['work_deadline_at']
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise BudgetError('budget_invalid_deadline')
        return TrainingBinding(path, task_id, digest, row['reservation_id'], deadline)


def reconcile_orphaned_run(agent_database: Path, record: object, *,
                           runs_database: Path) -> bool:
    """Keep unresolved child use held after cancellation or parent loss."""
    binding = load_training_binding(agent_database, record,
                                    runs_database=runs_database)
    if binding is None:
        return False
    with closing(sqlite3.connect(binding.agent_database, timeout=30,
                                 isolation_level=None)) as connection, connection:
        connection.row_factory = sqlite3.Row
        connection.execute('BEGIN IMMEDIATE')
        execution = connection.execute('''SELECT state FROM task_training_executions_v1
            WHERE reservation_id=? AND task_id=? AND run_id=?''',
            (binding.reservation_id, binding.task_id, record.run_id)).fetchone()
        if execution is not None and execution['state'] == 'settled':
            return True
        termination = connection.execute('''SELECT exited_at FROM
            task_training_terminations_v1 WHERE reservation_id=?''',
            (binding.reservation_id,)).fetchone()
        confirmed = execution is None or (termination is not None and
                                           termination['exited_at'] is not None)
        for dimension in ('model_fits', 'training_epochs'):
            row = connection.execute('''SELECT actual,held,status FROM
                task_budget_reservations_v1 WHERE task_id=? AND owner='backend'
                AND operation_id=? AND attempt_id='0' AND dimension=?''',
                (binding.task_id, binding.reservation_id, dimension)).fetchone()
            if row is None:
                raise BudgetError('budget_reservation_missing')
            if row['status'] in ('settled', 'released'):
                continue
            transition(connection, task_id=binding.task_id, owner='backend',
                       operation_id=binding.reservation_id, attempt_id='0',
                       dimension=dimension,
                       status='settled' if confirmed else 'unknown_pending',
                       actual=row['actual'] or 0,
                       held=None if confirmed else row['held'],
                       source_ref=f'run-{record.run_id}')
        if execution is not None:
            connection.execute('''UPDATE task_training_executions_v1
                SET state=?,finished_at=? WHERE reservation_id=?''',
                ('settled' if confirmed else 'unknown_pending',
                 _utc() if confirmed else None, binding.reservation_id))
    return True


class TrainingBudget:
    """One Run/claim's actual use of its previously reserved training bound."""

    def __init__(self, database_path: Path, *, task_id: str, policy_digest: str,
                 reservation_id: str, run_id: str, claim_token: str,
                 claim_check: Callable[[], None]) -> None:
        if not all(isinstance(value, str) and value for value in
                   (task_id, policy_digest, reservation_id, run_id, claim_token)):
            raise BudgetError('budget_training_identity_invalid')
        self.database_path = Path(database_path)
        self.task_id = task_id
        self.policy_digest = policy_digest
        self.reservation_id = reservation_id
        self.run_id = run_id
        self.claim_digest = hashlib.sha256(claim_token.encode('utf-8')).hexdigest()
        self.claim_check = claim_check

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute('PRAGMA busy_timeout=30000')
            with connection:
                yield connection
        finally:
            connection.close()

    def _binding(self, connection: sqlite3.Connection) -> dict:
        row = connection.execute('''SELECT r.state,r.run_id,s.budget_task_id,
            s.budget_policy_digest,s.owner_id AS session_owner,
            s.tenant_id AS session_tenant,r.owner_id AS reservation_owner,
            r.tenant_id AS reservation_tenant
            FROM agent_experiment_reservations_v1 r
            JOIN agent_sessions_v1 s ON s.session_id=r.session_id
            WHERE r.reservation_id=?''', (self.reservation_id,)).fetchone()
        if (row is None or row['state'] != 'bound' or row['run_id'] != self.run_id or
                row['budget_task_id'] != self.task_id or
                row['budget_policy_digest'] != self.policy_digest or
                row['session_owner'] != row['reservation_owner'] or
                row['session_tenant'] != row['reservation_tenant']):
            raise BudgetError('budget_training_binding_invalid')
        policy = connection.execute('''SELECT policy_json,policy_digest FROM
            task_budget_policies_v1 WHERE task_id=? AND owner='backend' ''',
            (self.task_id,)).fetchone()
        if policy is None or policy['policy_digest'] != self.policy_digest:
            raise BudgetError('budget_policy_conflict')
        # dimension_summary verifies the stored policy body against its digest.
        dimension_summary(connection, task_id=self.task_id, owner='backend',
                          dimension='model_fits')
        return json.loads(policy['policy_json'])

    def _execution(self, connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute('''SELECT * FROM task_training_executions_v1
            WHERE reservation_id=?''', (self.reservation_id,)).fetchone()
        if (row is None or row['task_id'] != self.task_id or
                row['run_id'] != self.run_id or row['claim_digest'] != self.claim_digest):
            raise BudgetError('budget_training_claim_conflict')
        return row

    def bind(self, *, now: float | None = None) -> None:
        self.claim_check()
        current = time.time() if now is None else now
        if type(current) not in (int, float) or not math.isfinite(current):
            raise BudgetError('budget_invalid_deadline')
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            policy = self._binding(connection)
            row = connection.execute('''SELECT * FROM task_training_executions_v1
                WHERE reservation_id=?''', (self.reservation_id,)).fetchone()
            if row is not None:
                if (row['task_id'], row['run_id'], row['claim_digest'], row['state']) != (
                        self.task_id, self.run_id, self.claim_digest, 'active'):
                    raise BudgetError('budget_training_claim_conflict')
                return
            if current >= policy['work_deadline_at']:
                raise BudgetError('budget_deadline_exceeded')
            connection.execute('''INSERT INTO task_training_executions_v1
                (reservation_id,task_id,run_id,claim_digest,state,started_at)
                VALUES(?,?,?,?,'active',?)''',
                (self.reservation_id, self.task_id, self.run_id,
                 self.claim_digest, _utc()))

    def settle_unstarted(self) -> None:
        """Record zero actual use when the work deadline rejects before spawn."""
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            self._binding(connection)
            if connection.execute('''SELECT 1 FROM task_training_executions_v1
                    WHERE reservation_id=?''', (self.reservation_id,)).fetchone():
                raise BudgetError('budget_training_execution_already_started')
            for dimension in ('model_fits', 'training_epochs'):
                row = connection.execute('''SELECT status,actual,held FROM
                    task_budget_reservations_v1 WHERE task_id=? AND owner='backend'
                    AND operation_id=? AND attempt_id='0' AND dimension=?''',
                    (self.task_id, self.reservation_id, dimension)).fetchone()
                if row is None:
                    raise BudgetError('budget_reservation_missing')
                if row['status'] == 'settled' and row['actual'] == 0 and row['held'] == 0:
                    continue
                transition(connection, task_id=self.task_id, owner='backend',
                           operation_id=self.reservation_id, attempt_id='0',
                           dimension=dimension, status='settled', actual=0,
                           source_ref=f'run-{self.run_id}')

    def enter(self, event_id: str, *, dimension: str, kind: str,
              now: float | None = None) -> None:
        if dimension not in ('model_fits', 'training_epochs') or not event_id or not kind:
            raise BudgetError('budget_training_event_invalid')
        current = time.time() if now is None else now
        if type(current) not in (int, float) or not math.isfinite(current):
            raise BudgetError('budget_invalid_deadline')
        self.claim_check()
        digest = hashlib.sha256(json.dumps({'dimension': dimension, 'kind': kind},
                                            sort_keys=True).encode('utf-8')).hexdigest()
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            policy = self._binding(connection)
            execution = self._execution(connection)
            existing = connection.execute('''SELECT payload_digest FROM
                task_training_events_v1 WHERE reservation_id=? AND event_id=?''',
                (self.reservation_id, event_id)).fetchone()
            if existing is not None:
                if existing['payload_digest'] != digest:
                    raise BudgetError('budget_training_event_conflict')
                return
            if execution['state'] != 'active':
                raise BudgetError('budget_training_execution_closed')
            if current >= policy['work_deadline_at']:
                raise BudgetError('budget_deadline_exceeded')
            row = connection.execute('''SELECT amount,actual,held,status FROM
                task_budget_reservations_v1 WHERE task_id=? AND owner='backend'
                AND operation_id=? AND attempt_id='0' AND dimension=?''',
                (self.task_id, self.reservation_id, dimension)).fetchone()
            if (row is None or row['status'] != 'reserved' or
                    (row['actual'] or 0) + row['held'] != row['amount'] or
                    row['held'] < 1):
                raise BudgetError('budget_training_bound_exhausted')
            connection.execute('''UPDATE task_budget_reservations_v1
                SET actual=?,held=? WHERE task_id=? AND owner='backend'
                AND operation_id=? AND attempt_id='0' AND dimension=?''',
                ((row['actual'] or 0) + 1, row['held'] - 1,
                 self.task_id, self.reservation_id, dimension))
            connection.execute('''INSERT INTO task_training_events_v1
                (reservation_id,event_id,payload_digest,dimension,kind,status,entered_at)
                VALUES(?,?,?,?,?,'entered',?)''',
                (self.reservation_id, event_id, digest, dimension, kind, _utc()))

    def complete(self, event_id: str) -> None:
        if not event_id:
            raise BudgetError('budget_training_event_invalid')
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            self._binding(connection)
            if self._execution(connection)['state'] == 'settled':
                raise BudgetError('budget_training_execution_closed')
            changed = connection.execute('''UPDATE task_training_events_v1
                SET status='completed',completed_at=?
                WHERE reservation_id=? AND event_id=? AND status='entered' ''',
                (_utc(), self.reservation_id, event_id)).rowcount
            if changed == 0 and connection.execute('''SELECT status FROM task_training_events_v1
                    WHERE reservation_id=? AND event_id=?''',
                    (self.reservation_id, event_id)).fetchone() is None:
                raise BudgetError('budget_training_event_missing')

    def settle(self, *, exit_confirmed: bool) -> None:
        """Release unused plan capacity only after the child tree is gone."""
        if type(exit_confirmed) is not bool:
            raise BudgetError('budget_training_exit_invalid')
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            self._binding(connection)
            execution = self._execution(connection)
            if execution['state'] == 'settled':
                if not exit_confirmed:
                    raise BudgetError('budget_training_settlement_conflict')
                return
            for dimension in ('model_fits', 'training_epochs'):
                row = connection.execute('''SELECT actual,held,status FROM
                    task_budget_reservations_v1 WHERE task_id=? AND owner='backend'
                    AND operation_id=? AND attempt_id='0' AND dimension=?''',
                    (self.task_id, self.reservation_id, dimension)).fetchone()
                if row is None:
                    raise BudgetError('budget_reservation_missing')
                transition(connection, task_id=self.task_id, owner='backend',
                           operation_id=self.reservation_id, attempt_id='0',
                           dimension=dimension,
                           status='settled' if exit_confirmed else 'unknown_pending',
                           actual=row['actual'] or 0,
                           held=None if exit_confirmed else row['held'],
                           source_ref=f'run-{self.run_id}')
            connection.execute('''UPDATE task_training_executions_v1
                SET state=?,finished_at=? WHERE reservation_id=?''',
                ('settled' if exit_confirmed else 'unknown_pending',
                 _utc() if exit_confirmed else None, self.reservation_id))

    def record_termination(self, evidence: object) -> None:
        """Keep supervisor evidence outside disposable Run artifacts."""
        values = (evidence.reason, evidence.requested_at, evidence.exited_at,
                  evidence.latency_seconds, evidence.exit_code)
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            self._binding(connection)
            self._execution(connection)
            old = connection.execute('''SELECT reason,requested_at,exited_at,
                latency_seconds,exit_code FROM task_training_terminations_v1
                WHERE reservation_id=?''', (self.reservation_id,)).fetchone()
            if old is not None:
                if tuple(old) != values:
                    raise BudgetError('budget_training_termination_conflict')
                return
            connection.execute('''INSERT INTO task_training_terminations_v1
                (reservation_id,reason,requested_at,exited_at,latency_seconds,exit_code)
                VALUES(?,?,?,?,?,?)''', (self.reservation_id, *values))
