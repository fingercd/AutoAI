"""Single-host JSON checkpoints, process exclusion and durable call accounting."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Iterator

from backend.app.agent.budget import (
    BudgetPolicy, BudgetError, freeze_policy, initialize_ledger, reserve,
    transition, dimension_summary,
)


class PersistenceError(RuntimeError):
    pass


class JSONSerializer:
    """No import hooks, arbitrary object reconstruction or pickle fallback."""

    def dumps_typed(self, value):
        try:
            return 'json', json.dumps(value, ensure_ascii=True, allow_nan=False,
                                      separators=(',', ':')).encode('utf-8')
        except (ValueError, TypeError):
            raise PersistenceError('checkpoint_not_json') from None

    def loads_typed(self, value):
        kind, payload = value
        if kind != 'json':
            raise PersistenceError('checkpoint_encoding_unsupported')
        try:
            return json.loads(payload, parse_constant=self._invalid_constant)
        except (ValueError, TypeError, UnicodeError):
            raise PersistenceError('checkpoint_invalid') from None

    @staticmethod
    def _invalid_constant(_value):
        raise ValueError('nonfinite')


@contextmanager
def thread_lock(directory: Path, thread_id: str) -> Iterator[None]:
    """OS-owned lock: also excludes another process, and releases after a kill."""
    directory.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(thread_id.encode()).hexdigest() + '.lock'
    handle = (directory / filename).open('a+b')
    acquired = False
    try:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            raise PersistenceError('thread_already_running') from None
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class CallJournal:
    """Commits before network dispatch; unresolved crash windows remain unknown.

    This is basic runtime call accounting, not the future scientific cost ledger.
    v4 also stores validated, bound proposals; raw provider responses are never logged.
    """

    def __init__(self, path: Path, thread_id: str):
        self.thread_id = thread_id
        self.canonical_path = os.path.normcase(str(Path(path).resolve()))
        self.budget_policy: BudgetPolicy | None = None
        self.connection = sqlite3.connect(path, timeout=10)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('''CREATE TABLE IF NOT EXISTS orchestration_calls_v1 (
            id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL, operation_id TEXT NOT NULL,
            kind TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL,
            started_at REAL NOT NULL, ended_at REAL, input_tokens INTEGER,
            output_tokens INTEGER, error_code TEXT)''')
        self.connection.execute('''CREATE TABLE IF NOT EXISTS orchestration_monitor_waits_v1 (
            id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL, operation_id TEXT NOT NULL,
            run_id TEXT NOT NULL, task_id TEXT, session_id TEXT,
            status TEXT NOT NULL, started_at_utc TEXT NOT NULL,
            started_at_epoch REAL NOT NULL, ended_at_utc TEXT, duration_seconds REAL,
            duration_clock TEXT NOT NULL, reason_code TEXT,
            UNIQUE(thread_id,operation_id))''')
        initialize_ledger(self.connection)
        self.connection.execute('''CREATE TABLE IF NOT EXISTS task_budget_journal_bindings_v1 (
            task_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL UNIQUE,
            canonical_path TEXT NOT NULL, policy_digest TEXT NOT NULL)''')
        self.connection.commit()
        # Nullable additions preserve old settled rows: do not invent totals or
        # change their historical State token-status projection during upgrade.
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            columns = {row[1] for row in self.connection.execute(
                'PRAGMA table_info(orchestration_calls_v1)')}
            for name, kind in (('total_tokens', 'INTEGER'), ('token_status', 'TEXT'), ('proposal_json','TEXT'),
                               ('attempt_index','INTEGER'),('request_started_at_utc','TEXT'),
                               ('response_ended_at_utc','TEXT'),('request_duration_seconds','REAL'),
                               ('measurement_version','TEXT'),('model_id_sha256','TEXT'),('model_id','TEXT'),
                               ('protocol','TEXT'),('phase','TEXT'),('request_outcome','TEXT'),
                               ('task_id','TEXT'),('session_id','TEXT'),('run_id','TEXT'),
                               ('budget_phase','TEXT'),('cached_tokens','INTEGER'),
                               ('prepare_duration_seconds','REAL'),('parse_duration_seconds','REAL')):
                if name not in columns:
                    self.connection.execute(f'ALTER TABLE orchestration_calls_v1 ADD COLUMN {name} {kind}')
            wait_columns={row[1] for row in self.connection.execute(
                'PRAGMA table_info(orchestration_monitor_waits_v1)')}
            for name in ('task_id','session_id'):
                if name not in wait_columns:
                    self.connection.execute(
                        f'ALTER TABLE orchestration_monitor_waits_v1 ADD COLUMN {name} TEXT')

    def bind_budget(self, policy: BudgetPolicy) -> None:
        """Freeze this task's call owner before any new physical attempt."""
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            freeze_policy(self.connection, policy, owner='journal')
            row = self.connection.execute('''SELECT task_id,thread_id,canonical_path,policy_digest
                FROM task_budget_journal_bindings_v1 WHERE task_id=? OR thread_id=?''',
                (policy.task_id, self.thread_id)).fetchone()
            expected = (policy.task_id, self.thread_id, self.canonical_path, policy.digest)
            if row is None:
                prior = self.connection.execute('''SELECT 1 FROM orchestration_calls_v1
                    WHERE thread_id=? LIMIT 1''', (self.thread_id,)).fetchone()
                if prior is not None:
                    raise PersistenceError('budget_binding_after_calls')
                self.connection.execute('''INSERT INTO task_budget_journal_bindings_v1
                    (task_id,thread_id,canonical_path,policy_digest) VALUES(?,?,?,?)''', expected)
            elif tuple(row) != expected:
                raise PersistenceError('budget_journal_binding_conflict')
            leaked = self.connection.execute('''SELECT 1 FROM orchestration_calls_v1
                WHERE thread_id=? AND (budget_phase IS NULL OR task_id IS NOT ?) LIMIT 1''',
                (self.thread_id, policy.task_id)).fetchone()
            if leaked is not None:
                raise PersistenceError('budget_unmetered_call_conflict')
        self.budget_policy = policy

    def _require_budget_binding(self, task_id: str | None = None) -> None:
        """A persisted budget cannot silently turn into legacy accounting on reopen."""
        row = self.connection.execute('''SELECT task_id,thread_id,canonical_path,policy_digest
            FROM task_budget_journal_bindings_v1 WHERE thread_id=? OR task_id IS ?''',
            (self.thread_id, task_id)).fetchone()
        if row is None:
            if self.budget_policy is not None:
                raise PersistenceError('budget_journal_binding_missing')
            return
        if row[1] != self.thread_id or row[2] != self.canonical_path:
            raise PersistenceError('budget_journal_binding_conflict')
        if self.budget_policy is None:
            raise PersistenceError('budget_binding_required')
        if row[0] != self.budget_policy.task_id or row[3] != self.budget_policy.digest:
            raise PersistenceError('budget_journal_binding_conflict')
        try:
            dimension_summary(self.connection, task_id=self.budget_policy.task_id,
                              owner='journal', dimension='api_calls')
        except BudgetError as error:
            raise PersistenceError(error.code) from None

    def budget_summary(self) -> dict[str, dict]:
        if self.budget_policy is None:
            raise PersistenceError('budget_policy_missing')
        return {name: dimension_summary(self.connection, task_id=self.budget_policy.task_id,
                                        owner='journal', dimension=name)
                for name, spec in self.budget_policy.as_dict()['dimensions'].items()
                if spec['owner'] == 'journal'}

    def _call_amounts(self, kind: str) -> dict[str, int]:
        if self.budget_policy is None:
            return {}
        amounts = {kind + '_calls': 1}
        if kind == 'llm':
            amounts['output_tokens'] = self.budget_policy.max_output_tokens_per_call
            if self.budget_policy.limits['input_tokens'] is not None:
                amounts['input_tokens'] = self.budget_policy.input_tokens_per_call_bound
            if self.budget_policy.limits['cached_tokens'] is not None:
                amounts['cached_tokens'] = self.budget_policy.input_tokens_per_call_bound
        return amounts

    def begin(self, *, operation_id: str, kind: str, name: str, maximum: int,
              max_attempts: int, deadline: float, now: float | None = None,
              task_id: str | None = None, session_id: str | None = None,
              run_id: str | None = None, phase: str = 'work') -> int:
        now = time.time() if now is None else now
        self._require_budget_binding(task_id)
        if now >= deadline:
            raise PersistenceError('deadline_exceeded')
        if self.budget_policy is not None:
            expected_attempts = (self.budget_policy.max_repair_attempts + 1 if kind == 'llm'
                                 else self.budget_policy.max_operation_attempts)
            if (maximum != self.budget_policy.limits.get(kind + '_calls') or
                    max_attempts != expected_attempts or
                    deadline != self.budget_policy.deadline_at):
                raise PersistenceError('budget_legacy_limit_conflict')
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            total = self.connection.execute(
                "SELECT count(*) FROM orchestration_calls_v1 WHERE thread_id=? AND kind=? AND status!='prepare_failed'",
                (self.thread_id, kind)).fetchone()[0]
            attempts = self.connection.execute(
                'SELECT count(*) FROM orchestration_calls_v1 WHERE thread_id=? AND operation_id=?',
                (self.thread_id, operation_id)).fetchone()[0]
            if total >= maximum:
                raise PersistenceError(f'{kind}_call_limit')
            if attempts >= max_attempts:
                raise PersistenceError('operation_attempt_limit')
            if self.budget_policy is not None:
                if task_id != self.budget_policy.task_id:
                    raise PersistenceError('budget_task_mismatch')
                try:
                    reserve(self.connection, task_id=task_id, owner='journal',
                            operation_id=operation_id, attempt_id=str(attempts),
                            amounts=self._call_amounts(kind),
                            payload={'kind': kind, 'name': name, 'session_id': session_id,
                                     'run_id': run_id}, phase=phase, now=now)
                except BudgetError as error:
                    raise PersistenceError(error.code) from None
            cursor = self.connection.execute('''INSERT INTO orchestration_calls_v1
                (thread_id,operation_id,kind,name,status,started_at,attempt_index,
                 task_id,session_id,run_id,budget_phase)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                (self.thread_id, operation_id, kind, name,
                 'prepared' if self.budget_policy is not None else 'dispatched', now, attempts,
                 task_id,session_id,run_id,phase if self.budget_policy is not None else None))
            return int(cursor.lastrowid)

    def mark_dispatched(self, call_id: int, *, now: float | None = None) -> float | None:
        """Persist the send boundary; a later crash remains an unknown hold."""
        self._require_budget_binding()
        now = time.time() if now is None else now
        if type(now) not in (int, float) or not math.isfinite(now):
            raise PersistenceError('budget_invalid_deadline')
        expired = False
        remaining = None
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            row = self.connection.execute('''SELECT task_id,operation_id,attempt_index,kind,status,budget_phase
                FROM orchestration_calls_v1 WHERE id=? AND thread_id=?''',
                (call_id, self.thread_id)).fetchone()
            if row is None:
                raise PersistenceError('call_missing')
            task_id, operation_id, attempt, kind, status, phase = row
            if status == 'dispatched':
                return None
            if status != 'prepared' or self.budget_policy is None:
                raise PersistenceError('call_dispatch_conflict')
            if phase not in ('work', 'finalization'):
                raise PersistenceError('budget_phase_invalid')
            deadline = (self.budget_policy.work_deadline_at if phase == 'work'
                        else self.budget_policy.deadline_at)
            expired = now >= deadline
            for dimension in self._call_amounts(kind):
                transition(self.connection, task_id=task_id, owner='journal',
                           operation_id=operation_id, attempt_id=str(attempt),
                           dimension=dimension, status='released' if expired else 'dispatched')
            self.connection.execute('''UPDATE orchestration_calls_v1
                SET status=?,ended_at=?,error_code=? WHERE id=? AND thread_id=?''',
                ('prepare_failed' if expired else 'dispatched', now if expired else None,
                 'budget_deadline_exceeded' if expired else None, call_id, self.thread_id))
            remaining = deadline - now
        if expired:
            raise PersistenceError('budget_deadline_exceeded')
        return remaining

    def finish(self, call_id: int, *, error_code: str | None = None,
               input_tokens: int | None = None, output_tokens: int | None = None,
               total_tokens: int | None = None, token_status: str | None = None,
               cached_tokens: int | None = None,
               proposal: dict | None = None, measurement: dict | None = None):
        # A second completion notification cannot rewrite settled measurements.
        self._require_budget_binding()
        with self.connection:
            measurement = measurement or {}
            if self.budget_policy is not None:
                self.connection.execute('BEGIN IMMEDIATE')
                row = self.connection.execute('''SELECT task_id,operation_id,attempt_index,kind,status
                    FROM orchestration_calls_v1 WHERE id=? AND thread_id=?''',
                    (call_id, self.thread_id)).fetchone()
                if row is None:
                    raise PersistenceError('call_missing')
                task_id, operation_id, attempt, kind, status = row
                if status in ('prepare_failed', 'confirmed', 'failed'):
                    return
                if status == 'prepared':
                    for dimension in self._call_amounts(kind):
                        transition(self.connection, task_id=task_id, owner='journal',
                                   operation_id=operation_id, attempt_id=str(attempt),
                                   dimension=dimension, status='released')
                    self.connection.execute('''UPDATE orchestration_calls_v1
                        SET status='prepare_failed',ended_at=?,error_code=?
                        WHERE id=? AND thread_id=?''',
                        (time.time(), error_code or 'prepare_failed', call_id, self.thread_id))
                    return
                if status == 'dispatched':
                    transition(self.connection, task_id=task_id, owner='journal',
                               operation_id=operation_id, attempt_id=str(attempt),
                               dimension=kind + '_calls', status='settled', actual=1,
                               source_ref=f'call-{call_id}')
                    if kind == 'llm':
                        for dimension, value in (('input_tokens', input_tokens),
                                                 ('output_tokens', output_tokens),
                                                 ('cached_tokens', cached_tokens)):
                            if dimension not in self._call_amounts(kind):
                                continue
                            transition(self.connection, task_id=task_id, owner='journal',
                                operation_id=operation_id, attempt_id=str(attempt),
                                dimension=dimension,
                                status='settled' if value is not None else 'unknown_pending',
                                actual=value, source_ref=f'call-{call_id}' if value is not None else None)
            self.connection.execute('''UPDATE orchestration_calls_v1 SET status=?,ended_at=?,
                input_tokens=?,output_tokens=?,total_tokens=?,token_status=?,error_code=?,proposal_json=?,
                request_started_at_utc=?,response_ended_at_utc=?,request_duration_seconds=?,
                measurement_version=?,model_id_sha256=?,model_id=?,protocol=?,phase=?,request_outcome=?,
                cached_tokens=?,prepare_duration_seconds=?,parse_duration_seconds=?
                WHERE id=? AND thread_id=? AND status='dispatched' ''',
                ('failed' if error_code else 'confirmed', time.time(), input_tokens,
                 output_tokens, total_tokens, token_status, error_code,
                 json.dumps(proposal,ensure_ascii=True,allow_nan=False) if proposal is not None else None,
                 measurement.get('request_started_at_utc'),measurement.get('response_ended_at_utc'),
                 measurement.get('request_duration_seconds'),
                 ('llm-client-request-v2' if any(key in measurement for key in
                    ('prepare_duration_seconds', 'parse_duration_seconds')) else
                  'llm-client-request-v1') if measurement else None,
                 measurement.get('model_id_sha256'),measurement.get('model_id'),measurement.get('protocol'),
                 measurement.get('phase'),measurement.get('outcome'),cached_tokens,
                 measurement.get('prepare_duration_seconds'),measurement.get('parse_duration_seconds'),
                 call_id, self.thread_id))

    def proposal(self, operation_id: str) -> str | None:
        row=self.connection.execute('''SELECT proposal_json FROM orchestration_calls_v1
            WHERE thread_id=? AND operation_id=? AND kind='llm' AND status='confirmed'
            ORDER BY id DESC LIMIT 1''',(self.thread_id,operation_id)).fetchone()
        if row is None:
            return None
        if row[0] is None:
            raise PersistenceError('journal_proposal_missing')
        return row[0]

    def snapshot(self):
        rows = self.connection.execute('''SELECT id,operation_id,kind,name,status,started_at,
            input_tokens,output_tokens,error_code,total_tokens,token_status,attempt_index,
            request_started_at_utc,response_ended_at_utc,request_duration_seconds,
            measurement_version,model_id_sha256,model_id,protocol,phase,request_outcome,
            task_id,session_id,run_id,cached_tokens,prepare_duration_seconds,
            parse_duration_seconds FROM orchestration_calls_v1
            WHERE thread_id=? ORDER BY id''', (self.thread_id,)).fetchall()
        keys = ('id','operation_id','kind','name','status','started_at',
                'input_tokens','output_tokens','error_code','total_tokens','token_status',
                'attempt_index','request_started_at_utc','response_ended_at_utc',
                'request_duration_seconds','measurement_version','model_id_sha256','model_id',
                'protocol','phase','request_outcome','task_id','session_id','run_id',
                'cached_tokens','prepare_duration_seconds','parse_duration_seconds')
        return [dict(zip(keys, row)) for row in rows]

    def export_llm_requests(self) -> list[dict]:
        return [dict(thread_id=self.thread_id,task_id=row['task_id'],
                     session_id=row['session_id'],run_id=row['run_id'],call_id=row['id'],
                     operation_id=row['operation_id'], retry_index=row['attempt_index'],
                     phase=row['phase'] or row['name'], status=row['status'],
                     request_started_at_utc=row['request_started_at_utc'],
                     response_ended_at_utc=row['response_ended_at_utc'],
                     request_duration_seconds=row['request_duration_seconds'],
                     measurement_version=row['measurement_version'] or 'legacy-wrapper-unknown',
                     model_id_sha256=row['model_id_sha256'],model_id=row['model_id'],protocol=row['protocol'],
                     request_outcome=row['request_outcome'],input_tokens=row['input_tokens'],
                     output_tokens=row['output_tokens'],total_tokens=row['total_tokens'],
                     cached_tokens=row['cached_tokens'],
                     prepare_duration_seconds=row['prepare_duration_seconds'],
                     parse_duration_seconds=row['parse_duration_seconds'],
                     token_status=row['token_status'] or 'unknown',error_code=row['error_code'])
                for row in self.snapshot() if row['kind']=='llm']

    def begin_monitor_wait(self, *, operation_id: str, run_id: str,
                           reason_code: str | None = None, task_id: str | None = None,
                           session_id: str | None = None) -> None:
        now=time.time()
        started=datetime.fromtimestamp(now,timezone.utc).isoformat()
        with self.connection:
            self.connection.execute('''INSERT OR IGNORE INTO orchestration_monitor_waits_v1
                (thread_id,operation_id,run_id,task_id,session_id,status,
                 started_at_utc,started_at_epoch,duration_clock,reason_code)
                VALUES (?,?,?,?,?,?,?,?,?,?)''',
                (self.thread_id,operation_id,run_id,task_id,session_id,'running',started,now,
                 'utc_wall_same_host_estimate',reason_code))

    def end_monitor_wait(self, *, operation_id: str, status: str = 'succeeded') -> None:
        if status not in ('succeeded','interrupted'):
            raise ValueError('invalid wait status')
        now=time.time()
        ended=datetime.fromtimestamp(now,timezone.utc).isoformat()
        with self.connection:
            self.connection.execute('''UPDATE orchestration_monitor_waits_v1
                SET status=CASE WHEN ?>=started_at_epoch THEN ? ELSE 'unknown' END,
                    ended_at_utc=?,duration_seconds=CASE WHEN ?>=started_at_epoch
                        THEN ?-started_at_epoch ELSE NULL END
                WHERE thread_id=? AND operation_id=? AND status='running' ''',
                (now,status,ended,now,now,self.thread_id,operation_id))

    def export_monitor_waits(self) -> list[dict]:
        rows=self.connection.execute('''SELECT operation_id,run_id,task_id,session_id,status,started_at_utc,
            ended_at_utc,duration_seconds,duration_clock,reason_code
            FROM orchestration_monitor_waits_v1 WHERE thread_id=? ORDER BY id''',
            (self.thread_id,)).fetchall()
        keys=('operation_id','run_id','task_id','session_id','status','started_at_utc','ended_at_utc',
              'duration_seconds','duration_clock','reason_code')
        return [dict(thread_id=self.thread_id,**dict(zip(keys,row))) for row in rows]

    def close(self):
        self.connection.close()
