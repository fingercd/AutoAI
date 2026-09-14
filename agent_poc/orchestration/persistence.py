"""Single-host JSON checkpoints, process exclusion and durable call accounting."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Iterator


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
        self.connection = sqlite3.connect(path, timeout=10)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('''CREATE TABLE IF NOT EXISTS orchestration_calls_v1 (
            id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL, operation_id TEXT NOT NULL,
            kind TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL,
            started_at REAL NOT NULL, ended_at REAL, input_tokens INTEGER,
            output_tokens INTEGER, error_code TEXT)''')
        self.connection.commit()
        # Nullable additions preserve old settled rows: do not invent totals or
        # change their historical State token-status projection during upgrade.
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            columns = {row[1] for row in self.connection.execute(
                'PRAGMA table_info(orchestration_calls_v1)')}
            for name, kind in (('total_tokens', 'INTEGER'), ('token_status', 'TEXT'), ('proposal_json','TEXT')):
                if name not in columns:
                    self.connection.execute(f'ALTER TABLE orchestration_calls_v1 ADD COLUMN {name} {kind}')

    def begin(self, *, operation_id: str, kind: str, name: str, maximum: int,
              max_attempts: int, deadline: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        if now >= deadline:
            raise PersistenceError('deadline_exceeded')
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            total = self.connection.execute(
                'SELECT count(*) FROM orchestration_calls_v1 WHERE thread_id=? AND kind=?',
                (self.thread_id, kind)).fetchone()[0]
            attempts = self.connection.execute(
                'SELECT count(*) FROM orchestration_calls_v1 WHERE thread_id=? AND operation_id=?',
                (self.thread_id, operation_id)).fetchone()[0]
            if total >= maximum:
                raise PersistenceError(f'{kind}_call_limit')
            if attempts >= max_attempts:
                raise PersistenceError('operation_attempt_limit')
            cursor = self.connection.execute('''INSERT INTO orchestration_calls_v1
                (thread_id,operation_id,kind,name,status,started_at) VALUES (?,?,?,?,?,?)''',
                (self.thread_id, operation_id, kind, name, 'dispatched', now))
            return int(cursor.lastrowid)

    def finish(self, call_id: int, *, error_code: str | None = None,
               input_tokens: int | None = None, output_tokens: int | None = None,
               total_tokens: int | None = None, token_status: str | None = None,
               proposal: dict | None = None):
        # A second completion notification cannot rewrite settled measurements.
        with self.connection:
            self.connection.execute('''UPDATE orchestration_calls_v1 SET status=?,ended_at=?,
                input_tokens=?,output_tokens=?,total_tokens=?,token_status=?,error_code=?,proposal_json=?
                WHERE id=? AND thread_id=? AND status='dispatched' ''',
                ('failed' if error_code else 'confirmed', time.time(), input_tokens,
                 output_tokens, total_tokens, token_status, error_code,
                 json.dumps(proposal,ensure_ascii=True,allow_nan=False) if proposal is not None else None,
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
            input_tokens,output_tokens,error_code,total_tokens,token_status FROM orchestration_calls_v1
            WHERE thread_id=? ORDER BY id''', (self.thread_id,)).fetchall()
        keys = ('id','operation_id','kind','name','status','started_at',
                'input_tokens','output_tokens','error_code','total_tokens','token_status')
        return [dict(zip(keys, row)) for row in rows]

    def close(self):
        self.connection.close()
