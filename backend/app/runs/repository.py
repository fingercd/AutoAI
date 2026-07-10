from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .contracts import RunRecord, RunState


ALLOWED_TRANSITIONS: dict[RunState, set[RunState]] = {
    'queued': {'running', 'cancelled', 'failed'},
    'running': {'succeeded', 'failed', 'cancelled'},
    'succeeded': set(),
    'failed': set(),
    'cancelled': set(),
}


class InvalidRunTransition(RuntimeError):
    pass


class RunNotFound(KeyError):
    pass


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _decode_json(value: str | None, default: object) -> object:
    if not value:
        return default
    return json.loads(value)


class RunRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=str(row['run_id']),
            state=row['state'],
            version=int(row['version']),
            dataset_id=row['dataset_id'],
            legacy_data_path=row['legacy_data_path'],
            config=dict(_decode_json(row['config_json'], {})),
            progress=dict(_decode_json(row['progress_json'], {})),
            claim_token=row['claim_token'],
            worker_id=row['worker_id'],
            lease_expires_at=row['lease_expires_at'],
            manifest_name=row['manifest_name'],
            error=row['error'],
        )

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
                    version INTEGER NOT NULL,
                    dataset_id TEXT,
                    legacy_data_path TEXT,
                    config_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    claim_token TEXT,
                    worker_id TEXT,
                    lease_expires_at TEXT,
                    manifest_name TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                '''
            )
            connection.execute('CREATE INDEX IF NOT EXISTS idx_runs_queue ON runs(state, created_at)')

    def create_queued(
        self,
        *,
        dataset_id: str | None,
        legacy_data_path: str | None = None,
        config: dict[str, Any],
    ) -> RunRecord:
        run_id = uuid.uuid4().hex
        now = _timestamp(datetime.now(timezone.utc))
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute(
                '''
                INSERT INTO runs (
                    run_id, state, version, dataset_id, legacy_data_path,
                    config_json, progress_json, created_at, updated_at
                ) VALUES (?, 'queued', 1, ?, ?, ?, '{}', ?, ?)
                ''',
                (run_id, dataset_id, legacy_data_path, json.dumps(config, ensure_ascii=False), now, now),
            )
            row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert row is not None
        return self._record(row)

    def claim_next(self, *, worker_id: str, now: datetime, lease_seconds: int = 30) -> RunRecord | None:
        now_text = _timestamp(now)
        lease_text = _timestamp(now + timedelta(seconds=lease_seconds))
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                "SELECT * FROM runs WHERE state = 'queued' ORDER BY created_at, run_id LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            claim_token = uuid.uuid4().hex
            changed = connection.execute(
                '''
                UPDATE runs
                SET state = 'running', version = version + 1, claim_token = ?,
                    worker_id = ?, lease_expires_at = ?, updated_at = ?
                WHERE run_id = ? AND state = 'queued'
                ''',
                (claim_token, worker_id, lease_text, now_text, row['run_id']),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot claim run {row["run_id"]}')
            updated = connection.execute('SELECT * FROM runs WHERE run_id = ?', (row['run_id'],)).fetchone()
            connection.commit()
        assert updated is not None
        return self._record(updated)

    def renew_lease(self, run_id: str, *, claim_token: str, now: datetime, lease_seconds: int = 30) -> RunRecord:
        now_text = _timestamp(now)
        lease_text = _timestamp(now + timedelta(seconds=lease_seconds))
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            changed = connection.execute(
                '''
                UPDATE runs
                SET lease_expires_at = ?, version = version + 1, updated_at = ?
                WHERE run_id = ? AND state = 'running' AND claim_token = ?
                  AND lease_expires_at > ?
                ''',
                (lease_text, now_text, run_id, claim_token, now_text),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot renew lease for run {run_id}')
            row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert row is not None
        return self._record(row)

    def update_progress(
        self,
        run_id: str,
        *,
        claim_token: str,
        now: datetime,
        progress: dict[str, Any],
    ) -> RunRecord:
        now_text = _timestamp(now)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            changed = connection.execute(
                '''
                UPDATE runs
                SET progress_json = ?, version = version + 1, updated_at = ?
                WHERE run_id = ? AND state = 'running' AND claim_token = ?
                ''',
                (json.dumps(progress, ensure_ascii=False), now_text, run_id, claim_token),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot update progress for run {run_id}')
            row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert row is not None
        return self._record(row)

    def cancel(self, run_id: str, *, now: datetime) -> RunRecord:
        now_text = _timestamp(now)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise RunNotFound(run_id)
            state = row['state']
            if 'cancelled' not in ALLOWED_TRANSITIONS[state]:
                connection.rollback()
                raise InvalidRunTransition(f'cannot cancel {state} run {run_id}')
            changed = connection.execute(
                '''
                UPDATE runs
                SET state = 'cancelled', version = version + 1, lease_expires_at = NULL, updated_at = ?
                WHERE run_id = ? AND state = ?
                ''',
                (now_text, run_id, state),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot cancel run {run_id}')
            updated = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert updated is not None
        return self._record(updated)

    def finish_success(
        self,
        run_id: str,
        *,
        claim_token: str,
        now: datetime,
        manifest_name: str = 'manifest.json',
    ) -> RunRecord:
        return self._finish(run_id, claim_token=claim_token, now=now, state='succeeded', manifest_name=manifest_name)

    def finish_failure(self, run_id: str, *, claim_token: str, now: datetime, error: str) -> RunRecord:
        return self._finish(run_id, claim_token=claim_token, now=now, state='failed', error=error)

    def _finish(
        self,
        run_id: str,
        *,
        claim_token: str,
        now: datetime,
        state: RunState,
        manifest_name: str | None = None,
        error: str | None = None,
    ) -> RunRecord:
        now_text = _timestamp(now)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            changed = connection.execute(
                '''
                UPDATE runs
                SET state = ?, version = version + 1, lease_expires_at = NULL,
                    manifest_name = ?, error = ?, updated_at = ?
                WHERE run_id = ? AND state = 'running' AND claim_token = ?
                ''',
                (state, manifest_name, error, now_text, run_id, claim_token),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot finish run {run_id}')
            row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert row is not None
        return self._record(row)

    def requeue_expired(self, *, now: datetime) -> int:
        now_text = _timestamp(now)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            changed = connection.execute(
                '''
                UPDATE runs
                SET state = 'queued', version = version + 1, claim_token = NULL,
                    worker_id = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE state = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?
                ''',
                (now_text, now_text),
            ).rowcount
            connection.commit()
        return int(changed)

    def assert_active(self, run_id: str, *, claim_token: str, now: datetime) -> None:
        now_text = _timestamp(now)
        with self._connection() as connection:
            row = connection.execute(
                '''
                SELECT 1 FROM runs
                WHERE run_id = ? AND state = 'running' AND claim_token = ?
                  AND lease_expires_at > ?
                ''',
                (run_id, claim_token, now_text),
            ).fetchone()
        if row is None:
            raise InvalidRunTransition(f'run {run_id} is not active for this claim')

    def get(self, run_id: str) -> RunRecord:
        with self._connection() as connection:
            row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
        if row is None:
            raise RunNotFound(run_id)
        return self._record(row)

    def list(self) -> list[RunRecord]:
        with self._connection() as connection:
            rows = connection.execute('SELECT * FROM runs ORDER BY created_at DESC, run_id DESC').fetchall()
        return [self._record(row) for row in rows]

    def exists(self, run_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute('SELECT 1 FROM runs WHERE run_id = ?', (run_id,)).fetchone()
        return row is not None

    def import_legacy(
        self,
        *,
        run_id: str,
        state: RunState,
        config: dict[str, Any],
        dataset_id: str | None,
        legacy_data_path: str | None,
    ) -> RunRecord:
        now = _timestamp(datetime.now(timezone.utc))
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            if connection.execute('SELECT 1 FROM runs WHERE run_id = ?', (run_id,)).fetchone() is not None:
                connection.rollback()
                return self.get(run_id)
            connection.execute(
                '''
                INSERT INTO runs (
                    run_id, state, version, dataset_id, legacy_data_path, config_json,
                    progress_json, created_at, updated_at
                ) VALUES (?, ?, 1, ?, ?, ?, '{}', ?, ?)
                ''',
                (run_id, state, dataset_id, legacy_data_path, json.dumps(config, ensure_ascii=False), now, now),
            )
            row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert row is not None
        return self._record(row)
