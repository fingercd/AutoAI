"""SQLite Run 状态机与 worker claim/lease 仓库。

所有关键迁移在 ``BEGIN IMMEDIATE`` 事务内校验当前 state、claim token 和 version。
queued Run 只能被一个 worker claim；运行 lease 过期后可回队列；终态不可逆。
这些数据库约束是并发正确性的来源，status.json 不参与状态决策。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .contracts import Principal, RunRecord, RunState


ALLOWED_TRANSITIONS: dict[RunState, set[RunState]] = {
    'queued': {'running', 'cancelled', 'failed'},
    'running': {'succeeded', 'failed', 'cancelled'},
    'succeeded': set(),
    'failed': set(),
    'cancelled': set(),
}
TERMINAL_STATES: set[RunState] = {'succeeded', 'failed', 'cancelled'}


class InvalidRunTransition(RuntimeError):
    """请求的状态迁移违反当前状态、claim 或版本约束。"""
    pass


class RunNotFound(KeyError):
    """请求的 run_id 在当前仓库和 Principal 范围内不存在。"""
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
    """提供 Run CRUD、合法迁移、claim、续租和过期回收操作。"""
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
        columns = set(row.keys())
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
            error_details=dict(_decode_json(row['error_json'], {})) if 'error_json' in columns else {},
            dataset_snapshot=dict(_decode_json(row['dataset_snapshot_json'], {})) if 'dataset_snapshot_json' in columns else {},
            owner_id=row['owner_id'] if 'owner_id' in columns else None,
            tenant_id=row['tenant_id'] if 'tenant_id' in columns else None,
            created_at=row['created_at'] if 'created_at' in columns else None,
            updated_at=row['updated_at'] if 'updated_at' in columns else None,
            started_at=row['started_at'] if 'started_at' in columns else None,
            finished_at=row['finished_at'] if 'finished_at' in columns else None,
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
                    error_json TEXT NOT NULL DEFAULT '{}',
                    dataset_snapshot_json TEXT NOT NULL DEFAULT '{}',
                    owner_id TEXT,
                    tenant_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
                '''
            )
            columns = {
                str(row['name'])
                for row in connection.execute('PRAGMA table_info(runs)').fetchall()
            }
            migrations = {
                'error_json': "ALTER TABLE runs ADD COLUMN error_json TEXT NOT NULL DEFAULT '{}'",
                'dataset_snapshot_json': "ALTER TABLE runs ADD COLUMN dataset_snapshot_json TEXT NOT NULL DEFAULT '{}'",
                'owner_id': 'ALTER TABLE runs ADD COLUMN owner_id TEXT',
                'tenant_id': 'ALTER TABLE runs ADD COLUMN tenant_id TEXT',
                'started_at': 'ALTER TABLE runs ADD COLUMN started_at TEXT',
                'finished_at': 'ALTER TABLE runs ADD COLUMN finished_at TEXT',
            }
            for name, statement in migrations.items():
                if name not in columns:
                    connection.execute(statement)
            connection.execute('CREATE INDEX IF NOT EXISTS idx_runs_queue ON runs(state, created_at)')
            connection.execute(
                'CREATE INDEX IF NOT EXISTS idx_runs_scope ON runs(owner_id, tenant_id, created_at)'
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS worker_heartbeats (
                    worker_id TEXT PRIMARY KEY,
                    last_seen_at TEXT NOT NULL,
                    active_run_id TEXT
                )
                '''
            )

    def create_queued(
        self,
        *,
        dataset_id: str | None,
        legacy_data_path: str | None = None,
        config: dict[str, Any],
        dataset_snapshot: dict[str, Any] | None = None,
        principal: Principal = Principal(),
    ) -> RunRecord:
        run_id = uuid.uuid4().hex
        now = _timestamp(datetime.now(timezone.utc))
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute(
                '''
                INSERT INTO runs (
                    run_id, state, version, dataset_id, legacy_data_path,
                    config_json, progress_json, dataset_snapshot_json,
                    owner_id, tenant_id, created_at, updated_at
                ) VALUES (?, 'queued', 1, ?, ?, ?, '{}', ?, ?, ?, ?, ?)
                ''',
                (
                    run_id,
                    dataset_id,
                    legacy_data_path,
                    json.dumps(config, ensure_ascii=False),
                    json.dumps(dataset_snapshot or {}, ensure_ascii=False),
                    principal.owner_id,
                    principal.tenant_id,
                    now,
                    now,
                ),
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
                    worker_id = ?, lease_expires_at = ?, updated_at = ?,
                    started_at = COALESCE(started_at, ?), finished_at = NULL,
                    error = NULL, error_json = '{}'
                WHERE run_id = ? AND state = 'queued'
                ''',
                (claim_token, worker_id, lease_text, now_text, now_text, row['run_id']),
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

    def update_dataset_snapshot(
        self,
        run_id: str,
        *,
        claim_token: str,
        now: datetime,
        dataset_snapshot: dict[str, Any],
    ) -> RunRecord:
        """在有效 claim 内补齐训练解析得到的可追溯数据集计数。"""
        now_text = _timestamp(now)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            changed = connection.execute(
                '''
                UPDATE runs
                SET dataset_snapshot_json = ?, version = version + 1, updated_at = ?
                WHERE run_id = ? AND state = 'running' AND claim_token = ?
                ''',
                (json.dumps(dataset_snapshot, ensure_ascii=False), now_text, run_id, claim_token),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot update dataset snapshot for run {run_id}')
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
                SET state = 'cancelled', version = version + 1, lease_expires_at = NULL,
                    updated_at = ?, finished_at = ?
                WHERE run_id = ? AND state = ?
                ''',
                (now_text, now_text, run_id, state),
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

    def finish_failure(
        self,
        run_id: str,
        *,
        claim_token: str,
        now: datetime,
        error: str,
        error_details: dict[str, Any] | None = None,
    ) -> RunRecord:
        return self._finish(
            run_id,
            claim_token=claim_token,
            now=now,
            state='failed',
            error=error,
            error_details=error_details,
        )

    def _finish(
        self,
        run_id: str,
        *,
        claim_token: str,
        now: datetime,
        state: RunState,
        manifest_name: str | None = None,
        error: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> RunRecord:
        now_text = _timestamp(now)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            changed = connection.execute(
                '''
                UPDATE runs
                SET state = ?, version = version + 1, lease_expires_at = NULL,
                    manifest_name = ?, error = ?, error_json = ?, updated_at = ?,
                    finished_at = ?
                WHERE run_id = ? AND state = 'running' AND claim_token = ?
                ''',
                (
                    state,
                    manifest_name,
                    error,
                    json.dumps(error_details or {}, ensure_ascii=False),
                    now_text,
                    now_text,
                    run_id,
                    claim_token,
                ),
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

    def record_worker_heartbeat(
        self,
        *,
        worker_id: str,
        now: datetime,
        active_run_id: str | None = None,
    ) -> None:
        now_text = _timestamp(now)
        stale_cutoff = _timestamp(now - timedelta(days=7))
        with self._connection() as connection:
            connection.execute(
                'DELETE FROM worker_heartbeats WHERE last_seen_at < ?',
                (stale_cutoff,),
            )
            connection.execute(
                '''
                INSERT INTO worker_heartbeats (worker_id, last_seen_at, active_run_id)
                VALUES (?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    last_seen_at = excluded.last_seen_at,
                    active_run_id = excluded.active_run_id
                ''',
                (worker_id, now_text, active_run_id),
            )

    def worker_health(self, *, now: datetime, stale_seconds: float = 15.0) -> dict[str, Any]:
        now_utc = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        with self._connection() as connection:
            rows = connection.execute(
                '''
                SELECT worker_id, last_seen_at, active_run_id
                FROM worker_heartbeats ORDER BY last_seen_at DESC LIMIT 20
                '''
            ).fetchall()
        workers = []
        for row in rows:
            try:
                last_seen = datetime.fromisoformat(str(row['last_seen_at']).replace('Z', '+00:00'))
                age = max(0.0, (now_utc.astimezone(timezone.utc) - last_seen.astimezone(timezone.utc)).total_seconds())
            except ValueError:
                age = None
            workers.append(
                {
                    'worker_id': str(row['worker_id']),
                    'last_seen_at': row['last_seen_at'],
                    'active_run_id': row['active_run_id'],
                    'age_seconds': age,
                    'live': age is not None and age <= stale_seconds,
                }
            )
        live_workers = [item for item in workers if item['live']]
        return {
            'available': bool(live_workers),
            'live_count': len(live_workers),
            'last_seen_at': workers[0]['last_seen_at'] if workers else None,
            'active_run_ids': [
                item['active_run_id']
                for item in live_workers
                if item['active_run_id'] is not None
            ],
            'workers': workers,
        }

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

    @staticmethod
    def _scope_values(principal: Principal) -> tuple[str | None, str | None]:
        return principal.owner_id, principal.tenant_id

    def get_scoped(self, run_id: str, *, principal: Principal) -> RunRecord:
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            row = connection.execute(
                '''
                SELECT * FROM runs
                WHERE run_id = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (run_id, owner_id, tenant_id),
            ).fetchone()
        if row is None:
            raise RunNotFound(run_id)
        return self._record(row)

    def list(self) -> list[RunRecord]:
        with self._connection() as connection:
            rows = connection.execute('SELECT * FROM runs ORDER BY created_at DESC, run_id DESC').fetchall()
        return [self._record(row) for row in rows]

    def list_scoped(
        self,
        *,
        principal: Principal,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[RunRecord]:
        owner_id, tenant_id = self._scope_values(principal)
        parameters: list[Any] = [owner_id, tenant_id]
        cursor_clause = ''
        if cursor:
            with self._connection() as connection:
                cursor_row = connection.execute(
                    '''
                    SELECT created_at, run_id FROM runs
                    WHERE run_id = ? AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (cursor, owner_id, tenant_id),
                ).fetchone()
            if cursor_row is None:
                raise RunNotFound(cursor)
            cursor_clause = ' AND (created_at < ? OR (created_at = ? AND run_id < ?))'
            parameters.extend([cursor_row['created_at'], cursor_row['created_at'], cursor_row['run_id']])
        sql = (
            'SELECT * FROM runs WHERE owner_id IS ? AND tenant_id IS ?'
            + cursor_clause
            + ' ORDER BY created_at DESC, run_id DESC'
        )
        if limit is not None:
            sql += ' LIMIT ?'
            parameters.append(max(1, int(limit)))
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._record(row) for row in rows]

    def cancel_scoped(self, run_id: str, *, now: datetime, principal: Principal) -> RunRecord:
        now_text = _timestamp(now)
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                '''
                SELECT * FROM runs
                WHERE run_id = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (run_id, owner_id, tenant_id),
            ).fetchone()
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
                SET state = 'cancelled', version = version + 1, lease_expires_at = NULL,
                    updated_at = ?, finished_at = ?
                WHERE run_id = ? AND state = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (now_text, now_text, run_id, state, owner_id, tenant_id),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot cancel run {run_id}')
            updated = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert updated is not None
        return self._record(updated)

    def exists(self, run_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute('SELECT 1 FROM runs WHERE run_id = ?', (run_id,)).fetchone()
        return row is not None

    def assign_unowned(self, run_id: str, *, principal: Principal) -> bool:
        """显式把 owner/tenant 均为空的历史 Run 绑定到一个服务端 Principal。"""
        if principal.owner_id is None and principal.tenant_id is None:
            raise ValueError('cannot bind an unowned run to an empty principal')
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                'SELECT owner_id, tenant_id FROM runs WHERE run_id = ?',
                (run_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RunNotFound(run_id)
            if row['owner_id'] is not None or row['tenant_id'] is not None:
                connection.commit()
                return False
            changed = connection.execute(
                '''
                UPDATE runs
                SET owner_id = ?, tenant_id = ?, version = version + 1
                WHERE run_id = ? AND owner_id IS NULL AND tenant_id IS NULL
                ''',
                (principal.owner_id, principal.tenant_id, run_id),
            ).rowcount
            connection.commit()
        return changed == 1

    def delete_terminal(self, run_id: str) -> None:
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT state FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise RunNotFound(run_id)
            state: RunState = row['state']
            if state not in TERMINAL_STATES:
                connection.rollback()
                raise InvalidRunTransition(f'cannot delete {state} run {run_id}')
            changed = connection.execute(
                'DELETE FROM runs WHERE run_id = ? AND state IN (\'succeeded\', \'failed\', \'cancelled\')',
                (run_id,),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot delete run {run_id}')
            connection.commit()

    def delete_terminal_scoped(self, run_id: str, *, principal: Principal) -> None:
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                '''
                SELECT state FROM runs
                WHERE run_id = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (run_id, owner_id, tenant_id),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RunNotFound(run_id)
            state: RunState = row['state']
            if state not in TERMINAL_STATES:
                connection.rollback()
                raise InvalidRunTransition(f'cannot delete {state} run {run_id}')
            changed = connection.execute(
                '''
                DELETE FROM runs
                WHERE run_id = ? AND state IN ('succeeded', 'failed', 'cancelled')
                  AND owner_id IS ? AND tenant_id IS ?
                ''',
                (run_id, owner_id, tenant_id),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot delete run {run_id}')
            connection.commit()

    def import_legacy(
        self,
        *,
        run_id: str,
        state: RunState,
        config: dict[str, Any],
        dataset_id: str | None,
        legacy_data_path: str | None,
        principal: Principal = Principal(),
        dataset_snapshot: dict[str, Any] | None = None,
        created_at: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        progress: dict[str, Any] | None = None,
        manifest_name: str | None = None,
        error: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> RunRecord:
        now = _timestamp(datetime.now(timezone.utc))
        created = created_at or now
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            if connection.execute('SELECT 1 FROM runs WHERE run_id = ?', (run_id,)).fetchone() is not None:
                connection.rollback()
                return self.get(run_id)
            connection.execute(
                '''
                INSERT INTO runs (
                    run_id, state, version, dataset_id, legacy_data_path, config_json,
                    progress_json, dataset_snapshot_json, owner_id, tenant_id,
                    manifest_name, error, error_json,
                    created_at, updated_at, started_at, finished_at
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    run_id,
                    state,
                    dataset_id,
                    legacy_data_path,
                    json.dumps(config, ensure_ascii=False),
                    json.dumps(progress or {}, ensure_ascii=False),
                    json.dumps(dataset_snapshot or {}, ensure_ascii=False),
                    principal.owner_id,
                    principal.tenant_id,
                    manifest_name,
                    error,
                    json.dumps(error_details or {}, ensure_ascii=False),
                    created,
                    finished_at or now,
                    started_at,
                    finished_at or (now if state in TERMINAL_STATES else None),
                ),
            )
            row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert row is not None
        return self._record(row)
