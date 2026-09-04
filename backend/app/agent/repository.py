"""Agent Session、幂等记录及 Experiment reservation 的 SQLite 仓储。"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..runs.contracts import Principal
from ..runs.repository import RunNotFound, RunRepository
from .contracts import AgentDomainError


def _timestamp(value: datetime | None = None) -> str:
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _decode(value: str | None, default: Any) -> Any:
    return json.loads(value) if value else default


class AgentSessionNotFound(AgentDomainError):
    def __init__(self, *_: object) -> None:
        super().__init__('agent_session_not_found', 'agent session 不存在', status_code=404)


class AgentExperimentNotFound(AgentDomainError):
    def __init__(self, *_: object) -> None:
        super().__init__('agent_experiment_not_found', 'agent experiment 不存在', status_code=404)


class AgentSessionClosed(AgentDomainError):
    def __init__(self, message: str = 'agent session 已 finalized') -> None:
        super().__init__('agent_session_finalized', message, status_code=409)


class AgentConfigCollision(AgentDomainError):
    def __init__(self, message: str = '相同 config_hash 已提交') -> None:
        super().__init__('agent_duplicate_config', message, status_code=409)


class AgentActiveRunExists(AgentDomainError):
    def __init__(self) -> None:
        super().__init__(
            'agent_active_run_exists', '当前 session 仍有活动实验', status_code=409,
            retryable=True, allowed_actions=('observe_ml_experiment', 'inspect_ml_session'),
        )


class AgentRunBudgetExhausted(AgentDomainError):
    def __init__(self) -> None:
        super().__init__('agent_run_budget_exhausted', '当前 session 已达到 max_runs 上限', status_code=409)


class AgentIdempotencyConflict(AgentDomainError):
    def __init__(self) -> None:
        super().__init__('agent_idempotency_conflict', 'client_request_id 已用于不同请求', status_code=409)


@dataclass(frozen=True)
class AgentSessionRecord:
    session_id: str
    state: str
    dataset_id: str
    selection_metric: str
    allowed_models: tuple[str, ...]
    max_runs: int
    seed: int
    evaluation_config: dict[str, Any]
    modules: tuple[str, ...]
    context_policy: dict[str, Any]
    selected_run_id: str | None
    created_at: str | None
    finalized_at: str | None
    owner_id: str | None
    tenant_id: str | None


@dataclass(frozen=True)
class AgentExperimentRecord:
    experiment_id: str
    session_id: str
    run_id: str | None
    attempt: int
    state: str
    parent_run_id: str | None
    action_json: dict[str, Any]
    rationale: str | None
    config_hash: str
    client_request_id: str | None
    failure_code: str | None
    created_at: str | None
    updated_at: str | None
    protocol_version: str | None
    last_reconciled_at: str | None
    reconcile_attempt_count: int
    resolution_code: str | None
    owner_id: str | None
    tenant_id: str | None


class AgentSessionRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA journal_mode=WAL')
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                '''
                CREATE TABLE IF NOT EXISTS agent_sessions_v1 (
                    session_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL CHECK(state IN ('open','finalized')),
                    dataset_id TEXT NOT NULL,
                    selection_metric TEXT NOT NULL,
                    allowed_models_json TEXT NOT NULL,
                    max_runs INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    evaluation_config_json TEXT NOT NULL,
                    modules_json TEXT NOT NULL DEFAULT '[]',
                    context_policy_json TEXT NOT NULL DEFAULT '{}',
                    selected_run_id TEXT,
                    client_request_id TEXT,
                    payload_hash TEXT,
                    created_at TEXT NOT NULL,
                    finalized_at TEXT,
                    owner_id TEXT,
                    tenant_id TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS ux_agent_session_request_v1
                ON agent_sessions_v1(COALESCE(owner_id,''), COALESCE(tenant_id,''), client_request_id)
                WHERE client_request_id IS NOT NULL;

                CREATE TABLE IF NOT EXISTS agent_experiment_reservations_v1 (
                    reservation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN
                        ('reserved','bound','released','compensation_required')),
                    run_id TEXT,
                    attempt INTEGER NOT NULL,
                    parent_run_id TEXT,
                    action_json TEXT NOT NULL,
                    rationale TEXT,
                    config_hash TEXT NOT NULL,
                    client_request_id TEXT,
                    payload_hash TEXT,
                    failure_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    protocol_version TEXT,
                    last_reconciled_at TEXT,
                    reconcile_attempt_count INTEGER NOT NULL DEFAULT 0,
                    resolution_code TEXT,
                    owner_id TEXT,
                    tenant_id TEXT,
                    UNIQUE(run_id)
                );
                CREATE INDEX IF NOT EXISTS ix_agent_reservation_session_v1
                ON agent_experiment_reservations_v1(session_id, attempt);
                CREATE UNIQUE INDEX IF NOT EXISTS ux_agent_experiment_request_v1
                ON agent_experiment_reservations_v1(
                    session_id, COALESCE(owner_id,''), COALESCE(tenant_id,''), client_request_id
                ) WHERE client_request_id IS NOT NULL;
                '''
            )
            reservation_columns = {
                str(row['name'])
                for row in connection.execute(
                    'PRAGMA table_info(agent_experiment_reservations_v1)'
                ).fetchall()
            }
            reservation_migrations = {
                'protocol_version': (
                    'ALTER TABLE agent_experiment_reservations_v1 '
                    'ADD COLUMN protocol_version TEXT'
                ),
                'last_reconciled_at': (
                    'ALTER TABLE agent_experiment_reservations_v1 '
                    'ADD COLUMN last_reconciled_at TEXT'
                ),
                'reconcile_attempt_count': (
                    'ALTER TABLE agent_experiment_reservations_v1 '
                    'ADD COLUMN reconcile_attempt_count INTEGER NOT NULL DEFAULT 0'
                ),
                'resolution_code': (
                    'ALTER TABLE agent_experiment_reservations_v1 '
                    'ADD COLUMN resolution_code TEXT'
                ),
            }
            for name, statement in reservation_migrations.items():
                if name not in reservation_columns:
                    connection.execute(statement)
            # 从概念验证版表做只增不改的兼容迁移；旧表继续保留，便于回滚读取。
            legacy_session = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_sessions'"
            ).fetchone()
            if legacy_session is not None:
                connection.execute(
                    '''INSERT OR IGNORE INTO agent_sessions_v1(
                       session_id,state,dataset_id,selection_metric,allowed_models_json,max_runs,seed,
                       evaluation_config_json,modules_json,context_policy_json,selected_run_id,
                       created_at,finalized_at,owner_id,tenant_id)
                       SELECT session_id,state,dataset_id,selection_metric,allowed_models_json,max_runs,seed,
                       evaluation_config_json,'[]','{}',selected_run_id,created_at,finalized_at,
                       owner_id,tenant_id FROM agent_sessions'''
                )
            legacy_experiment = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_experiments'"
            ).fetchone()
            if legacy_experiment is not None:
                connection.execute(
                    '''INSERT OR IGNORE INTO agent_experiment_reservations_v1(
                       reservation_id,session_id,state,run_id,attempt,parent_run_id,action_json,
                       rationale,config_hash,created_at,updated_at,owner_id,tenant_id)
                       SELECT experiment_id,session_id,'bound',run_id,attempt,parent_run_id,action_json,
                       rationale,config_hash,created_at,created_at,owner_id,tenant_id
                       FROM agent_experiments'''
                )

    @staticmethod
    def _scope(principal: Principal) -> tuple[str | None, str | None]:
        return principal.owner_id, principal.tenant_id

    @staticmethod
    def _session(row: sqlite3.Row) -> AgentSessionRecord:
        return AgentSessionRecord(
            session_id=row['session_id'], state=row['state'], dataset_id=row['dataset_id'],
            selection_metric=row['selection_metric'],
            allowed_models=tuple(_decode(row['allowed_models_json'], [])), max_runs=int(row['max_runs']),
            seed=int(row['seed']), evaluation_config=dict(_decode(row['evaluation_config_json'], {})),
            modules=tuple(_decode(row['modules_json'], [])),
            context_policy=dict(_decode(row['context_policy_json'], {})),
            selected_run_id=row['selected_run_id'], created_at=row['created_at'],
            finalized_at=row['finalized_at'], owner_id=row['owner_id'], tenant_id=row['tenant_id'],
        )

    @staticmethod
    def _experiment(row: sqlite3.Row) -> AgentExperimentRecord:
        return AgentExperimentRecord(
            experiment_id=row['reservation_id'], session_id=row['session_id'], run_id=row['run_id'],
            attempt=int(row['attempt']), state=row['state'], parent_run_id=row['parent_run_id'],
            action_json=dict(_decode(row['action_json'], {})), rationale=row['rationale'],
            config_hash=row['config_hash'], client_request_id=row['client_request_id'],
            failure_code=row['failure_code'], created_at=row['created_at'],
            updated_at=row['updated_at'], protocol_version=row['protocol_version'],
            last_reconciled_at=row['last_reconciled_at'],
            reconcile_attempt_count=int(row['reconcile_attempt_count'] or 0),
            resolution_code=row['resolution_code'],
            owner_id=row['owner_id'], tenant_id=row['tenant_id'],
        )

    def create_session(self, *, dataset_id: str, selection_metric: str, allowed_models: list[str],
                       max_runs: int, seed: int, evaluation_config: dict[str, Any],
                       modules: list[str], context_policy: dict[str, Any],
                       client_request_id: str | None, payload_hash: str,
                       principal: Principal) -> tuple[AgentSessionRecord, bool]:
        owner, tenant = self._scope(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            if client_request_id:
                row = connection.execute(
                    '''SELECT * FROM agent_sessions_v1 WHERE client_request_id = ?
                       AND owner_id IS ? AND tenant_id IS ?''', (client_request_id, owner, tenant)
                ).fetchone()
                if row is not None:
                    if row['payload_hash'] != payload_hash:
                        connection.rollback(); raise AgentIdempotencyConflict()
                    connection.commit(); return self._session(row), False
            session_id, now = uuid.uuid4().hex, _timestamp()
            connection.execute(
                '''INSERT INTO agent_sessions_v1(
                    session_id,state,dataset_id,selection_metric,allowed_models_json,max_runs,seed,
                    evaluation_config_json,modules_json,context_policy_json,client_request_id,payload_hash,
                    created_at,owner_id,tenant_id
                ) VALUES(?,'open',?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (session_id, dataset_id, selection_metric, _json(allowed_models), max_runs, seed,
                 _json(evaluation_config), _json(modules), _json(context_policy), client_request_id,
                 payload_hash, now, owner, tenant),
            )
            row = connection.execute('SELECT * FROM agent_sessions_v1 WHERE session_id=?', (session_id,)).fetchone()
            connection.commit()
        return self._session(row), True

    def get_session_scoped(self, session_id: str, *, principal: Principal) -> AgentSessionRecord:
        owner, tenant = self._scope(principal)
        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM agent_sessions_v1 WHERE session_id=? AND owner_id IS ? AND tenant_id IS ?',
                (session_id, owner, tenant),
            ).fetchone()
        if row is None: raise AgentSessionNotFound()
        return self._session(row)

    def list_experiments_scoped(self, session_id: str, *, principal: Principal,
                                include_released: bool = True) -> list[AgentExperimentRecord]:
        owner, tenant = self._scope(principal)
        clause = '' if include_released else " AND state != 'released'"
        with self._connection() as connection:
            rows = connection.execute(
                '''SELECT * FROM agent_experiment_reservations_v1
                   WHERE session_id=? AND owner_id IS ? AND tenant_id IS ?''' + clause +
                ' ORDER BY attempt,reservation_id', (session_id, owner, tenant)
            ).fetchall()
        return [self._experiment(row) for row in rows]

    def get_experiment_scoped(self, session_id: str, run_id: str, *, principal: Principal) -> AgentExperimentRecord:
        owner, tenant = self._scope(principal)
        with self._connection() as connection:
            row = connection.execute(
                '''SELECT * FROM agent_experiment_reservations_v1 WHERE session_id=? AND run_id=?
                   AND owner_id IS ? AND tenant_id IS ? AND state IN ('bound','compensation_required')''',
                (session_id, run_id, owner, tenant),
            ).fetchone()
        if row is None: raise AgentExperimentNotFound()
        return self._experiment(row)

    def reserve_experiment(self, *, session_id: str, action_json: dict[str, Any], rationale: str | None,
                           parent_run_id: str | None, config_hash: str, client_request_id: str | None,
                           payload_hash: str, active_run_ids: set[str], principal: Principal,
                           protocol_version: str | None = None,
                           ) -> tuple[AgentExperimentRecord, bool]:
        owner, tenant = self._scope(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            session_row = connection.execute(
                'SELECT * FROM agent_sessions_v1 WHERE session_id=? AND owner_id IS ? AND tenant_id IS ?',
                (session_id, owner, tenant),
            ).fetchone()
            if session_row is None: connection.rollback(); raise AgentSessionNotFound()
            if session_row['state'] != 'open': connection.rollback(); raise AgentSessionClosed()
            if client_request_id:
                row = connection.execute(
                    '''SELECT * FROM agent_experiment_reservations_v1 WHERE session_id=?
                       AND client_request_id=? AND owner_id IS ? AND tenant_id IS ?''',
                    (session_id, client_request_id, owner, tenant),
                ).fetchone()
                if row is not None:
                    if row['payload_hash'] != payload_hash:
                        connection.rollback(); raise AgentIdempotencyConflict()
                    connection.commit(); return self._experiment(row), False
            rows = connection.execute(
                '''SELECT * FROM agent_experiment_reservations_v1 WHERE session_id=?
                   AND owner_id IS ? AND tenant_id IS ?''', (session_id, owner, tenant)
            ).fetchall()
            if any(row['config_hash'] == config_hash and row['state'] != 'released' for row in rows):
                connection.rollback(); raise AgentConfigCollision()
            if any(row['state'] in ('reserved','compensation_required') or
                   (row['state']=='bound' and row['run_id'] in active_run_ids) for row in rows):
                connection.rollback(); raise AgentActiveRunExists()
            active_rows = [row for row in rows if row['state'] != 'released']
            if len(active_rows) >= int(session_row['max_runs']):
                connection.rollback(); raise AgentRunBudgetExhausted()
            if parent_run_id and not any(row['run_id'] == parent_run_id and row['state']=='bound' for row in rows):
                connection.rollback()
                raise AgentDomainError('agent_invalid_action', 'parent_run_id 不属于当前 session', status_code=422)
            now = _timestamp()
            # reservation_id 同时是永久 submission identity；released 历史行绝不复用。
            reservation_id = uuid.uuid4().hex
            attempt = max([int(row['attempt']) for row in rows], default=0) + 1
            connection.execute(
                '''INSERT INTO agent_experiment_reservations_v1(
                   reservation_id,session_id,state,attempt,parent_run_id,action_json,rationale,
                   config_hash,client_request_id,payload_hash,created_at,updated_at,protocol_version,
                   owner_id,tenant_id
                   ) VALUES(?,?,'reserved',?,?,?,?,?,?,?,?,?,?,?,?)''',
                (reservation_id, session_id, attempt, parent_run_id, _json(action_json), rationale,
                 config_hash, client_request_id, payload_hash, now, now, protocol_version,
                 owner, tenant),
            )
            row = connection.execute(
                'SELECT * FROM agent_experiment_reservations_v1 WHERE reservation_id=?', (reservation_id,)
            ).fetchone()
            connection.commit()
        return self._experiment(row), True

    def bind_experiment(self, reservation_id: str, *, run_id: str, principal: Principal) -> AgentExperimentRecord:
        owner, tenant = self._scope(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            changed = connection.execute(
                '''UPDATE agent_experiment_reservations_v1 SET state='bound',run_id=?,updated_at=?
                   WHERE reservation_id=? AND state='reserved' AND owner_id IS ? AND tenant_id IS ?''',
                (run_id, _timestamp(), reservation_id, owner, tenant),
            ).rowcount
            if changed != 1: connection.rollback(); raise AgentExperimentNotFound()
            row = connection.execute(
                'SELECT * FROM agent_experiment_reservations_v1 WHERE reservation_id=?', (reservation_id,)
            ).fetchone(); connection.commit()
        return self._experiment(row)

    def get_reservation_scoped(self, reservation_id: str, *, principal: Principal) -> AgentExperimentRecord:
        owner, tenant = self._scope(principal)
        with self._connection() as connection:
            row = connection.execute(
                '''SELECT * FROM agent_experiment_reservations_v1 WHERE reservation_id=?
                   AND owner_id IS ? AND tenant_id IS ?''',
                (reservation_id, owner, tenant),
            ).fetchone()
        if row is None: raise AgentExperimentNotFound()
        return self._experiment(row)

    def release_reservation(self, reservation_id: str, *, failure_code: str, principal: Principal) -> None:
        owner, tenant = self._scope(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute(
                '''UPDATE agent_experiment_reservations_v1 SET state='released',failure_code=?,updated_at=?
                   WHERE reservation_id=? AND state='reserved'
                   AND owner_id IS ? AND tenant_id IS ?''',
                (failure_code, _timestamp(), reservation_id, owner, tenant),
            ); connection.commit()

    def require_compensation(self, reservation_id: str, *, run_id: str | None,
                             failure_code: str, principal: Principal) -> None:
        owner, tenant = self._scope(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute(
                '''UPDATE agent_experiment_reservations_v1 SET state='compensation_required',
                   run_id=COALESCE(run_id,?),failure_code=?,updated_at=?
                   WHERE reservation_id=? AND state='reserved'
                   AND owner_id IS ? AND tenant_id IS ?''',
                (run_id, failure_code, _timestamp(), reservation_id, owner, tenant),
            ); connection.commit()

    def reconcile_transition(
        self,
        reservation_id: str,
        *,
        expected_state: str,
        expected_updated_at: str | None,
        new_state: str,
        run_id: str | None,
        resolution_code: str,
        reconciled_at: datetime,
        principal: Principal,
        increment_attempt: bool = True,
    ) -> tuple[AgentExperimentRecord, bool]:
        """用 state + updated_at 条件更新单条 reservation，并发失败后重读。"""
        allowed = {
            'reserved': {'released', 'bound', 'compensation_required'},
            'compensation_required': {'released', 'bound', 'compensation_required'},
        }
        if new_state not in allowed.get(expected_state, set()):
            raise ValueError('invalid reconciliation transition')
        owner, tenant = self._scope(principal)
        now_text = _timestamp(reconciled_at)
        increment = 1 if increment_attempt else 0
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            changed = connection.execute(
                '''UPDATE agent_experiment_reservations_v1
                   SET state=?,run_id=COALESCE(?,run_id),last_reconciled_at=?,
                       reconcile_attempt_count=reconcile_attempt_count+?,resolution_code=?,
                       updated_at=?
                   WHERE reservation_id=? AND state=? AND updated_at IS ?
                   AND owner_id IS ? AND tenant_id IS ?''',
                (
                    new_state, run_id, now_text, increment, resolution_code, now_text,
                    reservation_id, expected_state, expected_updated_at, owner, tenant,
                ),
            ).rowcount
            row = connection.execute(
                '''SELECT * FROM agent_experiment_reservations_v1
                   WHERE reservation_id=? AND owner_id IS ? AND tenant_id IS ?''',
                (reservation_id, owner, tenant),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise AgentExperimentNotFound()
            connection.commit()
        return self._experiment(row), changed == 1

    def count_budget_scoped(self, *, session_id: str, principal: Principal) -> int:
        return sum(1 for item in self.list_experiments_scoped(session_id, principal=principal)
                   if item.state != 'released')

    # 概念验证版兼容别名。
    count_experiments_scoped = count_budget_scoped

    def finalize_session(self, *, session_id: str, selected_run_id: str,
                         principal: Principal) -> AgentSessionRecord:
        owner, tenant = self._scope(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                'SELECT * FROM agent_sessions_v1 WHERE session_id=? AND owner_id IS ? AND tenant_id IS ?',
                (session_id, owner, tenant),
            ).fetchone()
            if row is None: connection.rollback(); raise AgentSessionNotFound()
            if row['state']=='finalized':
                if row['selected_run_id'] != selected_run_id:
                    connection.rollback(); raise AgentSessionClosed('session 已由另一个 run finalize')
                connection.commit(); return self._session(row)
            now = _timestamp()
            connection.execute(
                "UPDATE agent_sessions_v1 SET state='finalized',selected_run_id=?,finalized_at=? WHERE session_id=?",
                (selected_run_id, now, session_id),
            )
            row = connection.execute('SELECT * FROM agent_sessions_v1 WHERE session_id=?', (session_id,)).fetchone()
            connection.commit()
        return self._session(row)


def collect_run_states(repository: RunRepository, run_ids: list[str]) -> dict[str, str]:
    states: dict[str, str] = {}
    for run_id in run_ids:
        try: states[run_id] = repository.get(run_id).state
        except RunNotFound: states[run_id] = 'missing'
    return states
