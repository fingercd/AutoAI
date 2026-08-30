"""Agent Session / Experiment 持久化（SQLite）。

所有 Session 与 Experiment 状态都落 `storage/agent.sqlite3`；建表与轻量迁移
走 ``PRAGMA table_info`` 探测 + ``ALTER TABLE`` 补列，与 ``RunRepository.initialize``
保持同一种迁移风格（无 schema 版本号），方便在没有版本控制器的项目里幂等升级。

设计约束：
1. Session 与 Experiment 互引用：Experiment.session_id + Experiment.run_id 必须
   始终可被 Principal scope 校验（Agent 接口从不在请求体接收身份）。
2. Session 状态机：open → finalized（仅一次）。Finalize 不删数据，只锁住 Session。
4. 所有写操作均在 ``BEGIN IMMEDIATE`` 事务内做状态校验，确保并发提交实验
   不会越过 ``max_runs`` 或混入 finalized Session。
"""

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
from ..runs.repository import RunRepository, RunNotFound
from .contracts import AgentContextPolicy, AgentModuleFlags
from .memory import (
    AgentMemoryPayloadError,
    MAX_MEMORY_CONTEXT_ITEMS,
    build_memory_record,
    normalize_memory_record,
    normalize_terminal_outcome,
)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _decode_json(value: str | None, default: object) -> object:
    if not value:
        return default
    return json.loads(value)


def _validate_memory_signature(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in '0123456789abcdef' for character in normalized
    ):
        raise ValueError('memory signature must be a sha256 hex digest')
    return normalized


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )


_DEFAULT_MODULE_FLAGS = AgentModuleFlags().model_dump()
_DEFAULT_CONTEXT_POLICY = AgentContextPolicy().model_dump()


class AgentSessionNotFound(KeyError):
    pass


class AgentExperimentNotFound(KeyError):
    pass


class AgentSessionClosed(RuntimeError):
    pass


class AgentConfigCollision(RuntimeError):
    """同 Session 下重复 effective config 或违反 max_runs。"""


class AgentBudgetExceeded(AgentConfigCollision):
    """A hard Session budget would be exceeded by a new reservation."""


class AgentBudgetAccountingError(RuntimeError):
    """Persisted budget accounting is invalid or internally inconsistent."""


class AgentMemoryWriteDenied(RuntimeError):
    """Session policy does not authorize persistent Agent memory writes."""


class AgentMemoryCollision(RuntimeError):
    """A terminal source was already persisted with incompatible state."""


# Agent Session 完整规范快照。frozen=True 防止读取后被偷偷改。
@dataclass(frozen=True)
class AgentSessionRecord:
    session_id: str
    state: str  # open / finalized
    dataset_id: str
    selection_metric: str
    allowed_models: tuple[str, ...]
    max_runs: int
    seed: int
    evaluation_config: dict[str, Any]
    module_flags: dict[str, bool]
    context_policy: dict[str, Any]
    context: dict[str, Any]
    selected_run_id: str | None
    created_at: str | None
    finalized_at: str | None
    owner_id: str | None
    tenant_id: str | None


@dataclass(frozen=True)
class AgentExperimentRecord:
    experiment_id: str
    session_id: str
    run_id: str
    attempt: int
    parent_run_id: str | None
    action_json: dict[str, Any]
    rationale: str | None
    config_hash: str
    reserved_model_fits: int
    actual_model_fits: int | None
    settled: bool
    created_at: str | None
    owner_id: str | None
    tenant_id: str | None


@dataclass(frozen=True)
class AgentExperimentReservation:
    """跨 HTTP 请求的实验入队 reservation。

    Reservation 先于 Run 入队写入 Agent DB，用来把 max_runs、重复配置和
    “同一 Session 只能有一个活动提交”放进同一把 SQLite 写锁里。
    """

    reservation_id: str
    session_id: str
    attempt: int
    config_hash: str
    action_json: dict[str, Any]
    reserved_model_fits: int
    created_at: str | None
    owner_id: str | None
    tenant_id: str | None


@dataclass(frozen=True)
class AgentTerminalOutcomeRecord:
    """Internal idempotency record; identifiers are never Agent-visible."""

    outcome_id: str
    session_id: str
    run_id: str
    state: str
    task_signature: str
    evidence_signature: str
    payload: dict[str, Any]
    selected: bool
    recorded_at: str | None
    owner_id: str | None
    tenant_id: str | None


@dataclass(frozen=True)
class AgentMemoryRecord:
    """Internal case/failure row; only ``payload`` may enter frozen context."""

    memory_id: str
    source_outcome_id: str
    record_type: str
    task_signature: str
    evidence_signature: str
    payload: dict[str, Any]
    created_at: str | None
    owner_id: str | None
    tenant_id: str | None


class AgentSessionRepository:
    """Agent Session / Experiment 仓库。

    与 ``RunRepository`` 共用相同的 WAL+外键配置，保证在并发 worker 抢 Run
    时 Agent 表也能被独立事务安全读写。两张表之间用 ``run_id`` 软引用关联，
    没有强外键——这样删除 Run 不会反向牵连 Session 数据。
    """

    def __init__(
        self,
        database_path: Path,
        *,
        runs_database_path: Path | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.runs_database_path = (
            Path(runs_database_path) if runs_database_path is not None else None
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA busy_timeout=30000')
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """幂等建表 + 轻量列迁移。"""
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL CHECK (state IN ('open', 'finalized')),
                    dataset_id TEXT NOT NULL,
                    selection_metric TEXT NOT NULL,
                    allowed_models_json TEXT NOT NULL,
                    max_runs INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    evaluation_config_json TEXT NOT NULL,
                    module_flags_json TEXT NOT NULL DEFAULT '{}',
                    context_policy_json TEXT NOT NULL DEFAULT '{}',
                    context_json TEXT NOT NULL DEFAULT '{}',
                    selected_run_id TEXT,
                    created_at TEXT NOT NULL,
                    finalized_at TEXT,
                    owner_id TEXT,
                    tenant_id TEXT
                )
                '''
            )
            session_columns = {
                str(row['name'])
                for row in connection.execute('PRAGMA table_info(agent_sessions)').fetchall()
            }
            session_migrations = {
                'selected_run_id': 'ALTER TABLE agent_sessions ADD COLUMN selected_run_id TEXT',
                'finalized_at': 'ALTER TABLE agent_sessions ADD COLUMN finalized_at TEXT',
                'owner_id': 'ALTER TABLE agent_sessions ADD COLUMN owner_id TEXT',
                'tenant_id': 'ALTER TABLE agent_sessions ADD COLUMN tenant_id TEXT',
                'module_flags_json': (
                    "ALTER TABLE agent_sessions ADD COLUMN "
                    "module_flags_json TEXT NOT NULL DEFAULT '{}'"
                ),
                'context_policy_json': (
                    "ALTER TABLE agent_sessions ADD COLUMN "
                    "context_policy_json TEXT NOT NULL DEFAULT '{}'"
                ),
                'context_json': (
                    "ALTER TABLE agent_sessions ADD COLUMN "
                    "context_json TEXT NOT NULL DEFAULT '{}'"
                ),
            }
            for name, statement in session_migrations.items():
                if name not in session_columns:
                    connection.execute(statement)

            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS agent_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    parent_run_id TEXT,
                    action_json TEXT NOT NULL,
                    rationale TEXT,
                    config_hash TEXT NOT NULL,
                    reserved_model_fits INTEGER NOT NULL DEFAULT 0,
                    actual_model_fits INTEGER,
                    settled INTEGER NOT NULL DEFAULT 0
                        CHECK (settled IN (0, 1)),
                    created_at TEXT NOT NULL,
                    owner_id TEXT,
                    tenant_id TEXT,
                    UNIQUE (session_id, config_hash),
                    UNIQUE (run_id)
                )
                '''
            )
            experiment_columns = {
                str(row['name'])
                for row in connection.execute('PRAGMA table_info(agent_experiments)').fetchall()
            }
            experiment_migrations = {
                'owner_id': 'ALTER TABLE agent_experiments ADD COLUMN owner_id TEXT',
                'tenant_id': 'ALTER TABLE agent_experiments ADD COLUMN tenant_id TEXT',
                'rationale': 'ALTER TABLE agent_experiments ADD COLUMN rationale TEXT',
                'parent_run_id': 'ALTER TABLE agent_experiments ADD COLUMN parent_run_id TEXT',
                'attempt': "ALTER TABLE agent_experiments ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0",
                'reserved_model_fits': (
                    'ALTER TABLE agent_experiments ADD COLUMN '
                    'reserved_model_fits INTEGER NOT NULL DEFAULT 0'
                ),
                'actual_model_fits': (
                    'ALTER TABLE agent_experiments ADD COLUMN actual_model_fits INTEGER'
                ),
                'settled': (
                    'ALTER TABLE agent_experiments ADD COLUMN '
                    'settled INTEGER NOT NULL DEFAULT 0'
                ),
            }
            for name, statement in experiment_migrations.items():
                if name not in experiment_columns:
                    connection.execute(statement)
            connection.execute(
                'CREATE INDEX IF NOT EXISTS idx_agent_experiments_session '
                'ON agent_experiments(session_id, attempt)'
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS agent_experiment_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    config_hash TEXT NOT NULL,
                    action_json TEXT NOT NULL,
                    reserved_model_fits INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    owner_id TEXT,
                    tenant_id TEXT,
                    UNIQUE(session_id, config_hash),
                    UNIQUE(session_id, attempt)
                )
                '''
            )
            reservation_columns = {
                str(row['name'])
                for row in connection.execute(
                    'PRAGMA table_info(agent_experiment_reservations)'
                ).fetchall()
            }
            if 'reserved_model_fits' not in reservation_columns:
                connection.execute(
                    'ALTER TABLE agent_experiment_reservations ADD COLUMN '
                    'reserved_model_fits INTEGER NOT NULL DEFAULT 0'
                )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS idx_agent_reservations_session '
                'ON agent_experiment_reservations(session_id, attempt)'
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS agent_terminal_outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    state TEXT NOT NULL
                        CHECK (state IN ('succeeded', 'failed', 'cancelled')),
                    task_signature TEXT NOT NULL,
                    evidence_signature TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    selected INTEGER NOT NULL DEFAULT 0
                        CHECK (selected IN (0, 1)),
                    recorded_at TEXT NOT NULL,
                    owner_id TEXT,
                    tenant_id TEXT,
                    UNIQUE(session_id, run_id)
                )
                '''
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS idx_agent_outcomes_scope '
                'ON agent_terminal_outcomes(owner_id, tenant_id, session_id)'
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS agent_memory_records (
                    memory_id TEXT PRIMARY KEY,
                    source_outcome_id TEXT NOT NULL,
                    record_type TEXT NOT NULL
                        CHECK (record_type IN ('case', 'failure')),
                    task_signature TEXT NOT NULL,
                    evidence_signature TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    owner_id TEXT,
                    tenant_id TEXT,
                    UNIQUE(source_outcome_id, record_type)
                )
                '''
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS idx_agent_memory_retrieval '
                'ON agent_memory_records('
                'owner_id, tenant_id, task_signature, created_at DESC, memory_id)'
            )
            connection.commit()

    @staticmethod
    def _scope_values(principal: Principal) -> tuple[str | None, str | None]:
        return principal.owner_id, principal.tenant_id

    @staticmethod
    def _session(row: sqlite3.Row) -> AgentSessionRecord:
        stored_flags = dict(_decode_json(
            row['module_flags_json'] if 'module_flags_json' in row.keys() else None,
            {},
        ))
        module_flags = {**_DEFAULT_MODULE_FLAGS, **stored_flags}
        stored_policy = dict(_decode_json(
            row['context_policy_json'] if 'context_policy_json' in row.keys() else None,
            {},
        ))
        context_policy = {**_DEFAULT_CONTEXT_POLICY, **stored_policy}
        default_context = {
            'schema_version': 'agent-context-v1',
            'status': 'pending' if any(module_flags.values()) else 'disabled',
            'source_role': context_policy['source_role'],
        }
        stored_context = dict(_decode_json(
            row['context_json'] if 'context_json' in row.keys() else None,
            {},
        ))
        return AgentSessionRecord(
            session_id=str(row['session_id']),
            state=row['state'],
            dataset_id=row['dataset_id'],
            selection_metric=row['selection_metric'],
            allowed_models=tuple(json.loads(row['allowed_models_json'])),
            max_runs=int(row['max_runs']),
            seed=int(row['seed']),
            evaluation_config=dict(_decode_json(row['evaluation_config_json'], {})),
            module_flags=module_flags,
            context_policy=context_policy,
            context={**default_context, **stored_context},
            selected_run_id=row['selected_run_id'],
            created_at=row['created_at'],
            finalized_at=row['finalized_at'],
            owner_id=row['owner_id'] if 'owner_id' in row.keys() else None,
            tenant_id=row['tenant_id'] if 'tenant_id' in row.keys() else None,
        )

    @staticmethod
    def _experiment(row: sqlite3.Row) -> AgentExperimentRecord:
        return AgentExperimentRecord(
            experiment_id=str(row['experiment_id']),
            session_id=str(row['session_id']),
            run_id=str(row['run_id']),
            attempt=int(row['attempt']),
            parent_run_id=row['parent_run_id'],
            action_json=dict(_decode_json(row['action_json'], {})),
            rationale=row['rationale'],
            config_hash=row['config_hash'],
            reserved_model_fits=int(row['reserved_model_fits']),
            actual_model_fits=(
                int(row['actual_model_fits'])
                if row['actual_model_fits'] is not None
                else None
            ),
            settled=bool(row['settled']),
            created_at=row['created_at'],
            owner_id=row['owner_id'] if 'owner_id' in row.keys() else None,
            tenant_id=row['tenant_id'] if 'tenant_id' in row.keys() else None,
        )

    @staticmethod
    def _reservation(row: sqlite3.Row) -> AgentExperimentReservation:
        return AgentExperimentReservation(
            reservation_id=str(row['reservation_id']),
            session_id=str(row['session_id']),
            attempt=int(row['attempt']),
            config_hash=str(row['config_hash']),
            action_json=dict(_decode_json(row['action_json'], {})),
            reserved_model_fits=int(row['reserved_model_fits']),
            created_at=row['created_at'],
            owner_id=row['owner_id'] if 'owner_id' in row.keys() else None,
            tenant_id=row['tenant_id'] if 'tenant_id' in row.keys() else None,
        )

    @staticmethod
    def _terminal_outcome(row: sqlite3.Row) -> AgentTerminalOutcomeRecord:
        return AgentTerminalOutcomeRecord(
            outcome_id=str(row['outcome_id']),
            session_id=str(row['session_id']),
            run_id=str(row['run_id']),
            state=str(row['state']),
            task_signature=str(row['task_signature']),
            evidence_signature=str(row['evidence_signature']),
            payload=dict(_decode_json(row['payload_json'], {})),
            selected=bool(row['selected']),
            recorded_at=row['recorded_at'],
            owner_id=row['owner_id'],
            tenant_id=row['tenant_id'],
        )

    @staticmethod
    def _memory_record(row: sqlite3.Row) -> AgentMemoryRecord:
        return AgentMemoryRecord(
            memory_id=str(row['memory_id']),
            source_outcome_id=str(row['source_outcome_id']),
            record_type=str(row['record_type']),
            task_signature=str(row['task_signature']),
            evidence_signature=str(row['evidence_signature']),
            payload=dict(_decode_json(row['payload_json'], {})),
            created_at=row['created_at'],
            owner_id=row['owner_id'],
            tenant_id=row['tenant_id'],
        )

    @staticmethod
    def _model_fit_budget_limit(session_row: sqlite3.Row) -> int | None:
        """Read the immutable model-fit limit from a locked Session context."""
        flags = dict(_decode_json(session_row['module_flags_json'], {}))
        if flags.get('budget_control') is not True:
            return None
        context = dict(_decode_json(session_row['context_json'], {}))
        policy = context.get('budget_policy')
        limits = policy.get('limits') if isinstance(policy, dict) else None
        maximum = limits.get('max_model_fits') if isinstance(limits, dict) else None
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 1 <= maximum <= 1000
        ):
            raise AgentBudgetAccountingError(
                'budget_control Session is missing a valid max_model_fits limit'
            )
        return maximum

    @staticmethod
    def _model_fit_usage_locked(
        connection: sqlite3.Connection,
        *,
        session_id: str,
    ) -> tuple[int, int]:
        """Return ``(settled actual, still reserved)`` under the caller's lock."""
        experiment_row = connection.execute(
            '''
            SELECT
                COALESCE(SUM(
                    CASE WHEN settled = 1 THEN COALESCE(actual_model_fits, 0)
                         ELSE 0 END
                ), 0) AS actual,
                COALESCE(SUM(
                    CASE WHEN settled = 0 THEN reserved_model_fits ELSE 0 END
                ), 0) AS reserved
            FROM agent_experiments
            WHERE session_id = ?
            ''',
            (session_id,),
        ).fetchone()
        reservation_row = connection.execute(
            '''
            SELECT COALESCE(SUM(reserved_model_fits), 0) AS reserved
            FROM agent_experiment_reservations
            WHERE session_id = ?
            ''',
            (session_id,),
        ).fetchone()
        actual = int(experiment_row['actual']) if experiment_row is not None else 0
        reserved = (
            int(experiment_row['reserved']) if experiment_row is not None else 0
        ) + (
            int(reservation_row['reserved']) if reservation_row is not None else 0
        )
        if actual < 0 or reserved < 0:
            raise AgentBudgetAccountingError('negative model-fit accounting')
        return actual, reserved

    def create_session(
        self,
        *,
        dataset_id: str,
        selection_metric: str,
        allowed_models: list[str],
        max_runs: int,
        seed: int,
        evaluation_config: dict[str, Any],
        principal: Principal,
        module_flags: dict[str, bool] | None = None,
        context_policy: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AgentSessionRecord:
        session_id = uuid.uuid4().hex
        now = _timestamp(datetime.now(timezone.utc))
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute(
                '''
                INSERT INTO agent_sessions (
                    session_id, state, dataset_id, selection_metric,
                    allowed_models_json, max_runs, seed,
                    evaluation_config_json, module_flags_json,
                    context_policy_json, context_json,
                    created_at, owner_id, tenant_id
                ) VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    session_id,
                    dataset_id,
                    selection_metric,
                    json.dumps(list(allowed_models), ensure_ascii=False),
                    int(max_runs),
                    int(seed),
                    json.dumps(evaluation_config, ensure_ascii=False, sort_keys=True),
                    json.dumps(module_flags or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(context_policy or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(context or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    owner_id,
                    tenant_id,
                ),
            )
            row = connection.execute(
                'SELECT * FROM agent_sessions WHERE session_id = ?', (session_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        return self._session(row)

    def update_context_scoped(
        self,
        session_id: str,
        *,
        context: dict[str, Any],
        principal: Principal,
    ) -> AgentSessionRecord:
        """Replace the server-owned context snapshot under Principal scope."""
        owner_id, tenant_id = self._scope_values(principal)
        encoded = json.dumps(context, ensure_ascii=False, sort_keys=True)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            cursor = connection.execute(
                '''
                UPDATE agent_sessions SET context_json = ?
                WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (encoded, session_id, owner_id, tenant_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise AgentSessionNotFound(session_id)
            row = connection.execute(
                'SELECT * FROM agent_sessions WHERE session_id = ?', (session_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        return self._session(row)

    def get_session_scoped(
        self, session_id: str, *, principal: Principal
    ) -> AgentSessionRecord:
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            row = connection.execute(
                '''
                SELECT * FROM agent_sessions
                WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (session_id, owner_id, tenant_id),
            ).fetchone()
        if row is None:
            raise AgentSessionNotFound(session_id)
        return self._session(row)

    def list_experiments_scoped(
        self, session_id: str, *, principal: Principal
    ) -> list[AgentExperimentRecord]:
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            rows = connection.execute(
                '''
                SELECT * FROM agent_experiments
                WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                ORDER BY attempt ASC, experiment_id ASC
                ''',
                (session_id, owner_id, tenant_id),
            ).fetchall()
        return [self._experiment(row) for row in rows]

    def get_experiment_scoped(
        self, session_id: str, run_id: str, *, principal: Principal
    ) -> AgentExperimentRecord:
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            row = connection.execute(
                '''
                SELECT * FROM agent_experiments
                WHERE session_id = ? AND run_id = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (session_id, run_id, owner_id, tenant_id),
            ).fetchone()
        if row is None:
            raise AgentExperimentNotFound(run_id)
        return self._experiment(row)

    def find_duplicate_config(
        self, *, session_id: str, config_hash: str, principal: Principal
    ) -> bool:
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            row = connection.execute(
                '''
                SELECT 1 FROM agent_experiments
                WHERE session_id = ? AND config_hash = ? AND owner_id IS ? AND tenant_id IS ?
                LIMIT 1
                ''',
                (session_id, config_hash, owner_id, tenant_id),
            ).fetchone()
        return row is not None

    def count_experiments_scoped(
        self, *, session_id: str, principal: Principal
    ) -> int:
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            row = connection.execute(
                '''
                SELECT COUNT(*) AS cnt FROM agent_experiments
                WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (session_id, owner_id, tenant_id),
            ).fetchone()
        return int(row['cnt']) if row is not None else 0

    def count_reservations_scoped(
        self, *, session_id: str, principal: Principal
    ) -> int:
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            row = connection.execute(
                '''
                SELECT COUNT(*) AS cnt FROM agent_experiment_reservations
                WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (session_id, owner_id, tenant_id),
            ).fetchone()
        return int(row['cnt']) if row is not None else 0

    def reserve_experiment(
        self,
        *,
        session_id: str,
        config_hash: str,
        action_json: dict[str, Any],
        principal: Principal,
        parent_run_id: str | None = None,
        action_id: str | None = None,
        max_retries_per_failure: int | None = None,
        require_replan_for_failed: bool = False,
        reserved_model_fits: int = 0,
    ) -> AgentExperimentReservation:
        """原子预留一次 Experiment，防止并发请求越过 max_runs。

        Session、历史 Experiment 和未绑定 Reservation 的读取均在同一个
        ``BEGIN IMMEDIATE`` 内完成；Run 尚未创建时，其他请求已经能观察到
        这次 Reservation，从而不会重复入队。
        """
        reservation_id = uuid.uuid4().hex
        now = _timestamp(datetime.now(timezone.utc))
        owner_id, tenant_id = self._scope_values(principal)
        if (
            isinstance(reserved_model_fits, bool)
            or not isinstance(reserved_model_fits, int)
            or reserved_model_fits < 0
        ):
            raise AgentBudgetAccountingError('reserved_model_fits must be a non-negative integer')
        with self._connection() as connection:
            if self.runs_database_path is not None:
                connection.execute(
                    'ATTACH DATABASE ? AS run_state_db',
                    (str(self.runs_database_path),),
                )
            connection.execute('BEGIN IMMEDIATE')
            try:
                session_row = connection.execute(
                    '''
                    SELECT * FROM agent_sessions
                    WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (session_id, owner_id, tenant_id),
                ).fetchone()
                if session_row is None:
                    raise AgentSessionNotFound(session_id)
                if session_row['state'] != 'open':
                    raise AgentSessionClosed(
                        f'session {session_id} 已经 finalized，不能继续提交'
                    )

                active_reservation = connection.execute(
                    '''
                    SELECT 1 FROM agent_experiment_reservations
                    WHERE session_id = ?
                    LIMIT 1
                    ''',
                    (session_id,),
                ).fetchone()
                if active_reservation is not None:
                    raise AgentConfigCollision(
                        f'session {session_id} 仍有提交中的 experiment，请稍后重试'
                    )

                if self.runs_database_path is not None:
                    active_run = connection.execute(
                        '''
                        SELECT 1
                        FROM agent_experiments AS experiment
                        JOIN run_state_db.runs AS run
                          ON run.run_id = experiment.run_id
                        WHERE experiment.session_id = ?
                          AND run.state IN ('queued', 'running')
                        LIMIT 1
                        ''',
                        (session_id,),
                    ).fetchone()
                    if active_run is not None:
                        raise AgentConfigCollision(
                            f'session {session_id} 仍有非终态 Run，请等待完成后再提交'
                        )
                    if require_replan_for_failed and action_id is None:
                        unresolved_failure = connection.execute(
                            '''
                            SELECT 1
                            FROM agent_experiments AS failed_experiment
                            JOIN run_state_db.runs AS failed_run
                              ON failed_run.run_id = failed_experiment.run_id
                            WHERE failed_experiment.session_id = ?
                              AND failed_run.state = 'failed'
                              AND NOT EXISTS (
                                  SELECT 1 FROM agent_experiments AS child
                                  WHERE child.session_id = failed_experiment.session_id
                                    AND child.parent_run_id = failed_experiment.run_id
                              )
                            LIMIT 1
                            ''',
                            (session_id,),
                        ).fetchone()
                        if unresolved_failure is not None:
                            raise AgentConfigCollision(
                                '存在未处理失败实验时必须使用受控 REPLAN action'
                            )
                elif require_replan_for_failed:
                    raise AgentConfigCollision('有限重规划需要原子 Run 状态存储')

                duplicate = connection.execute(
                    '''
                    SELECT 1 FROM agent_experiments
                    WHERE session_id = ? AND config_hash = ?
                    UNION ALL
                    SELECT 1 FROM agent_experiment_reservations
                    WHERE session_id = ? AND config_hash = ?
                    LIMIT 1
                    ''',
                    (session_id, config_hash, session_id, config_hash),
                ).fetchone()
                if duplicate is not None:
                    raise AgentConfigCollision(
                        f'config_hash {config_hash} 在该 session 内已存在'
                    )

                if action_id is not None:
                    if (
                        parent_run_id is None
                        or max_retries_per_failure is None
                        or max_retries_per_failure < 1
                    ):
                        raise AgentConfigCollision('有限重规划 reservation 参数无效')
                    prior_actions = connection.execute(
                        '''
                        SELECT action_json FROM agent_experiments
                        WHERE session_id = ? AND parent_run_id = ?
                        ''',
                        (session_id, parent_run_id),
                    ).fetchall()
                    retry_count = sum(
                        1
                        for row in prior_actions
                        if dict(_decode_json(row['action_json'], {})).get('action_id')
                        == action_id
                    )
                    if retry_count >= max_retries_per_failure:
                        raise AgentConfigCollision(
                            '同一失败与 action_id 已达有限重试上限'
                        )

                experiment_count = int(
                    connection.execute(
                        'SELECT COUNT(*) FROM agent_experiments WHERE session_id = ?',
                        (session_id,),
                    ).fetchone()[0]
                )
                reservation_count = int(
                    connection.execute(
                        'SELECT COUNT(*) FROM agent_experiment_reservations WHERE session_id = ?',
                        (session_id,),
                    ).fetchone()[0]
                )
                if experiment_count + reservation_count >= int(session_row['max_runs']):
                    raise AgentConfigCollision(
                        f'session {session_id} 已达 max_runs={session_row["max_runs"]} 上限'
                    )

                model_fit_limit = self._model_fit_budget_limit(session_row)
                if model_fit_limit is None:
                    if reserved_model_fits != 0:
                        raise AgentBudgetAccountingError(
                            'disabled budget Session cannot reserve model fits'
                        )
                else:
                    if reserved_model_fits < 1:
                        raise AgentBudgetAccountingError(
                            'budgeted experiment requires a positive model-fit reservation'
                        )
                    actual_fits, existing_reserved_fits = self._model_fit_usage_locked(
                        connection,
                        session_id=session_id,
                    )
                    projected = (
                        actual_fits + existing_reserved_fits + reserved_model_fits
                    )
                    if projected > model_fit_limit:
                        raise AgentBudgetExceeded(
                            'model-fit budget exceeded: '
                            f'charged={actual_fits + existing_reserved_fits}, '
                            f'requested={reserved_model_fits}, limit={model_fit_limit}'
                        )

                attempt = int(
                    connection.execute(
                        '''
                        SELECT MAX(attempt) FROM (
                            SELECT attempt FROM agent_experiments WHERE session_id = ?
                            UNION ALL
                            SELECT attempt FROM agent_experiment_reservations WHERE session_id = ?
                        )
                        ''',
                        (session_id, session_id),
                    ).fetchone()[0]
                    or 0
                ) + 1
                connection.execute(
                    '''
                    INSERT INTO agent_experiment_reservations (
                        reservation_id, session_id, attempt, config_hash,
                        action_json, reserved_model_fits, created_at,
                        owner_id, tenant_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        reservation_id,
                        session_id,
                        attempt,
                        config_hash,
                        json.dumps(action_json, ensure_ascii=False, sort_keys=True),
                        reserved_model_fits,
                        now,
                        owner_id,
                        tenant_id,
                    ),
                )
                row = connection.execute(
                    'SELECT * FROM agent_experiment_reservations WHERE reservation_id = ?',
                    (reservation_id,),
                ).fetchone()
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise AgentConfigCollision(
                    f'config_hash {config_hash} 在该 session 内已存在或提交正在进行'
                ) from exc
            except BaseException:
                connection.rollback()
                raise
        assert row is not None
        return self._reservation(row)

    def bind_reservation(
        self,
        *,
        reservation_id: str,
        run_id: str,
        parent_run_id: str | None,
        rationale: str | None,
        principal: Principal,
    ) -> AgentExperimentRecord:
        """把已创建的 queued Run 与 reservation 绑定成 Experiment。"""
        experiment_id = uuid.uuid4().hex
        now = _timestamp(datetime.now(timezone.utc))
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                reservation_row = connection.execute(
                    '''
                    SELECT * FROM agent_experiment_reservations
                    WHERE reservation_id = ? AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (reservation_id, owner_id, tenant_id),
                ).fetchone()
                if reservation_row is None:
                    raise AgentExperimentNotFound(reservation_id)
                session_row = connection.execute(
                    '''
                    SELECT state FROM agent_sessions
                    WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (reservation_row['session_id'], owner_id, tenant_id),
                ).fetchone()
                if session_row is None:
                    raise AgentSessionNotFound(str(reservation_row['session_id']))
                if session_row['state'] != 'open':
                    raise AgentSessionClosed(
                        f'session {reservation_row["session_id"]} 已经 finalized，不能绑定实验'
                    )
                connection.execute(
                    '''
                    INSERT INTO agent_experiments (
                        experiment_id, session_id, run_id, attempt, parent_run_id,
                        action_json, rationale, config_hash, reserved_model_fits,
                        actual_model_fits, settled, created_at, owner_id, tenant_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?)
                    ''',
                    (
                        experiment_id,
                        reservation_row['session_id'],
                        run_id,
                        int(reservation_row['attempt']),
                        parent_run_id,
                        reservation_row['action_json'],
                        rationale,
                        reservation_row['config_hash'],
                        int(reservation_row['reserved_model_fits']),
                        now,
                        owner_id,
                        tenant_id,
                    ),
                )
                connection.execute(
                    'DELETE FROM agent_experiment_reservations WHERE reservation_id = ?',
                    (reservation_id,),
                )
                row = connection.execute(
                    'SELECT * FROM agent_experiments WHERE experiment_id = ?',
                    (experiment_id,),
                ).fetchone()
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise AgentConfigCollision(
                    f'cannot bind reservation {reservation_id}'
                ) from exc
            except BaseException:
                connection.rollback()
                raise
        assert row is not None
        return self._experiment(row)

    def release_reservation(
        self, *, reservation_id: str, principal: Principal
    ) -> bool:
        """释放 reservation；对已释放或越权 reservation 幂等返回 False。"""
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                changed = connection.execute(
                    '''
                    DELETE FROM agent_experiment_reservations
                    WHERE reservation_id = ? AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (reservation_id, owner_id, tenant_id),
                ).rowcount
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return changed == 1

    def get_budget_usage_scoped(
        self,
        *,
        session_id: str,
        principal: Principal,
    ) -> dict[str, Any] | None:
        """Project safe aggregate accounting; never returns individual Run data."""
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            connection.execute('BEGIN')
            try:
                session_row = connection.execute(
                    '''
                    SELECT * FROM agent_sessions
                    WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (session_id, owner_id, tenant_id),
                ).fetchone()
                if session_row is None:
                    raise AgentSessionNotFound(session_id)
                maximum = self._model_fit_budget_limit(session_row)
                if maximum is None:
                    connection.commit()
                    return None
                actual, reserved = self._model_fit_usage_locked(
                    connection,
                    session_id=session_id,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        charged = actual + reserved
        return {
            'schema_version': 'agent-budget-usage-v1',
            'model_fits': {
                'limit': maximum,
                'actual': actual,
                'reserved': reserved,
                'charged': charged,
                'remaining': max(0, maximum - charged),
            },
        }

    def settle_experiment_model_fits_scoped(
        self,
        *,
        session_id: str,
        run_id: str,
        actual_model_fits: int,
        principal: Principal,
    ) -> AgentExperimentRecord:
        """Idempotently replace a Run's reservation with an observed fit count."""
        if (
            isinstance(actual_model_fits, bool)
            or not isinstance(actual_model_fits, int)
            or actual_model_fits < 0
        ):
            raise AgentBudgetAccountingError(
                'actual_model_fits must be a non-negative integer'
            )
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                session_row = connection.execute(
                    '''
                    SELECT * FROM agent_sessions
                    WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (session_id, owner_id, tenant_id),
                ).fetchone()
                if session_row is None:
                    raise AgentSessionNotFound(session_id)
                if self._model_fit_budget_limit(session_row) is None:
                    raise AgentBudgetAccountingError(
                        'cannot settle model fits for a disabled budget Session'
                    )
                row = connection.execute(
                    '''
                    SELECT * FROM agent_experiments
                    WHERE session_id = ? AND run_id = ?
                      AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (session_id, run_id, owner_id, tenant_id),
                ).fetchone()
                if row is None:
                    raise AgentExperimentNotFound(run_id)
                reserved = int(row['reserved_model_fits'])
                if actual_model_fits > reserved:
                    raise AgentBudgetAccountingError(
                        'actual_model_fits exceeds the server reservation'
                    )
                if bool(row['settled']):
                    if int(row['actual_model_fits']) != actual_model_fits:
                        raise AgentBudgetAccountingError(
                            'model-fit settlement conflicts with the recorded actual'
                        )
                    connection.commit()
                    return self._experiment(row)
                connection.execute(
                    '''
                    UPDATE agent_experiments
                    SET actual_model_fits = ?, settled = 1
                    WHERE experiment_id = ? AND settled = 0
                    ''',
                    (actual_model_fits, row['experiment_id']),
                )
                updated = connection.execute(
                    'SELECT * FROM agent_experiments WHERE experiment_id = ?',
                    (row['experiment_id'],),
                ).fetchone()
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        assert updated is not None
        return self._experiment(updated)

    def has_non_terminal_experiment_scoped(
        self, *, session_id: str, run_states: dict[str, str], principal: Principal
    ) -> bool:
        """判断 Session 内是否存在非终态实验（queued/running）。

        ``run_states`` 是 {run_id: state} 的映射，由调用方先从 RunRepository
        按 session_id 的实验列表查一次得到，避免在 Repository 里再嵌一
        次 Run 查询造成循环引用。
        """
        for state in run_states.values():
            if state in {'queued', 'running'}:
                return True
        return False

    def next_attempt_scoped(
        self, *, session_id: str, principal: Principal
    ) -> int:
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            row = connection.execute(
                '''
                SELECT COALESCE(MAX(attempt), 0) AS last_attempt FROM agent_experiments
                WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (session_id, owner_id, tenant_id),
            ).fetchone()
        return int(row['last_attempt']) + 1 if row else 1

    def create_experiment(
        self,
        *,
        session_id: str,
        run_id: str,
        attempt: int,
        parent_run_id: str | None,
        action_json: dict[str, Any],
        rationale: str | None,
        config_hash: str,
        principal: Principal,
    ) -> AgentExperimentRecord:
        experiment_id = uuid.uuid4().hex
        now = _timestamp(datetime.now(timezone.utc))
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            # 二次防御：UNIQUE (session_id, config_hash) 已能兜底，但主动抛错能让上层拿到语义
            collision = connection.execute(
                '''
                SELECT 1 FROM agent_experiments
                WHERE session_id = ? AND config_hash = ?
                ''',
                (session_id, config_hash),
            ).fetchone()
            if collision is not None:
                connection.rollback()
                raise AgentConfigCollision(
                    f'config_hash {config_hash} already exists in session {session_id}'
                )
            connection.execute(
                '''
                INSERT INTO agent_experiments (
                    experiment_id, session_id, run_id, attempt, parent_run_id,
                    action_json, rationale, config_hash, created_at,
                    owner_id, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    experiment_id,
                    session_id,
                    run_id,
                    int(attempt),
                    parent_run_id,
                    json.dumps(action_json, ensure_ascii=False, sort_keys=True),
                    rationale,
                    config_hash,
                    now,
                    owner_id,
                    tenant_id,
                ),
            )
            row = connection.execute(
                'SELECT * FROM agent_experiments WHERE experiment_id = ?',
                (experiment_id,),
            ).fetchone()
            connection.commit()
        assert row is not None
        return self._experiment(row)

    def _memory_write_session_row(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        principal: Principal,
    ) -> sqlite3.Row:
        """Load and re-check memory write policy inside the write transaction."""
        owner_id, tenant_id = self._scope_values(principal)
        row = connection.execute(
            '''
            SELECT * FROM agent_sessions
            WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
            ''',
            (session_id, owner_id, tenant_id),
        ).fetchone()
        if row is None:
            raise AgentSessionNotFound(session_id)
        self._assert_memory_write_policy(row)
        return row

    @staticmethod
    def _memory_policy(row: sqlite3.Row) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_flags = _decode_json(
            row['module_flags_json'] if 'module_flags_json' in row.keys() else None,
            {},
        )
        raw_policy = _decode_json(
            row['context_policy_json']
            if 'context_policy_json' in row.keys()
            else None,
            {},
        )
        flags = raw_flags if isinstance(raw_flags, dict) else {}
        policy = raw_policy if isinstance(raw_policy, dict) else {}
        return flags, policy

    @classmethod
    def _assert_memory_write_policy(cls, row: sqlite3.Row) -> None:
        flags, policy = cls._memory_policy(row)
        required_modules = (
            'case_memory',
            'evidence_card',
            'fail_fast_guard',
            'feedback_diagnosis',
        )
        if not all(flags.get(name) is True for name in required_modules):
            raise AgentMemoryWriteDenied('case memory modules are not enabled')
        if policy.get('case_write') is not True:
            raise AgentMemoryWriteDenied('case_write policy is disabled')
        if policy.get('source_role', 'development') == 'benchmark':
            raise AgentMemoryWriteDenied('benchmark sessions are read-only')

    @classmethod
    def _session_requests_memory_write(cls, row: sqlite3.Row) -> bool:
        _flags, policy = cls._memory_policy(row)
        return policy.get('case_write') is True

    @staticmethod
    def _canonical_outcome_from_row(row: sqlite3.Row) -> dict[str, Any]:
        try:
            raw = json.loads(str(row['payload_json']))
            canonical = normalize_terminal_outcome(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise AgentMemoryCollision('stored terminal outcome is invalid') from exc
        if _canonical_json(raw) != _canonical_json(canonical):
            raise AgentMemoryCollision('stored terminal outcome is not canonical')
        if canonical['state'] != row['state']:
            raise AgentMemoryCollision('stored terminal outcome state mismatch')
        return canonical

    @staticmethod
    def _canonical_memory_from_row(row: sqlite3.Row) -> dict[str, Any]:
        try:
            raw = json.loads(str(row['payload_json']))
            canonical = normalize_memory_record(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise AgentMemoryCollision('stored memory record is invalid') from exc
        if _canonical_json(raw) != _canonical_json(canonical):
            raise AgentMemoryCollision('stored memory record is not canonical')
        if canonical['record_type'] != row['record_type']:
            raise AgentMemoryCollision('stored memory record type mismatch')
        return canonical

    def record_terminal_outcome_scoped(
        self,
        *,
        session_id: str,
        run_id: str,
        state: str,
        task_signature: str,
        evidence_signature: str,
        payload: dict[str, Any],
        principal: Principal,
        failure_memory_payload: dict[str, Any] | None = None,
    ) -> AgentTerminalOutcomeRecord:
        """Idempotently store one terminal outcome and optional failure lesson.

        ``BEGIN IMMEDIATE`` plus the two UNIQUE constraints make concurrent
        feedback requests converge on one outcome and one failure record.
        The Session policy and Experiment ownership are checked again in this
        repository even when the service already performed the same checks.
        """
        task_key = _validate_memory_signature(task_signature)
        evidence_key = _validate_memory_signature(evidence_signature)
        normalized_outcome = normalize_terminal_outcome(payload)
        normalized_state = str(state).strip().lower()
        if normalized_outcome['state'] != normalized_state:
            raise ValueError('terminal outcome state mismatch')
        if (
            normalized_state == 'succeeded'
            and 'validation_score' not in normalized_outcome['result']
        ):
            raise AgentMemoryWriteDenied(
                'successful memory outcome requires validation_score'
            )
        canonical_outcome_json = _canonical_json(normalized_outcome)
        normalized_failure: dict[str, Any] | None = None
        canonical_failure_json: str | None = None
        if failure_memory_payload is not None:
            normalized_failure = normalize_memory_record(failure_memory_payload)
            if normalized_state != 'failed' or normalized_failure['record_type'] != 'failure':
                raise ValueError('failure memory requires a failed outcome')
            canonical_failure_json = _canonical_json(normalized_failure)

        now = _timestamp(datetime.now(timezone.utc))
        owner_id, tenant_id = self._scope_values(principal)
        candidate_outcome_id = uuid.uuid4().hex
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                self._memory_write_session_row(
                    connection, session_id=session_id, principal=principal
                )
                experiment = connection.execute(
                    '''
                    SELECT 1 FROM agent_experiments
                    WHERE session_id = ? AND run_id = ?
                      AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (session_id, run_id, owner_id, tenant_id),
                ).fetchone()
                if experiment is None:
                    raise AgentExperimentNotFound(run_id)
                connection.execute(
                    '''
                    INSERT OR IGNORE INTO agent_terminal_outcomes (
                        outcome_id, session_id, run_id, state,
                        task_signature, evidence_signature, payload_json,
                        selected, recorded_at, owner_id, tenant_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    ''',
                    (
                        candidate_outcome_id,
                        session_id,
                        run_id,
                        normalized_state,
                        task_key,
                        evidence_key,
                        canonical_outcome_json,
                        now,
                        owner_id,
                        tenant_id,
                    ),
                )
                row = connection.execute(
                    '''
                    SELECT * FROM agent_terminal_outcomes
                    WHERE session_id = ? AND run_id = ?
                      AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (session_id, run_id, owner_id, tenant_id),
                ).fetchone()
                if row is None:
                    raise AgentMemoryCollision('terminal outcome could not be persisted')
                if (
                    row['state'] != normalized_state
                    or row['task_signature'] != task_key
                    or row['evidence_signature'] != evidence_key
                ):
                    raise AgentMemoryCollision(
                        'terminal outcome already exists with incompatible state'
                    )
                stored_outcome = self._canonical_outcome_from_row(row)
                if _canonical_json(stored_outcome) != canonical_outcome_json:
                    raise AgentMemoryCollision(
                        'terminal outcome already exists with incompatible payload'
                    )
                if normalized_failure is not None:
                    connection.execute(
                        '''
                        INSERT OR IGNORE INTO agent_memory_records (
                            memory_id, source_outcome_id, record_type,
                            task_signature, evidence_signature, payload_json,
                            created_at, owner_id, tenant_id
                        ) VALUES (?, ?, 'failure', ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            uuid.uuid4().hex,
                            row['outcome_id'],
                            task_key,
                            evidence_key,
                            canonical_failure_json,
                            now,
                            owner_id,
                            tenant_id,
                        ),
                    )
                    memory_row = connection.execute(
                        '''
                        SELECT * FROM agent_memory_records
                        WHERE source_outcome_id = ? AND record_type = 'failure'
                        ''',
                        (row['outcome_id'],),
                    ).fetchone()
                    if memory_row is None:
                        raise AgentMemoryCollision(
                            'failure memory could not be persisted'
                        )
                    stored_failure = self._canonical_memory_from_row(memory_row)
                    if (
                        _canonical_json(stored_failure) != canonical_failure_json
                        or memory_row['task_signature'] != task_key
                        or memory_row['evidence_signature'] != evidence_key
                    ):
                        raise AgentMemoryCollision(
                            'failure memory already exists with incompatible payload'
                        )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        assert row is not None
        return self._terminal_outcome(row)

    def _ensure_selected_case_locked(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        run_id: str,
        principal: Principal,
        now: str,
    ) -> AgentMemoryRecord:
        """Build/verify the selected case from canonical stored outcome only."""
        owner_id, tenant_id = self._scope_values(principal)
        experiment = connection.execute(
            '''
            SELECT 1 FROM agent_experiments
            WHERE session_id = ? AND run_id = ?
              AND owner_id IS ? AND tenant_id IS ?
            ''',
            (session_id, run_id, owner_id, tenant_id),
        ).fetchone()
        if experiment is None:
            raise AgentExperimentNotFound(run_id)
        outcome = connection.execute(
            '''
            SELECT * FROM agent_terminal_outcomes
            WHERE session_id = ? AND run_id = ?
              AND owner_id IS ? AND tenant_id IS ?
            ''',
            (session_id, run_id, owner_id, tenant_id),
        ).fetchone()
        if outcome is None or outcome['state'] != 'succeeded':
            raise AgentMemoryWriteDenied(
                'selected case requires a persisted successful outcome'
            )
        canonical_outcome = self._canonical_outcome_from_row(outcome)
        if 'validation_score' not in canonical_outcome['result']:
            raise AgentMemoryWriteDenied(
                'selected case requires a validation score'
            )
        canonical_case = build_memory_record('case', canonical_outcome)
        canonical_case_json = _canonical_json(canonical_case)
        connection.execute(
            '''
            INSERT OR IGNORE INTO agent_memory_records (
                memory_id, source_outcome_id, record_type,
                task_signature, evidence_signature, payload_json,
                created_at, owner_id, tenant_id
            ) VALUES (?, ?, 'case', ?, ?, ?, ?, ?, ?)
            ''',
            (
                uuid.uuid4().hex,
                outcome['outcome_id'],
                outcome['task_signature'],
                outcome['evidence_signature'],
                canonical_case_json,
                now,
                owner_id,
                tenant_id,
            ),
        )
        memory_row = connection.execute(
            '''
            SELECT * FROM agent_memory_records
            WHERE source_outcome_id = ? AND record_type = 'case'
            ''',
            (outcome['outcome_id'],),
        ).fetchone()
        if memory_row is None:
            raise AgentMemoryCollision('selected case could not be persisted')
        stored_case = self._canonical_memory_from_row(memory_row)
        if (
            _canonical_json(stored_case) != canonical_case_json
            or memory_row['task_signature'] != outcome['task_signature']
            or memory_row['evidence_signature'] != outcome['evidence_signature']
        ):
            raise AgentMemoryCollision(
                'selected case already exists with incompatible payload'
            )
        connection.execute(
            'UPDATE agent_terminal_outcomes SET selected = 1 WHERE outcome_id = ?',
            (outcome['outcome_id'],),
        )
        return self._memory_record(memory_row)

    def record_selected_case_scoped(
        self,
        *,
        session_id: str,
        run_id: str,
        principal: Principal,
    ) -> AgentMemoryRecord:
        """Repair/ensure a finalized selected case without caller payload."""
        owner_id, tenant_id = self._scope_values(principal)
        now = _timestamp(datetime.now(timezone.utc))
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                session_row = self._memory_write_session_row(
                    connection, session_id=session_id, principal=principal
                )
                if (
                    session_row['state'] != 'finalized'
                    or session_row['selected_run_id'] != run_id
                ):
                    raise AgentMemoryWriteDenied(
                        'only the finalized selected run may become a case'
                    )
                memory = self._ensure_selected_case_locked(
                    connection,
                    session_id=session_id,
                    run_id=run_id,
                    principal=principal,
                    now=now,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return memory

    def list_memory_payloads_scoped(
        self,
        *,
        task_signature: str,
        evidence_signature: str,
        principal: Principal,
        limit: int = MAX_MEMORY_CONTEXT_ITEMS,
    ) -> list[dict[str, Any]]:
        """Read deterministic Principal-scoped top-k payloads for a new Session."""
        task_key = _validate_memory_signature(task_signature)
        evidence_key = _validate_memory_signature(evidence_signature)
        bounded_limit = max(0, min(int(limit), MAX_MEMORY_CONTEXT_ITEMS))
        if bounded_limit == 0:
            return []
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            rows = connection.execute(
                '''
                SELECT * FROM agent_memory_records
                WHERE owner_id IS ? AND tenant_id IS ? AND task_signature = ?
                ORDER BY
                    CASE WHEN evidence_signature = ? THEN 0 ELSE 1 END ASC,
                    created_at DESC,
                    memory_id ASC
                LIMIT ?
                ''',
                (
                    owner_id,
                    tenant_id,
                    task_key,
                    evidence_key,
                    bounded_limit * 4,
                ),
            ).fetchall()
        payloads: list[dict[str, Any]] = []
        for row in rows:
            try:
                payloads.append(
                    normalize_memory_record(_decode_json(row['payload_json'], {}))
                )
            except (AgentMemoryPayloadError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if len(payloads) >= bounded_limit:
                break
        return payloads

    def finalize_session(
        self,
        *,
        session_id: str,
        selected_run_id: str,
        principal: Principal,
    ) -> AgentSessionRecord:
        """Atomically finalize and, when enabled, publish the selected case.

        Memory-enabled writes validate a canonical persisted successful outcome,
        build the case from that outcome, insert/verify it, mark it selected, and
        change Session state under one ``BEGIN IMMEDIATE`` transaction.
        """
        owner_id, tenant_id = self._scope_values(principal)
        now = _timestamp(datetime.now(timezone.utc))
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                row = connection.execute(
                    '''
                    SELECT * FROM agent_sessions
                    WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (session_id, owner_id, tenant_id),
                ).fetchone()
                if row is None:
                    raise AgentSessionNotFound(session_id)
                if (
                    row['state'] == 'finalized'
                    and row['selected_run_id'] != selected_run_id
                ):
                    raise AgentSessionClosed(
                        f'session {session_id} already finalized with different run'
                    )
                if self._session_requests_memory_write(row):
                    self._assert_memory_write_policy(row)
                    self._ensure_selected_case_locked(
                        connection,
                        session_id=session_id,
                        run_id=selected_run_id,
                        principal=principal,
                        now=now,
                    )
                if row['state'] == 'open':
                    changed = connection.execute(
                        '''
                        UPDATE agent_sessions
                        SET state = 'finalized', selected_run_id = ?, finalized_at = ?
                        WHERE session_id = ? AND state = 'open'
                          AND owner_id IS ? AND tenant_id IS ?
                        ''',
                        (
                            selected_run_id,
                            now,
                            session_id,
                            owner_id,
                            tenant_id,
                        ),
                    ).rowcount
                    if changed != 1:
                        raise AgentSessionClosed(
                            f'session {session_id} could not be finalized'
                        )
                updated = connection.execute(
                    '''
                    SELECT * FROM agent_sessions
                    WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                    ''',
                    (session_id, owner_id, tenant_id),
                ).fetchone()
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        assert updated is not None
        return self._session(updated)


# 联合查询辅助：让 Service 层在不互相 import 的前提下把 Experiment ↔ RunRecord
# 关系收齐。
def collect_run_states(
    repository: RunRepository, run_ids: list[str]
) -> dict[str, str]:
    """批量按 run_id 取 state；不存在的 run 视为已结束（status=missing）。"""
    states: dict[str, str] = {}
    for run_id in run_ids:
        try:
            states[run_id] = repository.get(run_id).state
        except RunNotFound:
            states[run_id] = 'missing'
    return states
