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


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _decode_json(value: str | None, default: object) -> object:
    if not value:
        return default
    return json.loads(value)


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
                    created_at TEXT NOT NULL,
                    owner_id TEXT,
                    tenant_id TEXT,
                    UNIQUE(session_id, config_hash),
                    UNIQUE(session_id, attempt)
                )
                '''
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS idx_agent_reservations_session '
                'ON agent_experiment_reservations(session_id, attempt)'
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
            created_at=row['created_at'],
            owner_id=row['owner_id'] if 'owner_id' in row.keys() else None,
            tenant_id=row['tenant_id'] if 'tenant_id' in row.keys() else None,
        )

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
    ) -> AgentExperimentReservation:
        """原子预留一次 Experiment，防止并发请求越过 max_runs。

        Session、历史 Experiment 和未绑定 Reservation 的读取均在同一个
        ``BEGIN IMMEDIATE`` 内完成；Run 尚未创建时，其他请求已经能观察到
        这次 Reservation，从而不会重复入队。
        """
        reservation_id = uuid.uuid4().hex
        now = _timestamp(datetime.now(timezone.utc))
        owner_id, tenant_id = self._scope_values(principal)
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
                        action_json, created_at, owner_id, tenant_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        reservation_id,
                        session_id,
                        attempt,
                        config_hash,
                        json.dumps(action_json, ensure_ascii=False, sort_keys=True),
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
                        action_json, rationale, config_hash, created_at,
                        owner_id, tenant_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def finalize_session(
        self,
        *,
        session_id: str,
        selected_run_id: str,
        principal: Principal,
    ) -> AgentSessionRecord:
        """原子地把 Session 切到 finalized。幂等：已 finalized 且 selected_run_id
        一致时返回旧记录；已 finalized 但 selected_run_id 不一致则报错（不允许
        "换锁"）。"""
        owner_id, tenant_id = self._scope_values(principal)
        now = _timestamp(datetime.now(timezone.utc))
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                '''
                SELECT * FROM agent_sessions
                WHERE session_id = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (session_id, owner_id, tenant_id),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise AgentSessionNotFound(session_id)
            if row['state'] == 'finalized':
                connection.commit()
                existing = self._session(row)
                if existing.selected_run_id != selected_run_id:
                    raise AgentSessionClosed(
                        f'session {session_id} already finalized with different run'
                    )
                return existing
            connection.execute(
                '''
                UPDATE agent_sessions
                SET state = 'finalized', selected_run_id = ?, finalized_at = ?
                WHERE session_id = ? AND state = 'open'
                ''',
                (selected_run_id, now, session_id),
            )
            updated = connection.execute(
                'SELECT * FROM agent_sessions WHERE session_id = ?', (session_id,)
            ).fetchone()
            connection.commit()
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
