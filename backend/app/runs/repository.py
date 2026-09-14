"""SQLite Run 状态机与 worker claim/lease 仓库。

所有关键迁移在 ``BEGIN IMMEDIATE`` 事务内校验当前 state、claim token 和 version。
queued Run 只能被一个 worker claim；运行 lease 过期后可回队列；终态不可逆。
这些数据库约束是并发正确性的来源，status.json 不参与状态决策。

教学级模块说明：
- 系统位置：这是 Run（训练任务）生命周期的持久化层，位于 backend/app/runs/ 下。
  上层（HTTP 训练接口、训练 worker、结果接口）不直接碰 SQLite，全部通过
  RunRepository 提供的方法创建队列任务、claim、续租、上报进度、终态落库、
  按 Principal 范围查询/取消/删除。存储文件默认为 storage/runs.sqlite3。
- 协作模块：数据类（Principal、RunRecord、RunState）来自 .contracts；worker
  契约版本来自 ..version（WORKER_CONTRACT_VERSION）；Run 的结果文件与
  manifest 由 artifacts.py 负责，本模块只在记录里保存 manifest_name 引用。
- 关键设计约束：
  1) 并发正确性来自数据库：每个关键写操作都在 BEGIN IMMEDIATE 事务里先校验
     当前 state、claim_token 和乐观锁 version，再用带条件的 UPDATE 影响行数
     判断是否成功；两个 worker 同时抢同一个 queued Run 时只有一个能成功。
  2) 状态机不可逆：queued → running → succeeded/failed/cancelled，终态无出边，
     由 ALLOWED_TRANSITIONS 与 SQL 中的 state 条件双重保证。
  3) lease 租约：running 的 Run 携带 lease_expires_at；worker 崩溃导致租约过期后，
     requeue_expired 可把它安全放回队列，由其他 worker 重新 claim。
  4) 权限隔离：server 模式下读取、取消、删除必须走 *_scoped 方法，按
     owner_id/tenant_id（Principal）过滤；身份只由服务端注入，不信任请求体。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from ..version import WORKER_CONTRACT_VERSION
from .contracts import Principal, RunRecord, RunState


# 状态机迁移表：键是当前状态，值是允许到达的下一状态集合。
# 终态（succeeded/failed/cancelled）出边为空集，保证终态不可逆；
# queued 可直接 failed（例如入队后校验失败），running 可被 cancel。
ALLOWED_TRANSITIONS: dict[RunState, set[RunState]] = {
    'queued': {'running', 'cancelled', 'failed'},
    'running': {'succeeded', 'failed', 'cancelled'},
    'succeeded': set(),
    'failed': set(),
    'cancelled': set(),
}
# 终态集合：只有终态 Run 才允许删除（delete_terminal），防止误删仍在排队的或执行中的任务。
TERMINAL_STATES: set[RunState] = {'succeeded', 'failed', 'cancelled'}


class InvalidRunTransition(RuntimeError):
    """请求的状态迁移违反当前状态、claim 或版本约束。"""
    pass


class RunNotFound(KeyError):
    """请求的 run_id 在当前仓库和 Principal 范围内不存在。"""
    pass


class RunSubmissionKeyConflict(RuntimeError):
    """submission key 已被不同 payload 或 Principal 使用。"""


class RunSubmissionMappingCorrupt(RuntimeError):
    """submission mapping 存在，但目标 Run 已不可验证。"""


@dataclass(frozen=True)
class RunSubmissionMappingLookup:
    """内部 submission mapping 查询结果；scope mismatch 不携带 Run 标识。"""

    status: str
    run_id: str | None = None
    payload_hash: str | None = None
    submission_source: str | None = None


# 统一把时间戳规范化为带 UTC 时区的 ISO 字符串：naive 时间一律按 UTC 解释。
# 时间以文本存进 SQLite、比较靠字典序，混入不同时区表示会得出错误的先后关系。
def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


# 把库里的 JSON 文本列解码回 Python 对象；NULL/空串一律返回 default，
# 兼容早期尚未写入该列的历史记录。
def _decode_json(value: str | None, default: object) -> object:
    if not value:
        return default
    return json.loads(value)


def _stopped_progress(
    raw_progress: str | None,
    *,
    stopped_at: str,
    reason: str,
    message: str,
) -> str:
    """把停止原因合并进现有进度，供 API/前端稳定展示 STOP。"""
    progress = dict(_decode_json(raw_progress, {}))
    progress.update(
        {
            'stop_status': 'stopped',
            'stop_reason': reason,
            'stop_message': message,
            'stopped_at': stopped_at,
        }
    )
    return json.dumps(progress, ensure_ascii=False)


# Run 持久化仓库。内部方法按职责分四组：
# - 生命周期写操作：create_queued / claim_next / renew_lease / update_progress /
#   update_dataset_snapshot / cancel / finish_success / finish_failure / requeue_expired。
# - 读取操作：get / list / exists，以及按 Principal 过滤的 *_scoped 变体。
# - worker 健康：record_worker_heartbeat / worker_health / assert_active。
# - 维护操作：assign_unowned（历史无主 Run 重新绑定）、delete_terminal(*_scoped)、
#   import_legacy（旧数据迁移导入）。
# __init__ 只记录数据库路径并确保父目录存在；真正的建表/迁移在 initialize()。
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

# 把一行 sqlite3.Row 映射为 RunRecord 数据类。
# 要点：config/progress/error/dataset_snapshot 在库里是 JSON 文本，这里统一解码回 dict；
# 对 error_json 等“后加列”用列存在性判断做兼容——旧数据库可能还没执行对应迁移，
# 缺列时给空默认值，保证读旧库不炸。
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

    # 建表 + 轻量迁移 + 索引，幂等可重复执行（启动时调用）。
    # 迁移策略是“用 PRAGMA table_info 探测，缺列就 ALTER TABLE 补上”，不维护版本号；
    # worker_heartbeats.contract_version 保持可空，使旧 worker 的心跳仍可保留并明确
    # 判为不兼容。idx_runs_queue 支撑 claim_next 的 queued 扫描，idx_runs_scope 支撑
    # 按 Principal 的范围列表查询。
    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS evaluation_plans_v1 (
                scope_digest TEXT NOT NULL, plan_digest TEXT NOT NULL, plan_json TEXT NOT NULL,
                PRIMARY KEY(scope_digest, plan_digest))""")
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
                    active_run_id TEXT,
                    contract_version TEXT
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS run_submission_keys_v1 (
                    submission_key TEXT PRIMARY KEY,
                    submission_source TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    run_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT,
                    tenant_id TEXT,
                    created_at TEXT NOT NULL
                )
                '''
            )
            heartbeat_columns = {
                str(row['name'])
                for row in connection.execute('PRAGMA table_info(worker_heartbeats)').fetchall()
            }
            if 'contract_version' not in heartbeat_columns:
                # 保持可空，使旧 Worker 写入的心跳仍可被保留并明确判为不兼容。
                connection.execute(
                    'ALTER TABLE worker_heartbeats ADD COLUMN contract_version TEXT'
                )

    # 创建 queued Run。HTTP 训练请求只做到这一步（入队），不直接启动训练——
    # 真正的执行由 worker 通过 claim_next 领取。owner/tenant 只来自服务端注入的
    # Principal，请求体里的同名字段一律不被信任。version 从 1 开始计数。
    def create_queued(
        self,
        *,
        dataset_id: str | None,
        legacy_data_path: str | None = None,
        config: dict[str, Any],
        dataset_snapshot: dict[str, Any] | None = None,
        principal: Principal = Principal(),
        submission_key: str | None = None,
        submission_source: str | None = None,
        submission_payload_hash: str | None = None,
    ) -> RunRecord:
        mapping_values = (submission_key, submission_source, submission_payload_hash)
        if any(value is not None for value in mapping_values):
            if not all(isinstance(value, str) and value for value in mapping_values):
                raise ValueError('submission mapping parameters must be provided together')
            if submission_source != 'agent':
                raise ValueError('submission mapping is only available to agent submissions')
            if len(submission_payload_hash or '') != 64 or any(
                character not in '0123456789abcdef'
                for character in (submission_payload_hash or '').lower()
            ):
                raise ValueError('submission_payload_hash must be a SHA-256 hex digest')
        run_id = uuid.uuid4().hex
        now = _timestamp(datetime.now(timezone.utc))
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            if submission_key is not None:
                existing_mapping = connection.execute(
                    'SELECT * FROM run_submission_keys_v1 WHERE submission_key = ?',
                    (submission_key,),
                ).fetchone()
                if existing_mapping is not None:
                    same_scope = (
                        existing_mapping['owner_id'] == principal.owner_id
                        and existing_mapping['tenant_id'] == principal.tenant_id
                    )
                    same_payload = (
                        existing_mapping['submission_source'] == submission_source
                        and existing_mapping['payload_hash'] == submission_payload_hash
                    )
                    if not same_scope or not same_payload:
                        connection.rollback()
                        raise RunSubmissionKeyConflict('submission key conflict')
                    row = connection.execute(
                        'SELECT * FROM runs WHERE run_id = ?',
                        (existing_mapping['run_id'],),
                    ).fetchone()
                    if row is None:
                        connection.rollback()
                        raise RunSubmissionMappingCorrupt('submission mapping target is missing')
                    connection.commit()
                    return self._record(row)
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
            if submission_key is not None:
                connection.execute(
                    '''
                    INSERT INTO run_submission_keys_v1(
                        submission_key, submission_source, payload_hash, run_id,
                        owner_id, tenant_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        submission_key,
                        submission_source,
                        submission_payload_hash,
                        run_id,
                        principal.owner_id,
                        principal.tenant_id,
                        now,
                    ),
                )
            row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert row is not None
        return self._record(row)

    def lookup_submission_mapping(
        self,
        submission_key: str,
        *,
        principal: Principal,
    ) -> RunSubmissionMappingLookup:
        """按内部 key 查询映射；跨 Principal 只返回不可枚举的 scope_mismatch。"""
        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM run_submission_keys_v1 WHERE submission_key = ?',
                (submission_key,),
            ).fetchone()
        if row is None:
            return RunSubmissionMappingLookup(status='missing')
        if row['owner_id'] != principal.owner_id or row['tenant_id'] != principal.tenant_id:
            return RunSubmissionMappingLookup(status='scope_mismatch')
        return RunSubmissionMappingLookup(
            status='found',
            run_id=str(row['run_id']),
            payload_hash=str(row['payload_hash']),
            submission_source=str(row['submission_source']),
        )

    # worker 领取队首任务（FIFO：created_at 最早优先，run_id 打破并列）。
    # 并发安全靠两点：BEGIN IMMEDIATE 串行化所有 claim；UPDATE 的 WHERE 带上
    # state='queued'，两个 worker 选中同一行时只有一个能命中（rowcount=1），另一个
    # 回滚抛 InvalidRunTransition。claim_token 是本次执行的随机凭证，后续所有写操作
    # 都要出示；lease_expires_at 是租约死线，worker 须周期续租，过期可被回收重排。
    # started_at 用 COALESCE 保留首次启动时间（重排后再次 claim 不刷新）。
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

    # 续租：只有仍持有 claim 且租约未过期（lease_expires_at > now）才能延长。
    # “未过期才能续”这个条件让过期任务立刻进入可回收状态，避免已死 worker 的任务
    # 被它自己迟到的续租请求救活。
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

    # worker 上报训练进度。WHERE 同时限定 state='running' 与 claim_token：
    # 只有“当前仍持有有效 claim 的 worker”才能写进度；Run 已被取消、被回收或
    # 被别的 worker 抢走时 rowcount=0，抛 InvalidRunTransition。
    # version 每次 +1 是乐观锁，调用方可据此检测并发写。
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

    # 在有效 claim 内回写数据集快照（各划分 split 的样本计数等可追溯信息）。
    # 与 update_progress 相同的并发约束：必须 running + 持有 claim_token 才允许写，
    # 保证快照一定属于本次执行解析出的数据，而不是过期 claim 的迟到写入。
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

    # 取消 Run（无权限过滤的内部版本；HTTP 层应优先用 cancel_scoped）。
    # 两道防线：先在 Python 侧查 ALLOWED_TRANSITIONS，再在 UPDATE 的 WHERE 里
    # 带上读到的旧 state——若并发下状态已被别人改掉，rowcount=0 抛错并回滚，
    # 避免“读到 queued、写入时已 running”造成的误取消。取消时清空租约并写 finished_at。
    def cancel(
        self,
        run_id: str,
        *,
        now: datetime,
        reason: str = 'user_requested',
        message: str = '用户已停止训练',
    ) -> RunRecord:
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
                SET state = 'cancelled', version = version + 1, claim_token = NULL,
                    worker_id = NULL, lease_expires_at = NULL, manifest_name = NULL,
                    progress_json = ?, updated_at = ?, finished_at = ?
                WHERE run_id = ? AND state = ?
                ''',
                (
                    _stopped_progress(
                        row['progress_json'],
                        stopped_at=now_text,
                        reason=reason,
                        message=message,
                    ),
                    now_text,
                    now_text,
                    run_id,
                    state,
                ),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot cancel run {run_id}')
            updated = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert updated is not None
        return self._record(updated)

    # 成功终态的薄封装：登记 manifest_name（指向 artifacts.py 发布的 manifest.json），
    # 其余委托 _finish。按契约只有 Manifest 完成后才应走到这里，Run 才能算成功。
    def finish_success(
        self,
        run_id: str,
        *,
        claim_token: str,
        now: datetime,
        manifest_name: str = 'manifest.json',
    ) -> RunRecord:
        return self._finish(run_id, claim_token=claim_token, now=now, state='succeeded', manifest_name=manifest_name)

    # 失败终态的薄封装：错误摘要进 error 列，结构化详情（异常类型、堆栈等）进
    # error_json，最终都委托给 _finish；finish_success 则额外登记 manifest_name，
    # 把 DB 记录与 artifacts.py 生成的 manifest.json 关联起来。
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

    # 终态落库的唯一出口。WHERE 强制 state='running' + claim_token 匹配：
    # 已终态的 Run 不可再次 finish（终态不可逆）；Run 被回收或取消后，旧 worker
    # 迟到的提交也会因 token/state 不匹配而失败。落库时清空租约、写 finished_at，
    # version +1 供乐观锁观察者感知。
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

    # 回收租约过期或缺失的 running Run：这说明原训练执行者已经失联，不能再把
    # 同一次训练静默重跑。统一落为 cancelled（对用户展示 STOP），保留历史记录但
    # 清空 claim/worker/manifest；返回被停止的 run_id，供上层删除半成品目录。
    def stop_expired(self, *, now: datetime) -> list[str]:
        now_text = _timestamp(now)
        stopped_run_ids: list[str] = []
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            rows = connection.execute(
                '''
                SELECT run_id, progress_json FROM runs
                WHERE state = 'running'
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                ''',
                (now_text,),
            ).fetchall()
            for row in rows:
                changed = connection.execute(
                    '''
                    UPDATE runs
                    SET state = 'cancelled', version = version + 1,
                        claim_token = NULL, worker_id = NULL, lease_expires_at = NULL,
                        manifest_name = NULL, progress_json = ?, error = NULL,
                        error_json = '{}', updated_at = ?, finished_at = ?
                    WHERE run_id = ? AND state = 'running'
                      AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                    ''',
                    (
                        _stopped_progress(
                            row['progress_json'],
                            stopped_at=now_text,
                            reason='worker_interrupted',
                            message='训练进程意外中断，任务已停止',
                        ),
                        now_text,
                        now_text,
                        row['run_id'],
                        now_text,
                    ),
                ).rowcount
                if changed == 1:
                    stopped_run_ids.append(str(row['run_id']))
            connection.commit()
        return stopped_run_ids

    def requeue_expired(self, *, now: datetime) -> int:
        """兼容旧调用名；过期任务现在停止而不再静默重跑。"""
        return len(self.stop_expired(now=now))

    # worker 心跳上报：UPSERT（存在则更新）每个 worker 一行，并顺带清理 7 天前的
    # 陈旧心跳，防止表无限膨胀。contract_version 用于识别新旧 worker 协议是否兼容。
    def record_worker_heartbeat(
        self,
        *,
        worker_id: str,
        now: datetime,
        active_run_id: str | None = None,
        contract_version: str | None = None,
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
                INSERT INTO worker_heartbeats (
                    worker_id, last_seen_at, active_run_id, contract_version
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    last_seen_at = excluded.last_seen_at,
                    active_run_id = excluded.active_run_id,
                    contract_version = excluded.contract_version
                ''',
                (worker_id, now_text, active_run_id, contract_version),
            )

    # 汇总 worker 健康状态供训练接口/前端判断“现在能不能跑训练”。
    # 逻辑：取最近 20 条心跳，计算每条距今秒数；age <= stale_seconds（默认 15s）
    # 视为 live；所有 live worker 的 contract_version 都等于期望值才算 compatible。
    # 多个 live worker 且版本不一致时 contract_version 返回 None（无法确定统一版本），
    # 上层据此提示“有 worker 但协议不兼容”，避免把任务派给旧协议 worker。
    def worker_health(
        self,
        *,
        now: datetime,
        stale_seconds: float = 15.0,
        expected_contract_version: str | None = WORKER_CONTRACT_VERSION,
    ) -> dict[str, Any]:
        now_utc = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        with self._connection() as connection:
            rows = connection.execute(
                '''
                SELECT worker_id, last_seen_at, active_run_id, contract_version
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
                    'contract_version': row['contract_version'],
                    'age_seconds': age,
                    'live': age is not None and age <= stale_seconds,
                    'compatible': (
                        expected_contract_version is not None
                        and row['contract_version'] == expected_contract_version
                    ),
                }
            )
        live_workers = [item for item in workers if item['live']]
        live_contract_versions = {
            item['contract_version']
            for item in live_workers
            if item['contract_version'] is not None
        }
        contract_version = (
            next(iter(live_contract_versions))
            if len(live_contract_versions) == 1
            and all(item['contract_version'] is not None for item in live_workers)
            else None
        )
        compatible = bool(live_workers) and all(
            item['compatible'] for item in live_workers
        )
        return {
            'available': bool(live_workers),
            'compatible': compatible,
            'contract_version': contract_version,
            'live_count': len(live_workers),
            'last_seen_at': workers[0]['last_seen_at'] if workers else None,
            'active_run_ids': [
                item['active_run_id']
                for item in live_workers
                if item['active_run_id'] is not None
            ],
            'workers': workers,
        }

    # 训练循环里的廉价自检：确认“我仍是这个 Run 的合法执行者”（running + token
    # 匹配 + 租约未过期）。worker 在长阶段边界调用，一旦被取消或回收能尽早中止，
    # 而不是训练完才发现终态提交会失败。
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

    # 按 run_id 直接读取（不做权限过滤）。HTTP 层对外暴露时应优先用 get_scoped；
    # 本方法主要给 worker、内部状态机与测试使用。不存在时抛 RunNotFound。
    def get(self, run_id: str) -> RunRecord:
        with self._connection() as connection:
            row = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
        if row is None:
            raise RunNotFound(run_id)
        return self._record(row)

    # Principal → (owner_id, tenant_id) 的小工具。所有 *_scoped 查询统一走这里取 scope，
    # 保证“身份只来自服务端注入的 Principal”，不可能被请求体里的 owner_id 篡改。
    @staticmethod
    def _scope_values(principal: Principal) -> tuple[str | None, str | None]:
        return principal.owner_id, principal.tenant_id

    # 按 Principal 范围读取单个 Run：WHERE 用 `owner_id IS ?` 而非 `=`，
    # 因为 SQLite 中 NULL = NULL 不为真，IS 才能正确匹配“双方都是 NULL”的无主记录。
    # 匹配不到统一抛 RunNotFound——不区分“不存在”和“无权访问”，避免向他人
    # 泄露 run_id 是否真实存在（防枚举）。
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

    # 按 Principal 范围分页列出 Run，按 created_at DESC、run_id DESC 排序。
    # 游标分页：cursor 是上一页最后一条的 run_id，先查出它的 (created_at, run_id)，
    # 再用 (created_at < ? OR (created_at = ? AND run_id < ?)) 取“排在它之后”的记录——
    # 这种 keyset 分页比 OFFSET 稳定，新 Run 插入不会导致翻页重复/漏项；
    # 联合键第二列 run_id 用于打破 created_at 相同（同秒批量创建）时的并列。
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

    # cancel 的 Principal 范围版本：SELECT 与 UPDATE 都带 owner/tenant 条件，
    # 两道条件缺一不可——先查不到就抛 RunNotFound（不泄露存在性），并发下
    # UPDATE 的 WHERE 再兜一次底。这是 server 模式下取消接口必须使用的入口。
    def cancel_scoped(
        self,
        run_id: str,
        *,
        now: datetime,
        principal: Principal,
        reason: str = 'user_requested',
        message: str = '用户已停止训练',
    ) -> RunRecord:
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
                SET state = 'cancelled', version = version + 1, claim_token = NULL,
                    worker_id = NULL, lease_expires_at = NULL, manifest_name = NULL,
                    progress_json = ?, updated_at = ?, finished_at = ?
                WHERE run_id = ? AND state = ? AND owner_id IS ? AND tenant_id IS ?
                ''',
                (
                    _stopped_progress(
                        row['progress_json'],
                        stopped_at=now_text,
                        reason=reason,
                        message=message,
                    ),
                    now_text,
                    now_text,
                    run_id,
                    state,
                    owner_id,
                    tenant_id,
                ),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot cancel run {run_id}')
            updated = connection.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            connection.commit()
        assert updated is not None
        return self._record(updated)

    def cancel_queued_unstarted_scoped(
        self,
        run_id: str,
        *,
        now: datetime,
        principal: Principal,
        reason: str = 'agent_reconciliation',
        message: str = 'Agent 对账已取消未开始的 Run',
    ) -> RunRecord:
        """仅在同一事务内证明 queued 且 started_at 为空时取消。"""
        now_text = _timestamp(now)
        owner_id, tenant_id = self._scope_values(principal)
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                '''SELECT * FROM runs WHERE run_id=?
                   AND owner_id IS ? AND tenant_id IS ?''',
                (run_id, owner_id, tenant_id),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RunNotFound(run_id)
            if row['state'] != 'queued' or row['started_at'] is not None:
                connection.rollback()
                raise InvalidRunTransition(f'run {run_id} is not queued and unstarted')
            changed = connection.execute(
                '''UPDATE runs SET state='cancelled',version=version+1,claim_token=NULL,
                   worker_id=NULL,lease_expires_at=NULL,manifest_name=NULL,progress_json=?,
                   updated_at=?,finished_at=?
                   WHERE run_id=? AND state='queued' AND started_at IS NULL
                   AND owner_id IS ? AND tenant_id IS ?''',
                (
                    _stopped_progress(
                        row['progress_json'], stopped_at=now_text,
                        reason=reason, message=message,
                    ),
                    now_text, now_text, run_id, owner_id, tenant_id,
                ),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise InvalidRunTransition(f'cannot cancel unstarted run {run_id}')
            updated = connection.execute(
                'SELECT * FROM runs WHERE run_id=?', (run_id,)
            ).fetchone()
            connection.commit()
        assert updated is not None
        return self._record(updated)

    # 轻量存在性探测，不加载整行。注意：它不做 scope 过滤，只用于内部判断；
    # 对外接口不能用“exists 返回 True/False”区分他人 Run 的存在性。
    def exists(self, run_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute('SELECT 1 FROM runs WHERE run_id = ?', (run_id,)).fetchone()
        return row is not None

    # 显式把 owner/tenant 均为空的历史 Run 绑定到指定 Principal（migration rebind 用）。
    # 空 Principal 直接拒绝——把无主记录绑给“空身份”等于没绑，还会掩盖数据问题。
    # UPDATE 的 WHERE 再次限定双 NULL，并发下已被别人绑走则返回 False（幂等友好）。
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

    # 删除 Run 记录（仅终态）。先读后删、DELETE 的 WHERE 再限定一次终态集合，
    # 两道检查防并发：若读完后 Run 被回收重新排队（非终态），DELETE 影响 0 行即报错。
    # 注意这只删 DB 记录；磁盘产物目录的清理由上层另行处理。
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

    # delete_terminal 的 Principal 范围版本，server 模式删除接口的必选入口：
    # 查与删都带 owner/tenant 条件，越权删除他人 Run 会因查不到记录而报 RunNotFound。
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

    # 旧数据迁移导入：按给定 run_id/state 原样补录历史 Run。已存在同 run_id 时
    # 直接返回现有记录（幂等，可安全重跑迁移流程）。终态导入时 finished_at 缺省
    # 补当前时间，保证终态记录的时间字段完整。
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


    def save_evaluation_plan(self, plan, *, principal: Principal):
        from ..evaluation_plan import digest
        scope = digest(list(self._scope_values(principal)))
        body = plan.model_dump_json()
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT plan_json FROM evaluation_plans_v1 WHERE scope_digest=? AND plan_digest=?',
                (scope, plan.plan_digest)).fetchone()
            if row is not None and json.loads(row['plan_json']) != json.loads(body):
                raise ValueError('evaluation_plan_conflict')
            if row is None:
                connection.execute('INSERT INTO evaluation_plans_v1 VALUES(?,?,?)',(scope,plan.plan_digest,body))
            connection.commit()
        return plan

    def get_evaluation_plan(self, plan_digest: str, *, principal: Principal):
        from ..evaluation_plan import digest, EvaluationPlan
        scope = digest(list(self._scope_values(principal)))
        with self._connection() as connection:
            row = connection.execute('SELECT plan_json FROM evaluation_plans_v1 WHERE scope_digest=? AND plan_digest=?',
                (scope,plan_digest)).fetchone()
        if row is None:
            raise ValueError('evaluation_plan_unavailable')
        plan = EvaluationPlan.model_validate_json(row['plan_json'])
        if plan.plan_digest != plan_digest:
            raise ValueError('evaluation_plan_binding_mismatch')
        return plan
