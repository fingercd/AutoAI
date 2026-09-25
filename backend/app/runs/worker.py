"""单并发本地训练 worker 与 lease 心跳守护。

worker 轮询 queued Run、事务性 claim，然后用独立心跳线程续租。训练完成时仍必须
携带原 claim token；取消、lease 丢失或其他 worker 接管后，旧执行结果不能提交。

系统位置：runs 子系统的执行层。HTTP 训练路由只负责把 Run 写成 queued，
真正的训练由本模块的 worker 进程领取执行；与 execution.py（单次 Run 执行边界）、
repository.py（Run 状态机与 lease 存储）、status_projection.py（磁盘状态投影）协作。

关键设计约束：
- 单并发：一个 worker 进程同一时刻只执行一个 Run，避免本地训练互相争抢资源。
- claim token 防护：claim / 续租 / 完成提交都必须携带同一个 claim_token；
  lease 过期或 Run 被取消后，旧 worker 的迟到提交会被 InvalidRunTransition 拒绝。
- worker 心跳与 Run lease 分离：前者用于运维可观测，后者用于所有权判定。
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Any

from .contracts import public_error_message
from .repository import InvalidRunTransition, RunDeadlineExceeded, RunRepository
from .supervisor import (ChildExecutionError, SupervisionStopped,
                         SupervisionUncertain, TerminationEvidence)
from ..version import WORKER_CONTRACT_VERSION


logger = logging.getLogger(__name__)


def _error_details(exc: Exception) -> dict[str, Any]:
    """把 worker 异常收敛为可供结果页稳定展示的非敏感诊断。

    参数:
        exc: 训练执行或数据集解析阶段抛出的任意异常。
    返回:
        包含 code/stage/message/type/retryable 的字典，会写入 Run 的 error_details，
        供结果页（run-result-v1）直接展示，前端只需识别 code。
    设计意图:
        异常类型到业务错误码的映射集中在这一处；message 经 public_error_message 过滤，
        避免把服务器内部细节泄露给浏览器。
    """
    error_type = type(exc).__name__
    message = public_error_message(exc)
    from .guard import GuardError
    if isinstance(exc, GuardError):
        return dict(code=exc.code, stage=exc.stage, report_id=exc.report_id,
                    message='训练完整性检查未通过', type='GuardError', retryable=False)
    # 数据集在 queued 之后被改动（SHA-256 校验失败）：独立 code，便于前端给出针对性提示
    if error_type == 'DatasetIntegrityError':
        code = 'dataset_changed'
        stage = 'dataset_validation'
        retryable = False
    elif isinstance(exc, (SupervisionStopped, RunDeadlineExceeded)):
        code = 'budget_deadline_exceeded' if (
            isinstance(exc, RunDeadlineExceeded) or
            exc.evidence.reason == 'deadline') else 'run_claim_lost'
        stage = 'training_supervision'
        retryable = False
        message = '训练已达到冻结截止时间' if code == 'budget_deadline_exceeded' else '训练执行权已失效'
    elif getattr(exc, 'code', None) == 'budget_deadline_exceeded':
        code = 'budget_deadline_exceeded'
        stage = 'training_supervision'
        retryable = False
        message = '训练已达到冻结截止时间'
    # 数据文件丢失/无权限：用固定中文提示覆盖 message，避免把服务器存储路径暴露给浏览器
    elif isinstance(exc, (FileNotFoundError, PermissionError)):
        code = 'dataset_unavailable'
        stage = 'dataset_resolution'
        retryable = False
        message = '训练数据文件不可用，请重新上传或检查服务器存储'
    # 训练数据或配置不合法（如每类 Sample_ID 不足、批次公共轴不一致等校验错误）
    elif isinstance(exc, ValueError):
        code = 'invalid_training_data_or_config'
        stage = 'training_validation'
        retryable = False
    # 兜底：未预期的训练内部错误
    else:
        code = 'training_failed'
        stage = 'training'
        retryable = False
    # 当前所有失败 retryable=False：自动重试无法修复数据/配置/文件类错误，避免无谓重跑
    return {
        'code': code,
        'stage': stage,
        'message': message,
        'type': error_type,
        'retryable': retryable,
    }


class LeaseGuard:
    """在训练执行期间周期续租，并记录 lease 丢失事件。

    作为上下文管理器使用：进入时启动守护线程，退出时停止并回收线程。
    训练可能持续几分钟到几小时，而 lease 有效期远短于此；只有持续 renew_lease
    才能证明"本 worker 仍活着并持有该 Run"。一旦续租抛 InvalidRunTransition
    （Run 被取消、lease 被回收或被其他 worker 接管），置位 self.lost，
    此后旧 worker 的所有提交都会被仓库拒绝。
    """
    def __init__(
        self,
        *,
        repository: RunRepository,
        run_id: str,
        claim_token: str,
        worker_id: str,
        now: Callable[[], datetime],
        heartbeat_seconds: float,
        on_lease_lost: Callable[[str], None] | None = None,
    ) -> None:
        self.repository = repository
        self.run_id = run_id
        self.claim_token = claim_token
        self.worker_id = worker_id
        self.now = now
        self.heartbeat_seconds = heartbeat_seconds
        self.on_lease_lost = on_lease_lost
        self.stop = Event()
        self.thread: Thread | None = None
        self.lost = Event()
        # stop 控制守护线程退出；lost 记录 lease 是否已丢失（取消/回收/被接管）

    def __enter__(self) -> 'LeaseGuard':
        def renew() -> None:
            while not self.stop.wait(self.heartbeat_seconds):
                # wait 同时充当 sleep：stop 置位会立即唤醒，保证退出时延可控
                try:
                # 先续 Run 的 lease（所有权判定），再记录 worker 心跳（运维可观测）
                    self.repository.renew_lease(
                        self.run_id,
                        claim_token=self.claim_token,
                        now=self.now(),
                    )
                    self.repository.record_worker_heartbeat(
                        worker_id=self.worker_id,
                        now=self.now(),
                        active_run_id=self.run_id,
                        contract_version=WORKER_CONTRACT_VERSION,
                    )
                except InvalidRunTransition:
                    # lease 已丢失：取消/回收/被接管都会走到这里；置位 lost 后线程退出，不再做无谓续租
                    self.lost.set()
                    if self.on_lease_lost is not None:
                        self.on_lease_lost(self.run_id)
                    return

        self.thread = Thread(target=renew, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            # 只等 2 秒：线程是 daemon，进程退出时即使没 join 完也不会悬挂


class RunWorker:
    """按单并发循环领取 queued Run，并把执行结果提交回仓库。

    参数:
        repository: Run 状态机存储，负责 claim/续租/完成提交。
        worker_id: 本 worker 实例的唯一标识（主机名+随机串），用于心跳审计。
        execute: 真正执行训练的回调（见 execution.execute_claimed_run），
            依赖注入便于测试替换。
        now: 时钟回调，注入式 UTC 时钟便于测试控制时间。
        heartbeat_seconds: lease 续租与心跳间隔。
        project_status: 可选的状态投影回调，训练结束后刷新磁盘上的 status.json。
    """
    def __init__(
        self,
        *,
        repository: RunRepository,
        worker_id: str,
        execute: Callable[[Any], dict[str, Any]],
        now: Callable[[], datetime],
        heartbeat_seconds: float = 5.0,
        project_status: Callable[[Any], None] | None = None,
        discard_artifacts: Callable[[str], None] | None = None,
        on_lease_lost: Callable[[str], None] | None = None,
        on_stopped_run: Callable[[str], None] | None = None,
    ) -> None:
        self.repository = repository
        self.worker_id = worker_id
        self.execute = execute
        self.now = now
        self.heartbeat_seconds = heartbeat_seconds
        self.project_status = project_status
        self.discard_artifacts = discard_artifacts
        self.on_lease_lost = on_lease_lost
        self.on_stopped_run = on_stopped_run

    def run_once(self) -> bool:
        """执行一轮"领取 → 训练 → 提交"循环。

        返回:
            True 表示本轮领到了 Run（无论成功或失败），False 表示队列空闲；
            调用方据此决定继续轮询还是（--once 模式）退出。
        """
        # 每轮先上报心跳：即使队列空闲，也能从心跳表看出 worker 存活
        self.repository.record_worker_heartbeat(
            worker_id=self.worker_id,
            now=self.now(),
            contract_version=WORKER_CONTRACT_VERSION,
        )
        # 租约失效代表原训练执行已中断：统一记为 STOP，不再静默重跑。
        for stopped_run_id in self.repository.stop_expired(now=self.now()):
            if self.on_stopped_run is not None:
                self.on_stopped_run(stopped_run_id)
            if self.discard_artifacts is not None:
                self.discard_artifacts(stopped_run_id)
        # 事务性 claim：同一 Run 只会被一个 worker 领到，claim_token 是后续所有写操作的凭证
        run = self.repository.claim_next(worker_id=self.worker_id, now=self.now())
        if run is None:
            return False
        try:
            # 训练期间由守护线程持续续租；claim_token 缺失时退化为空串（历史兼容）
            with LeaseGuard(
                repository=self.repository,
                run_id=run.run_id,
                claim_token=run.claim_token or '',
                worker_id=self.worker_id,
                now=self.now,
                heartbeat_seconds=self.heartbeat_seconds,
                on_lease_lost=self.on_lease_lost,
            ):
                self.repository.record_worker_heartbeat(
                    worker_id=self.worker_id,
                    now=self.now(),
                    active_run_id=run.run_id,
                    contract_version=WORKER_CONTRACT_VERSION,
                )
                # 真正的训练执行（耗时主体）；期间的取消由 execution 内部的 cancel_check 感知
                result = self.execute(run)
                # Hash I/O and the final publication fence remain under lease renewal.
                deadline_text = result.get('_work_deadline_at')
                publication = None
                if run.guard_policy is not None:
                    from .guard import check_outputs, GuardError
                    from .contracts import Principal
                    def active():
                        self.repository.assert_active(run.run_id, claim_token=run.claim_token or '', now=self.now())
                        if deadline_text and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline_text):
                            raise RunDeadlineExceeded('publication deadline exceeded')
                    child = self.repository.get_guard_report(result.get('guard_report_id'),
                        principal=Principal(run.owner_id, run.tenant_id))
                    publication = check_outputs(record=run, repository=self.repository,
                        run_dir=self.repository.database_path.parent/'runs'/run.run_id,
                        stage='publication', active=active, usage=result.get('_guard_usage'))
                    if child.stage != 'pre_publish' or child.status != 'passed' or child.bindings != publication.bindings:
                        raise GuardError('guard_report_binding_mismatch', stage='publication')
                finished = self.repository.finish_success(
                    run.run_id, claim_token=run.claim_token or '', now=self.now(),
                    manifest_name=str(result['manifest_name']),
                    **({'publication_report': publication} if publication is not None else {}),
                    **({'deadline_at': datetime.fromisoformat(deadline_text)} if deadline_text else {}))
            if self.project_status is not None:
                self.project_status(finished)
            self.repository.record_worker_heartbeat(
                worker_id=self.worker_id,
                now=self.now(),
                contract_version=WORKER_CONTRACT_VERSION,
            )
        # lease 丢失/状态被并发改写：本 worker 的结果作废，返回 True 继续下一轮
        except SupervisionUncertain:
            # Exit was not confirmed.  Never schedule another Run in this
            # execution slot or claim that the held training capacity is free.
            raise
        except InvalidRunTransition:
            if self.discard_artifacts is not None and 'execution_search_plan' not in run.config:
                self.discard_artifacts(run.run_id)
            return True
        except Exception as exc:
            try:
                # 失败提交同样携带 claim_token；错误详情收敛后写入仓库供结果页展示
                logger.exception('Run %s failed during worker execution', run.run_id)
                details = _error_details(exc)
                if run.guard_policy is not None:
                    # Unexpected I/O/model exceptions are not a public evidence
                    # channel, even when their text happens not to contain a path.
                    details['message'] = '训练检查或执行未完成；详细信息仅记录在服务器日志中'
                failed = self.repository.finish_failure(
                    run.run_id,
                    claim_token=run.claim_token or '',
                    now=self.now(),
                    error=str(details['message']),
                    error_details=details,
                )
                if self.project_status is not None:
                    self.project_status(failed)
                self.repository.record_worker_heartbeat(
                    worker_id=self.worker_id,
                    now=self.now(),
                    contract_version=WORKER_CONTRACT_VERSION,
                )
            # 提交失败结果时发现 lease 已丢失：状态已被新持有者改写，不再覆盖
            except InvalidRunTransition:
                pass
        return True


def utc_now() -> datetime:
    """提供可在测试中替换的 UTC 时钟。

    RunWorker/LeaseGuard 不直接调用它，而是通过 now 回调注入；
    只有 main() 在真实运行时把它接进去，测试里可注入固定时间。
    """
    return datetime.now(timezone.utc)


def execute_with_budget_supervision(record: Any, *, repository: RunRepository,
                                    agent_database: Any) -> dict[str, Any]:
    """Route only a server-bound budget Run through the child supervisor."""
    from ..agent.budget import BudgetError
    from ..agent.training_budget import TrainingBudget, load_training_binding
    from .execution import execute_claimed_run
    from .supervisor import supervise
    from .guard import GuardError

    binding = load_training_binding(agent_database, record,
                                    runs_database=repository.database_path)
    if binding is None:
        return execute_claimed_run(record, repository=repository)

    def check_claim() -> None:
        repository.assert_active(record.run_id, claim_token=record.claim_token or '',
                                 now=datetime.now(timezone.utc))

    def active() -> bool:
        try:
            check_claim()
        except InvalidRunTransition:
            return False
        return True

    ledger = TrainingBudget(binding.agent_database, task_id=binding.task_id,
                            policy_digest=binding.policy_digest,
                            reservation_id=binding.reservation_id,
                            run_id=record.run_id,
                            claim_token=record.claim_token or '',
                            claim_check=check_claim)
    try:
        ledger.bind()
    except BudgetError as exc:
        if exc.code == 'budget_deadline_exceeded':
            ledger.settle_unstarted()
        raise
    deadline = datetime.fromtimestamp(binding.work_deadline_at, timezone.utc)
    payload = dict(kind='training-run-v1', run_id=record.run_id,
                   claim_token=record.claim_token,
                   runs_database=str(repository.database_path),
                   training_budget=binding.child_payload())
    try:
        if record.guard_policy is not None:
            from .guard import require_current_policy, report
            try:
                require_current_policy(record.guard_policy)
            except GuardError as exc:
                checked = report(record, 'pre_fit', reason=exc.code)
                repository.save_guard_report(checked)
                raise GuardError(exc.code, report_id=checked.report_id) from exc
        result = supervise(payload, deadline_at=deadline, active=active,
                           on_stop=ledger.record_termination)
    except (SupervisionStopped, ChildExecutionError, GuardError):
        ledger.settle(exit_confirmed=True)
        raise
    except SupervisionUncertain:
        ledger.settle(exit_confirmed=False)
        raise
    except BaseException as exc:
        ledger.settle(exit_confirmed=False)
        raise SupervisionUncertain(TerminationEvidence(
            'supervisor_error', datetime.now(timezone.utc).isoformat(),
            None, None, None)) from exc
    ledger.settle(exit_confirmed=True)
    return {**result, '_work_deadline_at': deadline.isoformat(),
            **({'_guard_usage': ledger.guard_snapshot(settled=True)} if record.guard_policy is not None else {})}


def main() -> None:
    """解析轮询/lease 参数并持续运行 worker，直到进程被终止。

    --once 用于测试/调试：只尝试领取一个 Run 后退出。
    --contract-version 由启动器传入，防止新旧 worker 与数据库中的 Run
    产物契约不一致时误启动。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=0.5)
    parser.add_argument(
        '--contract-version',
        default=None,
        help='启动器期望的 Worker 产物契约；不一致时拒绝启动',
    )
    args = parser.parse_args()
    # 启动器与 worker 可能版本不一：契约版本不一致时直接拒绝启动，防止写出不兼容产物
    if args.contract_version and args.contract_version != WORKER_CONTRACT_VERSION:
        parser.error(
            'Worker 契约版本与启动器不一致：'
            f'expected={args.contract_version}, actual={WORKER_CONTRACT_VERSION}'
        )

    # 延迟导入：模块加载期不触碰存储路径，便于测试在临时目录下替换
    from ..paths import RUNS_DATABASE, RUNS_DIR
    from .artifacts import discard_run_artifacts
    from .status_projection import project_status
    from ..agent.training_budget import reconcile_orphaned_run

    # 同样延迟导入 execution/training：worker 空闲时不加载 torch/sklearn 等重依赖
    def execute_claimed(record: Any) -> dict[str, str]:
        from ..paths import AGENT_DATABASE
        return execute_with_budget_supervision(record, repository=repository,
                                               agent_database=AGENT_DATABASE)

    repository = RunRepository(RUNS_DATABASE)
    repository.initialize()

    def discard_artifacts(run_id: str) -> None:
        try:
            discard_run_artifacts(RUNS_DIR / run_id)
        except OSError:
            logger.warning('Run %s stopped but artifact cleanup must be retried', run_id)

    def reconcile_stopped(run_id: str) -> None:
        from ..paths import AGENT_DATABASE
        reconcile_orphaned_run(AGENT_DATABASE, repository.get(run_id),
                               runs_database=repository.database_path)

    def exit_after_lease_loss(run_id: str) -> None:
        if repository.get(run_id).config.get('execution_budget_task_id'):
            # The supervisor polls this claim every 0.25s and owns the child
            # Job Object; os._exit here would skip its termination evidence.
            return
        # AggMap/SciPy 等原生调用无法可靠地用 Python 异常打断。停止状态已由
        # SQLite 提交后，退出 worker 才能立即释放 CPU/GPU；启动器会重新拉起。
        discard_artifacts(run_id)
        os._exit(75)

    # 上一次异常退出可能留下 STOP Run 的半成品，worker 启动时补做清理。
    for stopped_record in repository.list():
        if stopped_record.state == 'cancelled':
            reconcile_stopped(stopped_record.run_id)
            discard_artifacts(stopped_record.run_id)
    worker = RunWorker(
        repository=repository,
        worker_id=f'{socket.gethostname()}-{uuid.uuid4().hex[:8]}',
        # worker_id = 主机名 + 8 位随机 hex：多机/多进程部署时可区分心跳来源
        execute=execute_claimed,
        now=utc_now,
        project_status=lambda record: project_status(RUNS_DIR / record.run_id, record),
        discard_artifacts=discard_artifacts,
        on_lease_lost=exit_after_lease_loss,
        on_stopped_run=reconcile_stopped,
    )
    while worker.run_once() or not args.once:
        # 领到任务立即进入下一轮（可能还有积压）；空闲且非 once 模式则休眠后重试
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
