"""单并发本地训练 worker 与 lease 心跳守护。

worker 轮询 queued Run、事务性 claim，然后用独立心跳线程续租。训练完成时仍必须
携带原 claim token；取消、lease 丢失或其他 worker 接管后，旧执行结果不能提交。
"""

from __future__ import annotations

import argparse
import logging
import socket
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Any

from .contracts import public_error_message
from .repository import InvalidRunTransition, RunRepository


logger = logging.getLogger(__name__)


def _error_details(exc: Exception) -> dict[str, Any]:
    """把 worker 异常收敛为可供结果页稳定展示的非敏感诊断。"""
    error_type = type(exc).__name__
    message = public_error_message(exc)
    if error_type == 'DatasetIntegrityError':
        code = 'dataset_changed'
        stage = 'dataset_validation'
        retryable = False
    elif isinstance(exc, (FileNotFoundError, PermissionError)):
        code = 'dataset_unavailable'
        stage = 'dataset_resolution'
        retryable = False
        message = '训练数据文件不可用，请重新上传或检查服务器存储'
    elif isinstance(exc, ValueError):
        code = 'invalid_training_data_or_config'
        stage = 'training_validation'
        retryable = False
    else:
        code = 'training_failed'
        stage = 'training'
        retryable = False
    return {
        'code': code,
        'stage': stage,
        'message': message,
        'type': error_type,
        'retryable': retryable,
    }


class LeaseGuard:
    """在训练执行期间周期续租，并记录首次心跳异常。"""
    def __init__(
        self,
        *,
        repository: RunRepository,
        run_id: str,
        claim_token: str,
        worker_id: str,
        now: Callable[[], datetime],
        heartbeat_seconds: float,
    ) -> None:
        self.repository = repository
        self.run_id = run_id
        self.claim_token = claim_token
        self.worker_id = worker_id
        self.now = now
        self.heartbeat_seconds = heartbeat_seconds
        self.stop = Event()
        self.thread: Thread | None = None
        self.lost = Event()

    def __enter__(self) -> 'LeaseGuard':
        def renew() -> None:
            while not self.stop.wait(self.heartbeat_seconds):
                try:
                    self.repository.renew_lease(
                        self.run_id,
                        claim_token=self.claim_token,
                        now=self.now(),
                    )
                    self.repository.record_worker_heartbeat(
                        worker_id=self.worker_id,
                        now=self.now(),
                        active_run_id=self.run_id,
                    )
                except InvalidRunTransition:
                    self.lost.set()
                    return

        self.thread = Thread(target=renew, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


class RunWorker:
    """按单并发循环领取 queued Run，并把执行结果提交回仓库。"""
    def __init__(
        self,
        *,
        repository: RunRepository,
        worker_id: str,
        execute: Callable[[Any], dict[str, Any]],
        now: Callable[[], datetime],
        heartbeat_seconds: float = 5.0,
        project_status: Callable[[Any], None] | None = None,
    ) -> None:
        self.repository = repository
        self.worker_id = worker_id
        self.execute = execute
        self.now = now
        self.heartbeat_seconds = heartbeat_seconds
        self.project_status = project_status

    def run_once(self) -> bool:
        self.repository.record_worker_heartbeat(worker_id=self.worker_id, now=self.now())
        self.repository.requeue_expired(now=self.now())
        run = self.repository.claim_next(worker_id=self.worker_id, now=self.now())
        if run is None:
            return False
        try:
            with LeaseGuard(
                repository=self.repository,
                run_id=run.run_id,
                claim_token=run.claim_token or '',
                worker_id=self.worker_id,
                now=self.now,
                heartbeat_seconds=self.heartbeat_seconds,
            ):
                self.repository.record_worker_heartbeat(
                    worker_id=self.worker_id,
                    now=self.now(),
                    active_run_id=run.run_id,
                )
                result = self.execute(run)
            finished = self.repository.finish_success(
                run.run_id,
                claim_token=run.claim_token or '',
                now=self.now(),
                manifest_name=str(result['manifest_name']),
            )
            if self.project_status is not None:
                self.project_status(finished)
            self.repository.record_worker_heartbeat(worker_id=self.worker_id, now=self.now())
        except InvalidRunTransition:
            return True
        except Exception as exc:
            try:
                logger.exception('Run %s failed during worker execution', run.run_id)
                details = _error_details(exc)
                failed = self.repository.finish_failure(
                    run.run_id,
                    claim_token=run.claim_token or '',
                    now=self.now(),
                    error=str(details['message']),
                    error_details=details,
                )
                if self.project_status is not None:
                    self.project_status(failed)
                self.repository.record_worker_heartbeat(worker_id=self.worker_id, now=self.now())
            except InvalidRunTransition:
                pass
        return True


def utc_now() -> datetime:
    """提供可在测试中替换的 UTC 时钟。"""
    return datetime.now(timezone.utc)


def main() -> None:
    """解析轮询/lease 参数并持续运行 worker，直到进程被终止。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=0.5)
    args = parser.parse_args()

    from ..paths import RUNS_DATABASE, RUNS_DIR
    from .status_projection import project_status

    def execute_claimed(record: Any) -> dict[str, str]:
        from .execution import execute_claimed_run

        return execute_claimed_run(record, repository=repository)

    repository = RunRepository(RUNS_DATABASE)
    repository.initialize()
    worker = RunWorker(
        repository=repository,
        worker_id=f'{socket.gethostname()}-{uuid.uuid4().hex[:8]}',
        execute=execute_claimed,
        now=utc_now,
        project_status=lambda record: project_status(RUNS_DIR / record.run_id, record),
    )
    while worker.run_once() or not args.once:
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
