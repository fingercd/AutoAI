from __future__ import annotations

import argparse
import socket
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Any

from .repository import InvalidRunTransition, RunRepository


class LeaseGuard:
    def __init__(
        self,
        *,
        repository: RunRepository,
        run_id: str,
        claim_token: str,
        now: Callable[[], datetime],
        heartbeat_seconds: float,
    ) -> None:
        self.repository = repository
        self.run_id = run_id
        self.claim_token = claim_token
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
        self.repository.requeue_expired(now=self.now())
        run = self.repository.claim_next(worker_id=self.worker_id, now=self.now())
        if run is None:
            return False
        try:
            with LeaseGuard(
                repository=self.repository,
                run_id=run.run_id,
                claim_token=run.claim_token or '',
                now=self.now,
                heartbeat_seconds=self.heartbeat_seconds,
            ):
                result = self.execute(run)
            finished = self.repository.finish_success(
                run.run_id,
                claim_token=run.claim_token or '',
                now=self.now(),
                manifest_name=str(result['manifest_name']),
            )
            if self.project_status is not None:
                self.project_status(finished)
        except InvalidRunTransition:
            return True
        except Exception as exc:
            try:
                failed = self.repository.finish_failure(
                    run.run_id,
                    claim_token=run.claim_token or '',
                    now=self.now(),
                    error=str(exc),
                )
                if self.project_status is not None:
                    self.project_status(failed)
            except InvalidRunTransition:
                pass
        return True


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def main() -> None:
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
