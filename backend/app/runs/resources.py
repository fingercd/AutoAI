"""Local process/thread budget; no training imports at launcher startup."""
from __future__ import annotations

import os

THREAD_VARIABLES = (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
)


def worker_resource_plan(workers: int | None = None, threads: int | None = None, *, cpu_count: int | None = None) -> tuple[int, int]:
    cpus = max(1, cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    count = workers if workers is not None else min(4, max(1, cpus - 1))
    if not 1 <= count <= cpus:
        raise ValueError(f'Worker 数必须为 1..{cpus}（可用逻辑 CPU 数）')
    budget = threads if threads is not None else max(1, (cpus - 1) // count)
    if budget < 1 or count * budget > cpus:
        raise ValueError(f'Worker 数 × 每 Worker 线程数不能超过 {cpus}')
    return count, budget


def worker_environment(threads: int, workers: int, *, base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment.update({key: str(threads) for key in THREAD_VARIABLES})
    environment.update(AUTOAI_WORKER_THREADS=str(threads), AUTOAI_WORKER_COUNT=str(workers))
    return environment
