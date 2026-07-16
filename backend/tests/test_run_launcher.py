from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from run import WorkerSupervisor


ROOT = Path(__file__).resolve().parents[2]


class _FakeProcess:
    def __init__(self, *, timeout_once: bool = False) -> None:
        self.timeout_once = timeout_once
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float] = []

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float):
        self.wait_timeouts.append(timeout)
        if self.timeout_once:
            self.timeout_once = False
            raise subprocess.TimeoutExpired(cmd='worker', timeout=timeout)
        return 0


def test_worker_supervisor_stop_terminates_and_waits_for_child() -> None:
    process = _FakeProcess()
    supervisor = WorkerSupervisor([sys.executable, '-m', 'backend.app.runs.worker'])
    supervisor._process = process  # type: ignore[assignment]

    supervisor.stop()

    assert process.terminated is True
    assert process.killed is False
    assert process.wait_timeouts == [10]


def test_worker_supervisor_stop_kills_child_after_graceful_timeout() -> None:
    process = _FakeProcess(timeout_once=True)
    supervisor = WorkerSupervisor([sys.executable, '-m', 'backend.app.runs.worker'])
    supervisor._process = process  # type: ignore[assignment]

    supervisor.stop()

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_timeouts == [10, 5]


def test_launcher_rejects_non_loopback_bind_without_server_mode() -> None:
    environment = dict(os.environ)
    for name in (
        'AUTOAI_DEPLOYMENT_MODE',
        'AUTOAI_API_TOKEN',
        'AUTOAI_PRINCIPAL_ID',
        'AUTOAI_TENANT_ID',
        'AUTOAI_ALLOWED_ORIGINS',
    ):
        environment.pop(name, None)

    completed = subprocess.run(
        [
            sys.executable,
            'run.py',
            '--host',
            '0.0.0.0',
            '--no-browser',
            '--no-worker',
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=15,
    )

    assert completed.returncode == 2
    assert '对外绑定必须同时启用 --server' in completed.stderr


def test_launcher_server_mode_fails_fast_without_token() -> None:
    environment = dict(os.environ)
    environment.pop('AUTOAI_API_TOKEN', None)
    environment.pop('AUTOAI_DEPLOYMENT_MODE', None)

    completed = subprocess.run(
        [
            sys.executable,
            'run.py',
            '--server',
            '--host',
            '0.0.0.0',
            '--no-browser',
            '--no-worker',
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=15,
    )

    assert completed.returncode == 2
    assert 'AUTOAI_API_TOKEN' in completed.stderr


def test_launcher_help_documents_server_and_worker_controls() -> None:
    completed = subprocess.run(
        [sys.executable, 'run.py', '--help'],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=15,
    )

    assert completed.returncode == 0
    assert '--server' in completed.stdout
    assert '--no-worker' in completed.stdout


def test_cluster_launcher_defaults_to_loopback_and_fails_fast_for_external_server() -> None:
    script = (ROOT / 'deploy' / 'run_on_node3.sh').read_text(encoding='utf-8')

    assert 'HOST="${HOST:-127.0.0.1}"' in script
    assert 'export AUTOAI_DEPLOYMENT_MODE="server"' in script
    assert 'server mode requires AUTOAI_API_TOKEN with at least 32 characters' in script
    assert 'echo "$AUTOAI_API_TOKEN"' not in script
