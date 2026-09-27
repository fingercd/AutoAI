"""Unprivileged Linux supervision using a dedicated, single-threaded subreaper.

The worker's private pipe is a liveness lease; it is never inherited by training.
The guardian survives worker SIGKILL and reaps even double-fork/setsid descendants.
Only kernel-confirmed children are signalled, before reaping (no PID reuse race).
This is a trusted workload boundary, not a sandbox against hostile same-UID code.
"""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time


def _reap() -> bool:
    """True only when waitpid proves there are no children, including adoptees."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True
        if pid == 0:
            return False


def _kill_and_reap() -> None:
    # Do not abandon a D-state child after a timeout. The worker independently
    # times out and holds its reservation; the guardian continues cleanup.
    children = Path(f'/proc/self/task/{os.getpid()}/children')
    while True:
        # No wait/reap or SIGCHLD handler between enumeration and signalling.
        # Every listed PID remains our live child or our unreaped zombie.
        for text in children.read_text().split():
            try:
                os.kill(int(text), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if _reap():
            return
        time.sleep(0.01)


def _guardian() -> None:
    # Avoid buffered stdin read-ahead: subsequent EOF is the worker death signal.
    control = sys.stdin.buffer
    request = json.loads(control.readline())
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'cannot establish child subreaper')
    # Fail before starting training if this kernel lacks direct-child discovery.
    Path(f'/proc/self/task/{os.getpid()}/children').read_text()
    deadline = datetime.fromisoformat(request['deadline_at'])
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    until = time.monotonic() + remaining
    process = None
    response = None
    try:
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as output:
            # Parent may have died during guardian startup. Never start work then.
            if (not request['start_allowed'] or
                    select.select([control], [], [], 0)[0] or remaining <= 0):
                response = dict(tree_empty=True, stopped=True, exit_code=None)
            else:
                process = subprocess.Popen(
                    [sys.executable, '-m', request['module'], '--child'],
                    stdin=subprocess.PIPE, stdout=output, stderr=None,
                    text=True, close_fds=True, start_new_session=True)
                assert process.stdin is not None
                process.stdin.write(json.dumps(request['payload']) + '\n')
                process.stdin.close()
                stopped = False
                while process.poll() is None:
                    if (select.select([control], [], [], 0.02)[0]
                            or time.monotonic() >= until
                            or datetime.now(timezone.utc) >= deadline):
                        stopped = True
                        break
                exit_code = process.poll()
                # poll reaps only the root. Other descendants remain ours;
                # ECHILD is the final proof, not root exit or a process-group scan.
                descendants = not _reap() if exit_code is not None else True
                _kill_and_reap()
                if exit_code is None:
                    exit_code = -signal.SIGKILL
                output.seek(0)
                response = dict(tree_empty=True, stopped=stopped,
                                descendants=descendants, exit_code=exit_code,
                                output=output.read())
    finally:
        _kill_and_reap()
    # The proof is emitted only after kernel-confirmed empty tree.
    sys.stdout.write(json.dumps(response))
    sys.stdout.flush()


def supervise_linux(payload, *, deadline_at, active, module, poll_seconds,
                    termination_grace_seconds, on_stop):
    from .supervisor import (TerminationEvidence, SupervisionStopped,
                             SupervisionUncertain, ChildExecutionError,
                             _child_environment, _decode_result)
    until = time.monotonic() + (deadline_at - datetime.now(timezone.utc)).total_seconds()
    def reason():
        if time.monotonic() >= until or datetime.now(timezone.utc) >= deadline_at:
            return 'deadline'
        return None if active() else 'claim_inactive'
    output = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
    guardian = subprocess.Popen(
        [sys.executable, '-m', 'backend.app.runs.linux_supervisor'],
        cwd=str(Path(__file__).resolve().parents[3]), env=_child_environment(),
        stdin=subprocess.PIPE, stdout=output, text=True,
        close_fds=True, start_new_session=True)
    assert guardian.stdin is not None
    requested = datetime.now(timezone.utc)
    start = time.monotonic()
    stop_reason = None
    try:
        stop_reason = reason()
        guardian.stdin.write(json.dumps(dict(payload=payload, module=module,
                                             start_allowed=stop_reason is None,
                                             deadline_at=deadline_at.isoformat())) + '\n')
        guardian.stdin.flush()
        while guardian.poll() is None and stop_reason is None:
            stop_reason = reason()
            if stop_reason:
                requested = datetime.now(timezone.utc)
                start = time.monotonic()
                guardian.stdin.close()
                break
            time.sleep(min(poll_seconds, max(0.001, until - time.monotonic())))
        if stop_reason is None:
            stop_reason = reason()
        try:
            guardian.wait(timeout=termination_grace_seconds)
            output.seek(0)
            proof = json.loads(output.read())
        except (subprocess.TimeoutExpired, ValueError):
            proof = None
        if (guardian.returncode != 0 or not isinstance(proof, dict)
                or proof.get('tree_empty') is not True):
            evidence = TerminationEvidence(stop_reason or 'guardian_lost',
                                          requested.isoformat(), None, None, None)
            if on_stop:
                on_stop(evidence)
            raise SupervisionUncertain(evidence)
        if stop_reason or proof.get('stopped'):
            evidence = TerminationEvidence(stop_reason or 'deadline', requested.isoformat(),
                datetime.now(timezone.utc).isoformat(), time.monotonic() - start,
                proof.get('exit_code'))
            if on_stop:
                on_stop(evidence)
            raise SupervisionStopped(evidence)
        if proof.get('descendants'):
            raise ChildExecutionError('ChildTreeStillActive', 'training child left active descendants')
        if proof.get('exit_code') != 0:
            raise ChildExecutionError('ChildProcessExit', 'training child failed')
        return _decode_result(proof.get('output'))
    finally:
        # An active callback error also revokes the guardian's lease. Never kill
        # the guardian: it must remain alive until all training descendants exit.
        if not guardian.stdin.closed:
            guardian.stdin.close()
        try:
            guardian.wait(timeout=termination_grace_seconds)
        except subprocess.TimeoutExpired:
            pass  # caller's unknown hold remains; guardian continues reaping
        output.close()


if __name__ == '__main__':
    _guardian()
