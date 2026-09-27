"""Platform process-tree supervision for a single claimed training Run.

The worker owns the claim and the Job Object.  A child cannot begin work until
it has been assigned to that job.  Closing the last job handle after a parent
crash kills the child and any processes it spawned.
Linux uses a dedicated subreaper with a private worker-liveness pipe.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable


@dataclass(frozen=True)
class TerminationEvidence:
    reason: str
    requested_at: str
    exited_at: str | None
    latency_seconds: float | None
    exit_code: int | None


class SupervisionStopped(RuntimeError):
    def __init__(self, evidence: TerminationEvidence) -> None:
        super().__init__(f'supervised Run stopped: {evidence.reason}')
        self.evidence = evidence


class SupervisionUncertain(RuntimeError):
    def __init__(self, evidence: TerminationEvidence) -> None:
        super().__init__('supervised Run process exit could not be confirmed')
        self.evidence = evidence


class ChildExecutionError(RuntimeError):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


if sys.platform == 'win32':
    class _BasicLimit(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', ctypes.c_longlong),
            ('PerJobUserTimeLimit', ctypes.c_longlong),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.c_size_t),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]


    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
            'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


    class _ExtendedLimit(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', _BasicLimit),
                    ('IoInfo', _IoCounters),
                    ('ProcessMemoryLimit', ctypes.c_size_t),
                    ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t),
                    ('PeakJobMemoryUsed', ctypes.c_size_t)]


    class _BasicAccounting(ctypes.Structure):
        _fields_ = [('TotalUserTime', ctypes.c_longlong),
                    ('TotalKernelTime', ctypes.c_longlong),
                    ('ThisPeriodTotalUserTime', ctypes.c_longlong),
                    ('ThisPeriodTotalKernelTime', ctypes.c_longlong),
                    ('TotalPageFaultCount', wintypes.DWORD),
                    ('TotalProcesses', wintypes.DWORD),
                    ('ActiveProcesses', wintypes.DWORD),
                    ('TotalTerminatedProcesses', wintypes.DWORD)]


    _kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                wintypes.LPVOID, wintypes.DWORD]
    _kernel.SetInformationJobObject.restype = wintypes.BOOL
    _kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel.TerminateJobObject.restype = wintypes.BOOL
    _kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                  wintypes.LPVOID, wintypes.DWORD,
                                                  ctypes.POINTER(wintypes.DWORD)]
    _kernel.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL


def _job_create() -> int:
    if sys.platform != 'win32':
        raise RuntimeError('supervised Run execution is supported on Windows only')
    info = _ExtendedLimit()
    info.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
    handle = _kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if not _kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        _kernel.CloseHandle(handle)
        raise ctypes.WinError(error)
    return handle


def _child_environment() -> dict[str, str]:
    allowed = {
        'SYSTEMROOT', 'WINDIR', 'PATH', 'PATHEXT', 'TEMP', 'TMP',
        'PROGRAMFILES', 'PROGRAMFILES(X86)', 'PROGRAMW6432',
        'CUDA_VISIBLE_DEVICES', 'CUDA_PATH', 'CUDA_HOME',
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX',
        'HOME', 'TMPDIR', 'LD_LIBRARY_PATH', 'LANG', 'LC_ALL', 'PYTHONUTF8',
    }
    return {key: value for key, value in os.environ.items() if key.upper() in allowed}


def _active_job_processes(job: int) -> int:
    info = _BasicAccounting()
    size = wintypes.DWORD()
    if not _kernel.QueryInformationJobObject(job, 1, ctypes.byref(info),
                                              ctypes.sizeof(info), ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    return info.ActiveProcesses


def _await_empty_job(job: int, *, seconds: float) -> bool:
    until = time.monotonic() + seconds
    while True:
        if _active_job_processes(job) == 0:
            return True
        if time.monotonic() >= until:
            return False
        time.sleep(0.02)


def supervise(
    payload: dict[str, object], *,
    deadline_at: datetime,
    active: Callable[[], bool],
    module: str = 'backend.app.runs.supervisor',
    poll_seconds: float = 0.25,
    termination_grace_seconds: float = 2.0,
    on_stop: Callable[[TerminationEvidence], None] | None = None,
) -> dict[str, object]:
    """Run one subprocess under platform supervision and return its JSON result.

    ``active`` checks the parent-owned claim and cancellation state.  The
    caller must check the claim again before publishing success.  If exit
    cannot be confirmed, this raises and the caller must quarantine its slot.
    """
    if deadline_at.tzinfo is None or deadline_at.utcoffset() is None:
        raise ValueError('deadline_at must be timezone aware')
    if poll_seconds <= 0 or termination_grace_seconds <= 0:
        raise ValueError('supervision intervals must be positive')
    if datetime.now(timezone.utc) >= deadline_at:
        raise SupervisionStopped(TerminationEvidence(
            'deadline', datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(), 0.0, None))
    if not active():
        raise SupervisionStopped(TerminationEvidence(
            'claim_inactive', datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(), 0.0, None))
    monotonic_deadline = time.monotonic() + (deadline_at - datetime.now(timezone.utc)).total_seconds()

    if sys.platform == 'linux':
        from .linux_supervisor import supervise_linux
        return supervise_linux(payload, deadline_at=deadline_at, active=active,
            module=module, poll_seconds=poll_seconds,
            termination_grace_seconds=termination_grace_seconds, on_stop=on_stop)

    job = _job_create()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            [sys.executable, '-m', module, '--child'],
            cwd=str(Path(__file__).resolve().parents[3]),
            env=_child_environment(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=None, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True,
        )
        # The child blocks on stdin until assignment succeeds.  No fit can
        # start in the unprotected spawn/assignment interval.
        if not _kernel.AssignProcessToJobObject(job, process._handle):
            raise ctypes.WinError(ctypes.get_last_error())
        # Recheck after spawn/assignment, before releasing the child handshake.
        stop_reason = ('deadline' if time.monotonic() >= monotonic_deadline or
                       datetime.now(timezone.utc) >= deadline_at else
                       'claim_inactive' if not active() else None)
        if stop_reason is None:
            assert process.stdin is not None
            process.stdin.write(json.dumps(payload, separators=(',', ':')) + '\n')
            process.stdin.close()
            while process.poll() is None:
                now = datetime.now(timezone.utc)
                if time.monotonic() >= monotonic_deadline or now >= deadline_at:
                    stop_reason = 'deadline'
                    break
                if not active():
                    stop_reason = 'claim_inactive'
                    break
                remaining = min(monotonic_deadline - time.monotonic(),
                                (deadline_at - now).total_seconds())
                time.sleep(min(poll_seconds, max(0.001, remaining)))
            if stop_reason is None:
                stop_reason = ('deadline' if time.monotonic() >= monotonic_deadline or
                               datetime.now(timezone.utc) >= deadline_at else
                               'claim_inactive' if not active() else None)
        if stop_reason is not None:
            requested_at = datetime.now(timezone.utc)
            start = time.monotonic()
            if not _kernel.TerminateJobObject(job, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                process.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired as exc:
                evidence = TerminationEvidence(stop_reason, requested_at.isoformat(),
                                                None, None, None)
                if on_stop is not None:
                    on_stop(evidence)
                raise SupervisionUncertain(evidence) from exc
            if not _await_empty_job(job, seconds=termination_grace_seconds):
                evidence = TerminationEvidence(stop_reason, requested_at.isoformat(),
                                                None, None, process.returncode)
                if on_stop is not None:
                    on_stop(evidence)
                raise SupervisionUncertain(evidence)
            evidence = TerminationEvidence(stop_reason, requested_at.isoformat(),
                                            datetime.now(timezone.utc).isoformat(),
                                            time.monotonic() - start, process.returncode)
            if on_stop is not None:
                on_stop(evidence)
            raise SupervisionStopped(evidence)

        # A successful child result is still provisional until the caller's
        # claim/budget/deadline checks and repository success transaction.
        # Windows can signal the main process just before Job accounting drops
        # its active count. Allow a bounded accounting drain, never accept a
        # nonempty tree as a successful exit.
        drain_seconds = max(0.0, min(0.1, monotonic_deadline - time.monotonic()))
        if not _await_empty_job(job, seconds=drain_seconds):
            if not _kernel.TerminateJobObject(job, 1) or not _await_empty_job(
                    job, seconds=termination_grace_seconds):
                raise SupervisionUncertain(TerminationEvidence(
                    'child_tree_active', datetime.now(timezone.utc).isoformat(),
                    None, None, process.returncode))
            raise ChildExecutionError('ChildTreeStillActive',
                                      'training child left active descendants')
        assert process.stdout is not None
        output = process.stdout.read()
        if process.returncode != 0:
            raise ChildExecutionError('ChildProcessExit', f'training child exited {process.returncode}')
        return _decode_result(output)
    finally:
        # Kill-on-close also covers failures between Popen and normal exit.
        _kernel.CloseHandle(job)
        if process is not None:
            if process.poll() is None:
                try:
                    process.wait(timeout=termination_grace_seconds)
                except subprocess.TimeoutExpired as exc:
                    raise SupervisionUncertain(TerminationEvidence(
                        'cleanup', datetime.now(timezone.utc).isoformat(), None,
                        None, None)) from exc
            if process.stdout is not None:
                process.stdout.close()
            if process.stdin is not None:
                process.stdin.close()


def _decode_result(output: str) -> dict[str, object]:
    try:
        envelope = json.loads(output)
    except (TypeError, ValueError) as exc:
        raise ChildExecutionError('ChildProtocolError', 'invalid training child result') from exc
    if not isinstance(envelope, dict) or type(envelope.get('ok')) is not bool:
        raise ChildExecutionError('ChildProtocolError', 'invalid training child result')
    if not envelope['ok']:
        if envelope.get('error_type') == 'GuardError':
            from .guard import GuardError
            details = envelope.get('guard', {})
            # The child carries only registered fields, never exception prose.
            from pydantic import TypeAdapter
            from .guard import Stage
            stage = TypeAdapter(Stage).validate_python(details.get('stage'))
            ref = details.get('report_id')
            import re
            if ref is not None and (not isinstance(ref, str) or not re.fullmatch('guard-[a-f0-9]{64}', ref)):
                raise ChildExecutionError('ChildProtocolError', 'invalid guard reference')
            raise GuardError(details.get('code'), stage=stage, report_id=ref)
        raise ChildExecutionError(str(envelope.get('error_type', 'ChildExecutionError')),
                                  str(envelope.get('message', 'training child failed')))
    result = envelope.get('result')
    if not isinstance(result, dict):
        raise ChildExecutionError('ChildProtocolError', 'invalid training child result')
    return result


def _child_main() -> None:
    line = sys.stdin.readline()
    if not line:
        return
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        payload = json.loads(line)
        if payload.get('probe') == 'block':
            time.sleep(30)
            result: dict[str, object] = {'probe': 'finished'}
        elif payload.get('probe') == 'tree':
            grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                          stdin=subprocess.DEVNULL,
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL,
                                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                                          start_new_session=sys.platform == 'linux')
            Path(str(payload['identity_path'])).write_text(json.dumps({
                'child': os.getpid(), 'grandchild': grandchild.pid}))
            time.sleep(30)
            result = {'probe': 'finished'}
        elif payload.get('probe') == 'return':
            result = {'value': payload.get('value')}
        elif payload.get('kind') == 'training-run-v1':
            from .execution import execute_claimed_run
            from .repository import RunRepository

            run_id = payload.get('run_id')
            claim_token = payload.get('claim_token')
            database_path = payload.get('runs_database')
            if not all(isinstance(value, str) and value for value in
                       (run_id, claim_token, database_path)):
                raise ValueError('invalid supervised Run identity')
            repository = RunRepository(Path(database_path))
            record = repository.get(run_id)
            if record.claim_token != claim_token:
                raise ValueError('supervised Run claim changed')
            def check_claim() -> None:
                repository.assert_active(run_id, claim_token=claim_token,
                                         now=datetime.now(timezone.utc))

            check_claim()
            training_budget = None
            binding = payload.get('training_budget')
            if binding is not None:
                if not isinstance(binding, dict):
                    raise ValueError('invalid supervised budget binding')
                from ..agent.training_budget import TrainingBudget

                training_budget = TrainingBudget(
                    Path(str(binding['agent_database'])),
                    task_id=str(binding['task_id']),
                    policy_digest=str(binding['policy_digest']),
                    reservation_id=str(binding['reservation_id']),
                    run_id=run_id, claim_token=claim_token,
                    claim_check=check_claim)
                training_budget.bind()
            result = execute_claimed_run(record, repository=repository,
                                         storage_root=repository.database_path.parent,
                                         training_budget=training_budget)
        else:
            raise ValueError('unknown supervised child request')
        envelope = {'ok': True, 'result': result}
    except Exception as exc:
        envelope = {'ok': False, 'error_type': type(exc).__name__, 'message': str(exc)}
        from .guard import GuardError
        if isinstance(exc, GuardError):
            envelope = dict(ok=False, error_type='GuardError', guard=dict(
                code=exc.code, stage=exc.stage, report_id=exc.report_id))
    original_stdout.write(json.dumps(envelope, separators=(',', ':')))
    original_stdout.flush()


if __name__ == '__main__' and sys.argv[1:] == ['--child']:
    _child_main()
