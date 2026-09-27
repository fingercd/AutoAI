"""Linux release controller. Site adapters supply paths and private environment.

No credentials or machine-specific paths are persisted by this module. The same
identity, admission gate and exit proof are used for stop, restart and rollback.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import time
from typing import Callable
import urllib.request

VERSION = 'release-controller-v1'


class ControlError(RuntimeError):
    """A bounded, actionable refusal; never a reason to force-kill a process."""


def _stat(pid: int):
    try:
        parts = (Path('/proc')/str(pid)/'stat').read_text().split(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None if parts[0] == 'Z' else (parts[19], parts[0])


def _identity_once(pid: int) -> dict | None:
    """A missing/zombie stat proves exit; permission errors alone never do."""
    first = _stat(pid)
    if first is None:
        return None
    proc = Path('/proc')/str(pid)
    try:
        uid = proc.stat().st_uid
        cwd = os.readlink(proc/'cwd')
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        after = _stat(pid)
        if after is None:
            return None
        if after[0] != first[0]:
            raise ControlError('pid_reused_during_identity') from None
        raise  # A live, unchanged process with denied cwd access is not stopped.
    after = _stat(pid)
    if after is None:
        return None
    if after[0] != first[0]:
        raise ControlError('pid_reused_during_identity')
    return dict(pid=pid, start_ticks=first[0], cwd=cwd, uid=uid)


def identity(pid: int) -> dict | None:
    """Re-observe the short procfs teardown window; persistent denial is an error."""
    deadline = time.monotonic()+.15
    first = _stat(pid)
    while True:
        try:
            result = _identity_once(pid)
            if result is not None and first is not None and result['start_ticks'] != first[0]:
                raise ControlError('pid_reused_during_identity')
            return result
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.005)


def owned(expected: dict) -> dict | None:
    actual = identity(expected['pid'])
    if actual is not None and actual != expected:
        raise ControlError('process_identity_mismatch')
    if expected['uid'] != os.getuid():
        raise ControlError('foreign_process_owner')
    return actual


def send_owned(expected: dict, sig: int) -> None:
    if owned(expected) is None:
        return
    # The deployment host does not expose pidfd_open. Recheck all four identity
    # fields immediately before signaling, as the existing supervisor does.
    if owned(expected) is not None:
        try:
            os.kill(expected['pid'], sig)
        except ProcessLookupError:
            if _stat(expected['pid']) is not None:
                raise


def save(path: Path, value: dict) -> None:
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


@dataclass(frozen=True)
class Config:
    base: Path
    python: str
    port: int = 18085
    stop_timeout: float = 8
    start_timeout: float = 40
    port_timeout: float = 15
    # Exact, previously audited tuples, never a blanket NULL-scope exception.
    legacy_bindings: tuple[tuple[str, str, str], ...] = ()

    @property
    def storage(self): return self.base/'storage'
    @property
    def runtime(self): return self.base/'runtime'
    @property
    def current(self): return self.base/'current'


class Controller:
    def __init__(self, config: Config, environment: Callable[[Path], dict]):
        if os.name != 'posix' or not Path('/proc/self/stat').exists():
            raise ControlError('linux_proc_required')
        self.config = config
        self.environment = environment
        config.runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.record_path = config.runtime/'service.json'

    @contextmanager
    def operation(self):
        import fcntl
        with (self.config.runtime/'control.lock').open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ControlError('another_control_operation') from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def record(self):
        return json.loads(self.record_path.read_text()) if self.record_path.exists() else None

    def children(self, web):
        if owned(web) is None:
            return []
        try:
            pids = (Path('/proc')/str(web['pid'])/'task'/str(web['pid'])/'children').read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            if owned(web) is None:
                return []
            raise
        result = []
        for pid in pids:
            child = identity(int(pid))
            if child is not None:
                if child['uid'] != web['uid'] or child['cwd'] != web['cwd']:
                    raise ControlError('unexpected_service_child')
                result.append(child)
        owned(web)
        return result

    def status(self):
        record = self.record()
        if record is None:
            return dict(controller=VERSION, state='unregistered')
        web = owned(record['identity'])
        workers = [w for w in record.get('workers', []) if owned(w) is not None]
        if web is not None:
            workers = self.children(web)
        return dict(controller=VERSION, state='running' if web else 'orphaned_worker' if workers else 'stopped',
                    record=record, web=web, workers=workers)

    def connect(self, name):
        return sqlite3.connect('file:'+str(self.config.storage/name)+'?mode=rw', uri=True, timeout=.5)

    def activity(self):
        with closing(self.connect('runs.sqlite3')) as db:
            states = dict(db.execute('SELECT run_id,state FROM runs'))
        with closing(self.connect('agent.sqlite3')) as db:
            rows = db.execute('''SELECT r.reservation_id,r.session_id,r.run_id,r.state,
                r.owner_id,r.tenant_id,s.owner_id,s.tenant_id,s.budget_task_id
                FROM agent_experiment_reservations_v1 r
                LEFT JOIN agent_sessions_v1 s ON r.session_id=s.session_id''').fetchall()
            held = db.execute("SELECT task_id,dimension,status FROM task_budget_reservations_v1 WHERE status IN ('reserved','dispatched','unknown_pending') OR held>0").fetchall()
            executions = db.execute("SELECT reservation_id,state FROM task_training_executions_v1 WHERE state!='settled'").fetchall()
            session_ids = {r[0] for r in db.execute('SELECT session_id FROM agent_sessions_v1')}
        active = [rid for rid, state in states.items() if state not in ('succeeded','failed','cancelled')]
        pending = []
        preserved = []
        for row in rows:
            legacy = (tuple(row[:3]) in self.config.legacy_bindings and row[3]=='bound'
                      and row[2] not in states and row[1] in session_ids and all(v is None for v in row[4:]))
            if legacy:
                preserved.append(row[0])
            elif (row[1] not in session_ids or row[3] in ('reserved','compensation_required')
                  or row[3]=='bound' and states.get(row[2]) not in ('succeeded','failed','cancelled')):
                pending.append(row[0])
        return dict(active_runs=active, pending_reservations=pending, held_budget=held,
                    unsettled_executions=executions, preserved_legacy_bindings=preserved)

    @staticmethod
    def require_idle(activity):
        if any(activity[k] for k in ('active_runs','pending_reservations','held_budget','unsettled_executions')):
            raise ControlError('business_not_drained')

    @contextmanager
    def admission_gate(self):
        locks = []
        try:
            for name in ('agent.sqlite3','datasets.sqlite3','runs.sqlite3'):
                db = self.connect(name)
                locks.append(db)
                db.execute('BEGIN IMMEDIATE')
            self.require_idle(self.activity())
            yield
        finally:
            for db in reversed(locks):
                db.rollback()
                db.close()

    def stop(self):
        with self.operation():
            state = self.status()
            if state['state'] in ('unregistered','stopped'):
                return state
            self.require_idle(self.activity())
            record = state['record']
            # Capture the supervisor's current child, including legitimate restarts.
            record.update(workers=state['workers'], controller_version=VERSION, phase='stopping')
            save(self.record_path, record)
            began = time.monotonic()
            with self.admission_gate():
                if state['web']:
                    send_owned(record['identity'], signal.SIGINT)
                else:
                    for child in record['workers']:
                        send_owned(child, signal.SIGTERM)
                while time.monotonic()-began < self.config.stop_timeout:
                    web = owned(record['identity'])
                    if web:
                        for child in self.children(web):
                            if child not in record['workers']:
                                record['workers'].append(child)
                                save(self.record_path, record)
                    children = [c for c in record['workers'] if owned(c) is not None]
                    if web is None and not children:
                        break
                    time.sleep(.05)
                else:
                    raise ControlError('graceful_stop_incomplete_gate_released')
            self.require_idle(self.activity())
            record.update(phase='stopped', stopped_at=time.time(), stop_seconds=time.monotonic()-began)
            save(self.record_path, record)
            return self.status()

    def wait_port(self):
        deadline = time.monotonic()+self.config.port_timeout
        while True:
            with socket.socket() as probe:
                probe.settimeout(.2)
                if probe.connect_ex(('127.0.0.1',self.config.port)) == 0:
                    raise ControlError('port_has_unknown_listener')
            with socket.socket() as probe:
                # Match uvicorn's reuse semantics: TIME_WAIT is not a live service.
                probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                try:
                    probe.bind(('127.0.0.1',self.config.port))
                    return
                except OSError as exc:
                    if exc.errno != errno.EADDRINUSE:
                        raise
            if time.monotonic() >= deadline:
                raise ControlError('port_unavailable_without_listener')
            time.sleep(.1)

    def release_info(self, release):
        def git(*args): return subprocess.check_output(['git',*args],cwd=release,text=True).strip()
        if git('status','--porcelain'):
            raise ControlError('release_not_clean')
        if (release/'storage').resolve() != self.config.storage.resolve():
            raise ControlError('storage_binding_mismatch')
        return dict(source_head=git('rev-parse','HEAD'),source_tree=git('rev-parse','HEAD^{tree}'))

    def health(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{self.config.port}/health',timeout=1) as response:
            body = json.load(response)
        worker = body.get('worker', {})
        return (body.get('deployment_mode') == 'server' and worker.get('available')
                and worker.get('compatible') and worker.get('live_count') == 1
                and worker.get('contract_version') == 'training-worker-guard-v1')

    def start(self):
        with self.operation():
            state = self.status()
            release = self.config.current.resolve()
            if state['state'] == 'running':
                if state['web']['cwd'] != str(release):
                    raise ControlError('current_differs_from_running_release')
                if not self.health():
                    raise ControlError('existing_service_not_ready')
                return state
            if state['state'] == 'orphaned_worker':
                raise ControlError('owned_worker_still_running')
            info = self.release_info(release)
            self.wait_port()
            env = self.environment(release)
            if (env.get('AUTOAI_DEPLOYMENT_MODE') != 'server'
                    or len(env.get('AUTOAI_API_TOKEN','')) < 32
                    or not env.get('AUTOAI_ALLOWED_ORIGINS') or '*' in env['AUTOAI_ALLOWED_ORIGINS']):
                raise ControlError('invalid_server_environment')
            with (self.config.runtime/'service.log').open('a') as log:
                process = subprocess.Popen([self.config.python,'run.py','--server','--host','127.0.0.1',
                    '--port',str(self.config.port),'--no-browser'],cwd=release,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            del env
            expected = identity(process.pid)
            if expected is None:
                raise ControlError('server_exited_during_spawn')
            record = dict(identity=expected, workers=[], **info, controller_version=VERSION,
                          phase='starting', started_at=time.time(), port=self.config.port,
                          storage=str(self.config.storage))
            save(self.record_path,record)
            deadline = time.monotonic()+self.config.start_timeout
            while time.monotonic() < deadline:
                if owned(expected) is None:
                    raise ControlError('server_exited_before_ready')
                children = self.children(expected)
                if children != record['workers']:
                    record['workers']=children
                    save(self.record_path,record)
                try:
                    ready = len(children)==1 and self.health()
                except (OSError,ValueError):
                    ready=False
                if ready:
                    record['phase']='running'
                    save(self.record_path,record)
                    return self.status()
                time.sleep(.2)
            # Keep identities for diagnosis/explicit safe stop. Do not erase PIDs.
            raise ControlError('readiness_timeout_use_status_or_stop')

    def run(self, action):
        if action == 'status': return self.status()
        if action == 'start': return self.start()
        if action == 'stop': return self.stop()
        if action == 'restart':
            self.stop()
            return self.start()
        raise ControlError('unknown_action')
