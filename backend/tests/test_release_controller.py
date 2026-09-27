"""Linux fault injection uses only temporary databases and test-owned processes."""
from contextlib import closing
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import time

import pytest

from scripts import service_control as control

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux /proc controller')


@pytest.fixture
def controller(tmp_path):
    storage = tmp_path/'storage'; storage.mkdir()
    for name, sql in {
        'runs.sqlite3': 'CREATE TABLE runs(run_id TEXT,state TEXT);',
        'datasets.sqlite3': 'CREATE TABLE datasets(dataset_id TEXT);',
        'agent.sqlite3': '''CREATE TABLE agent_experiment_reservations_v1
            (reservation_id TEXT,session_id TEXT,run_id TEXT,state TEXT,owner_id TEXT,tenant_id TEXT);
            CREATE TABLE agent_sessions_v1(session_id TEXT,owner_id TEXT,tenant_id TEXT,budget_task_id TEXT);
            CREATE TABLE task_budget_reservations_v1(task_id TEXT,dimension TEXT,status TEXT,held INTEGER);
            CREATE TABLE task_training_executions_v1(reservation_id TEXT,state TEXT);''',
    }.items():
        with closing(sqlite3.connect(storage/name)) as db:
            db.execute('PRAGMA journal_mode=WAL');db.executescript(sql);db.commit()
    return control.Controller(control.Config(tmp_path,sys.executable,stop_timeout=.25),lambda release: {})


def execute(controller, name, sql):
    with closing(controller.connect(name)) as db:
        db.executescript(sql);db.commit()


def test_identity_exit_permission_and_reuse(monkeypatch):
    monkeypatch.setattr(control.os,'readlink',lambda _: (_ for _ in ()).throw(PermissionError()))
    stats=iter([('123','R'),('123','R'),None])
    monkeypatch.setattr(control,'_stat',lambda _:next(stats))
    assert control.identity(os.getpid()) is None
    monkeypatch.setattr(control,'_stat',lambda _:('123','R'))
    with pytest.raises(PermissionError):control.identity(os.getpid())
    stats=iter([('123','R'),('123','R'),('456','R')])
    monkeypatch.setattr(control,'_stat',lambda _:next(stats))
    with pytest.raises(control.ControlError,match='pid_reused'):control.identity(os.getpid())


def test_identity_mismatch_never_signals():
    expected=control.identity(os.getpid());expected['start_ticks']='incorrect'
    with pytest.raises(control.ControlError,match='identity_mismatch'):
        control.send_owned(expected,signal.SIGTERM)


def test_signals_owned_child_and_natural_exit():
    p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
    try:
        expected=control.identity(p.pid)
        control.send_owned(expected,signal.SIGTERM)
        p.wait(timeout=3)
        assert control.owned(expected) is None
        control.send_owned(expected,signal.SIGTERM)
    finally:
        if p.poll() is None:p.terminate();p.wait(timeout=3)


@pytest.mark.parametrize('name,sql',[
 ('runs.sqlite3',"INSERT INTO runs VALUES('r','queued')"),
 ('runs.sqlite3',"INSERT INTO runs VALUES('r','running')"),
 ('agent.sqlite3',"INSERT INTO task_budget_reservations_v1 VALUES('t','fits','unknown_pending',2)"),
 ('agent.sqlite3',"INSERT INTO task_budget_reservations_v1 VALUES('t','fits','settled',1)"),
 ('agent.sqlite3',"INSERT INTO task_training_executions_v1 VALUES('r','active')"),
 ('agent.sqlite3',"INSERT INTO task_training_executions_v1 VALUES('r','unknown_pending')"),
 ('agent.sqlite3',"INSERT INTO agent_experiment_reservations_v1 VALUES('r','s','missing','bound',NULL,NULL)"),
])
def test_live_business_refused(controller,name,sql):
    execute(controller,name,sql)
    with pytest.raises(control.ControlError,match='not_drained'):controller.require_idle(controller.activity())


def test_exact_legacy_only_and_open_empty_session(controller):
    execute(controller,'agent.sqlite3',"INSERT INTO agent_sessions_v1 VALUES('s',NULL,NULL,NULL); INSERT INTO agent_experiment_reservations_v1 VALUES('r','s','old','bound',NULL,NULL)")
    config=control.Config(controller.config.base,sys.executable,legacy_bindings=(('r','s','old'),))
    c=control.Controller(config,lambda release:{})
    c.require_idle(c.activity())
    execute(c,'agent.sqlite3',"UPDATE agent_experiment_reservations_v1 SET owner_id='unexpected'")
    with pytest.raises(control.ControlError):c.require_idle(c.activity())


def test_gate_blocks_writes_then_releases_on_exception(controller):
    with pytest.raises(RuntimeError,match='injected'):
        with controller.admission_gate():
            for name in ('runs.sqlite3','datasets.sqlite3','agent.sqlite3'):
                with closing(controller.connect(name)) as db:
                    with pytest.raises(sqlite3.OperationalError,match='locked'):db.execute('BEGIN IMMEDIATE')
            raise RuntimeError('injected')
    for name in ('runs.sqlite3','datasets.sqlite3','agent.sqlite3'):
        with closing(controller.connect(name)) as db:db.execute('BEGIN IMMEDIATE');db.rollback()


def test_operation_mutex_and_unregistered_idempotence(controller):
    with controller.operation():
        with pytest.raises(control.ControlError,match='another_control'):controller.stop()
    assert controller.stop()['state']=='unregistered'
    assert controller.stop()['state']=='unregistered'


def test_unknown_listener_is_not_stopped(controller):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));listener.listen()
        c=control.Controller(control.Config(controller.config.base,sys.executable,port=listener.getsockname()[1]),lambda release:{})
        with pytest.raises(control.ControlError,match='unknown_listener'):c.wait_port()
        assert listener.fileno()>=0


def test_graceful_timeout_releases_gate_keeps_identity(controller,tmp_path):
    ready=tmp_path/'ready'
    p=subprocess.Popen([sys.executable,'-c',
        "import signal,time,pathlib;signal.signal(signal.SIGINT,signal.SIG_IGN);pathlib.Path(%r).touch();time.sleep(20)"%str(ready)])
    try:
        deadline=time.monotonic()+3
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert ready.exists()
        expected=control.identity(p.pid)
        control.save(controller.record_path,dict(identity=expected,workers=[]))
        with pytest.raises(control.ControlError,match='incomplete_gate_released'):controller.stop()
        assert controller.status()['state']=='running'
        assert controller.record()['identity']==expected
        with controller.admission_gate():pass
    finally:
        p.terminate();p.wait(timeout=3)
    assert controller.stop()['state']=='stopped'


def test_orphan_worker_is_stopped_and_confirmed(controller):
    gone=subprocess.Popen([sys.executable,'-c','pass']);expected=control.identity(gone.pid);gone.wait(timeout=3)
    worker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
    try:
        control.save(controller.record_path,dict(identity=expected,workers=[control.identity(worker.pid)]))
        assert controller.status()['state']=='orphaned_worker'
        assert controller.stop()['state']=='stopped'
        worker.wait(timeout=3)
        assert controller.stop()['state']=='stopped'
    finally:
        if worker.poll() is None:worker.terminate();worker.wait(timeout=3)
