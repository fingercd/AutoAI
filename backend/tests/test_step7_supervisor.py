"""Real Windows process supervision; no training data or provider is needed."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from backend.app.runs.supervisor import SupervisionStopped, _child_environment, supervise
from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker, execute_with_budget_supervision
from backend.app.agent.budget import (BudgetPolicy, DIMENSIONS, dimension_summary,
                                      freeze_policy)
from backend.tests.test_agent_model_sessions import api
from backend.tests.test_finite_search import search_session_request, HEADERS


pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows Job Object contract')


def _live(pid: int) -> bool:
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 0x00000102
    finally:
        kernel.CloseHandle(handle)


def _deadline(seconds: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def test_child_environment_omits_provider_credentials(monkeypatch):
    monkeypatch.setenv('AUTOAI_API_TOKEN', 'test-only-value')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only-value')
    child = _child_environment()
    assert 'AUTOAI_API_TOKEN' not in child
    assert 'OPENAI_API_KEY' not in child


def test_supervised_child_returns_structured_result():
    assert supervise({'probe': 'return', 'value': 7}, deadline_at=_deadline(8),
                     active=lambda: True)['value'] == 7


def test_supervisor_kills_blocked_child_at_deadline():
    evidence = []
    with pytest.raises(SupervisionStopped) as caught:
        supervise({'probe': 'block'}, deadline_at=_deadline(0.5),
                  active=lambda: True, poll_seconds=0.05, on_stop=evidence.append)
    assert caught.value.evidence == evidence[0]
    assert evidence[0].reason == 'deadline'
    assert evidence[0].exit_code is not None
    assert evidence[0].latency_seconds is not None
    assert evidence[0].latency_seconds < 2


def test_claim_loss_kills_child_and_grandchild(tmp_path):
    identity = tmp_path / 'identity.json'
    evidence = []

    def active() -> bool:
        return not identity.exists()

    with pytest.raises(SupervisionStopped) as caught:
        supervise({'probe': 'tree', 'identity_path': str(identity)},
                  deadline_at=_deadline(8), active=active,
                  poll_seconds=0.05, on_stop=evidence.append)
    assert caught.value.evidence.reason == 'claim_inactive'
    ids = json.loads(identity.read_text())
    until = time.monotonic() + 2
    while (_live(ids['child']) or _live(ids['grandchild'])) and time.monotonic() < until:
        time.sleep(0.02)
    assert not _live(ids['child'])
    assert not _live(ids['grandchild'])


def test_parent_crash_closes_job_and_kills_process_tree(tmp_path):
    identity = tmp_path / 'identity.json'
    code = (
        'from backend.app.runs.supervisor import supervise; '
        'from datetime import datetime,timedelta,timezone; '
        'import sys; '
        "supervise({'probe':'tree','identity_path':sys.argv[1]}, "
        'deadline_at=datetime.now(timezone.utc)+timedelta(seconds=20), '
        'active=lambda:True,poll_seconds=0.05)'
    )
    parent = subprocess.Popen([sys.executable, '-c', code, str(identity)],
                              cwd=str(Path(__file__).resolve().parents[2]),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        until = time.monotonic() + 8
        while not identity.exists() and time.monotonic() < until:
            assert parent.poll() is None
            time.sleep(0.05)
        assert identity.exists()
        ids = json.loads(identity.read_text())
        assert _live(ids['child']) and _live(ids['grandchild'])
        parent.kill()
        parent.wait(timeout=5)
        until = time.monotonic() + 5
        while (_live(ids['child']) or _live(ids['grandchild'])) and time.monotonic() < until:
            time.sleep(0.05)
        assert not _live(ids['child'])
        assert not _live(ids['grandchild'])
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)


@pytest.mark.parametrize('model,params,fit_limit,epoch_limit', [
    ('logistic_regression', {}, 2, 0),
    ('cnn1d', {'epochs': 1, 'batch_size': 8}, 1, 1),
])
def test_supervised_child_uses_existing_training_execution(
        api, model, params, fit_limit, epoch_limit):
    client, storage, dataset_id = api
    created = client.post('/api/agent/v2/sessions', headers=HEADERS,
                          json=search_session_request(dataset_id, [model],
                              **({'model_configs': {model: params}} if params else {})))
    assert created.status_code == 201, created.text
    locked = created.json()['locked_config']
    recipe = locked['preparation']['catalog']['recipes'][0]
    agent_database = storage / 'agent.sqlite3'
    now = time.time()
    limits = {name: 100 for name in DIMENSIONS}
    limits.update(experiments=1, model_fits=fit_limit, training_epochs=epoch_limit,
                  llm_calls=6, api_calls=12, input_tokens=None,
                  cached_tokens=None, output_tokens=6144)
    policy = BudgetPolicy('supervised-fit-task', now - 1, now + 90,
                          now + 80, limits)
    with sqlite3.connect(agent_database) as connection:
        connection.execute('BEGIN IMMEDIATE')
        freeze_policy(connection, policy, owner='backend')
        connection.execute('''UPDATE agent_sessions_v1 SET
            budget_task_id=?,budget_policy_digest=? WHERE session_id=?''',
            (policy.task_id, policy.digest, created.json()['session_id']))
    submitted = client.post(
        f"/api/agent/v2/sessions/{created.json()['session_id']}/experiments",
        headers=HEADERS, json={
            'recipe_id': recipe['recipe_id'],
            'recipe_digest': recipe['recipe_digest'],
            'catalog_digest': locked['preparation']['catalog']['catalog_digest'],
            'knowledge_refs': [], 'client_request_id': 'supervised-real-run',
        })
    assert submitted.status_code == 202, submitted.text
    with sqlite3.connect(agent_database) as connection:
        reservation_id = connection.execute('''SELECT reservation_id FROM
            agent_experiment_reservations_v1 WHERE run_id=?''',
            (submitted.json()['run_id'],)).fetchone()[0]
        prepared = json.loads(connection.execute('''SELECT compiled_config_json FROM
            agent_experiment_reservations_v1 WHERE reservation_id=?''',
            (reservation_id,)).fetchone()[0])
        assert prepared['execution_budget_task_id'] == policy.task_id
        assert prepared['execution_budget_policy_digest'] == policy.digest
    repository = RunRepository(storage / 'runs.sqlite3')
    assert repository.get(submitted.json()['run_id']).config['execution_budget_task_id'] == policy.task_id
    worker = RunWorker(repository=repository, worker_id='supervisor-test',
        execute=lambda record: execute_with_budget_supervision(
            record, repository=repository, agent_database=agent_database),
        now=lambda: datetime.now(timezone.utc), heartbeat_seconds=1)
    assert worker.run_once()
    run_id = submitted.json()['run_id']
    assert repository.get(run_id).state == 'succeeded'
    assert (storage / 'runs' / run_id / 'manifest.json').is_file()
    assert (storage / 'runs' / run_id / 'search_summary.json').is_file()
    with sqlite3.connect(agent_database) as connection:
        actual = dimension_summary(connection, task_id=policy.task_id,
                                   owner='backend', dimension='model_fits')
        assert (actual['known_actual'], actual['held_reserved']) == (fit_limit, 0)
        epochs = dimension_summary(connection, task_id=policy.task_id,
                                   owner='backend', dimension='training_epochs')
        assert (epochs['known_actual'], epochs['held_reserved']) == (epoch_limit, 0)
        events = connection.execute('''SELECT kind,status FROM task_training_events_v1
            WHERE reservation_id=? ORDER BY event_id''', (reservation_id,)).fetchall()
        assert events == ([('final_refit', 'completed'), ('trial', 'completed')]
                          if model == 'logistic_regression' else
                          [('trial', 'completed'), ('trial', 'completed')])
    assert not (Path(__file__).resolve().parents[2] / 'storage' / 'runs' /
                run_id).exists()
