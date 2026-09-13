"""Upgrade old schemas with real concurrent connections and processes."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import sqlite3
import threading

import pytest

from backend.app.agent.repository import AgentSessionRepository
from backend.app.runs.contracts import Principal


def initialize_worker(path, barrier):
    barrier.wait(timeout=30)
    AgentSessionRepository(path).initialize()


@pytest.mark.parametrize('missing', [(), ('dataset_sha256',), ('metadata_version',),
                                     ('dataset_sha256', 'metadata_version')])
def test_concurrent_old_database_upgrade_preserves_history(tmp_path, missing):
    path = tmp_path / 'old.sqlite'
    repo = AgentSessionRepository(path)
    repo.initialize()
    repo.create_session(dataset_id='historical-data', selection_metric='macro_f1',
        allowed_models=['svm'], max_runs=1, seed=42, evaluation_config={}, modules=[],
        context_policy={}, client_request_id='original', payload_hash='original-hash',
        principal=Principal())
    with sqlite3.connect(path) as db:
        original = db.execute('SELECT * FROM agent_sessions_v1').fetchall()
        original_columns = [row[1] for row in db.execute('PRAGMA table_info(agent_sessions_v1)')]
        for column in missing:
            db.execute(f'ALTER TABLE agent_sessions_v1 DROP COLUMN {column}')
    barrier = threading.Barrier(16)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(initialize_worker, path, barrier) for _ in range(16)]
        for future in futures:
            future.result(timeout=40)
    # Recheck values by name; a partial migration can change column order.
    with sqlite3.connect(path) as db:
        columns = [row[1] for row in db.execute('PRAGMA table_info(agent_sessions_v1)')]
        assert columns.count('dataset_sha256') == columns.count('metadata_version') == 1
        assert db.execute('SELECT dataset_id,payload_hash,dataset_sha256,metadata_version '
                          'FROM agent_sessions_v1').fetchall() == [('historical-data','original-hash',None,None)]
        assert len(db.execute('SELECT * FROM agent_sessions_v1').fetchall()) == len(original) == 1
        row = db.execute('SELECT * FROM agent_sessions_v1').fetchone()
        assert dict(zip(columns,row)) == dict(zip(original_columns,original[0]))


def test_old_database_upgrade_across_processes(tmp_path):
    path = tmp_path / 'old.sqlite'
    AgentSessionRepository(path).initialize()
    with sqlite3.connect(path) as db:
        for column in ('dataset_sha256', 'metadata_version'):
            db.execute(f'ALTER TABLE agent_sessions_v1 DROP COLUMN {column}')
    ctx = multiprocessing.get_context('spawn')
    barrier = ctx.Barrier(4)
    processes = [ctx.Process(target=initialize_worker, args=(path, barrier)) for _ in range(4)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def test_concurrent_empty_database_initialize(tmp_path):
    barrier = threading.Barrier(8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(initialize_worker, tmp_path/'empty.sqlite', barrier) for _ in range(8)]
        for future in futures:
            future.result(timeout=40)
