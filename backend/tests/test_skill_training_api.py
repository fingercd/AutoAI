"""Strict preview, transactional submission identity and compatibility."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunRepository, SubmissionConflict


@pytest.fixture
def api(tmp_path, monkeypatch):
    from backend.app.routers import datasets, deps, runs, batches, preflight
    from backend.app import training_preflight
    root = tmp_path / 'storage'
    (root / 'uploads').mkdir(parents=True)
    repository = RunRepository(root / 'runs.sqlite3')
    repository.initialize()
    for module in (datasets, deps, runs, training_preflight):
        monkeypatch.setattr(module, 'DATASETS_DATABASE', root / 'datasets.sqlite3')
        monkeypatch.setattr(module, 'STORAGE_DIR', root)
    monkeypatch.setattr(datasets, 'UPLOADS_DIR', root / 'uploads')
    for module in (runs, batches, preflight):
        monkeypatch.setattr(module, 'get_run_repository', lambda: repository)
    for module in (runs, batches):
        monkeypatch.setattr(module, 'get_run_dir', lambda run_id: root / 'runs' / run_id)
    return TestClient(app), repository, root


def upload(client, *, prefix='main', groups=12, shift=0, offset=0):
    rng = np.random.default_rng(42)
    rows = [[i + 1, str(i // groups), f'{prefix}-{i:03d}', f'{i}.csv', *values]
            for i, values in enumerate(rng.normal(size=(groups * 2, 20)))]
    frame = pd.DataFrame(rows, columns=['Index', 'Label', 'Sample_ID', 'Name', *[str(i + shift) for i in range(20)]])
    reply = client.post('/api/datasets/upload', files={'file': ('data.csv', frame.to_csv(index=False).encode(), 'text/csv')})
    assert reply.status_code == 200, reply.text
    return reply.json()['dataset_id']


def config(**extra):
    return {'experiment_version': 'word-0904', 'model_type': 'logistic_regression', **extra}


def test_preview_no_queue_or_artifacts_and_strict_submission(api):
    client, repository, root = api
    data_id = upload(client)
    preview = client.post('/api/training/preflight', json={'dataset_id': data_id, 'config': config()}).json()
    assert preview['runnable'], preview
    assert preview['run_count'] == 1
    assert preview['data']['primary']['sample_count'] == 24
    assert preview['normalized_configs'][0]['training_profile'] == 'quick'
    assert repository.list_scoped(principal=Principal()) == []
    assert not (root / 'runs').exists()
    created = client.post('/api/training/runs', json=preview['submit_payload'], headers={'Idempotency-Key': 'once'})
    assert created.status_code == 202, created.text
    first = created.json()['run_id']
    claimed = repository.claim_next(worker_id='fixture', now=datetime.now(timezone.utc))
    assert claimed.run_id == first
    replay = client.post('/api/training/runs', json=preview['submit_payload'], headers={'Idempotency-Key': 'once'})
    assert replay.json()['run_id'] == first
    assert replay.json()['state'] == 'running'
    altered = {**preview['submit_payload'], 'config': {**preview['submit_payload']['config'], 'seed': 73}}
    assert client.post('/api/training/runs', json=altered, headers={'Idempotency-Key': 'once'}).status_code == 409


@pytest.mark.parametrize('mode,external', [('stratified_holdout', False), ('leave_one_sample_id_cv', False),
                                          ('external_test_holdout', True), ('leave_one_sample_id_cv_with_external_test', True)])
def test_all_four_modes_have_group_disjoint_partitions(api, mode, external):
    client, _, _ = api
    body = {'dataset_id': upload(client), 'config': config(split_mode=mode)}
    if external:
        body['test_dataset_id'] = upload(client, prefix='ext')
    preview = client.post('/api/training/preflight', json=body).json()
    assert preview['runnable'], preview
    assert preview['data']['evaluation_strategy'] == mode
    for fold in preview['data']['folds']:
        partitions = fold['partitions']
        train, valid, test = [set(partitions[key]['sample_ids']) for key in ('train', 'valid', 'test')]
        assert not train & valid and not train & test and not valid & test


def test_invalid_config_data_and_full_search_are_blocked(api):
    client, repository, _ = api
    data_id = upload(client, groups=3)
    for extra in ({'learning_rte': 0.1}, {'logistic_c': 3}, {'split_mode': 'typo'}, {'split_train': 8.5}):
        result = client.post('/api/training/preflight', json={'dataset_id': data_id, 'config': config(**extra)}).json()
        assert not result['runnable'] and result['errors']
    assert client.post('/api/training/preflight', json={'dataset_id': data_id, 'config': config(training_profile='full')}).json()['runnable'] is False
    assert client.post('/api/training/preflight', json={'dataset_id': data_id, 'config': config()}).json()['runnable'] is True
    # Legacy unknown-field handling remains a warning instead of a strict rejection.
    reply = client.post('/api/training/runs', json={'dataset_id': data_id, 'config': {'model_type': 'logistic_regression', 'typo': 1}})
    assert reply.status_code == 202 and reply.json()['warnings']
    assert len(repository.list_scoped(principal=Principal())) == 1


def test_batch_roundtrip_seed_and_replay(api):
    client, repository, _ = api
    body = {'task_type': 'batch', 'dataset_id': upload(client), 'model_types': ['logistic_regression', 'svm'],
            'base_seed': 73, 'config': {'experiment_version': 'word-0904'}}
    preview = client.post('/api/training/preflight', json=body).json()
    assert preview['runnable'], preview
    for _ in range(2):
        response = client.post('/api/training/batches', json=preview['submit_payload'], headers={'Idempotency-Key': 'batch-once'})
        assert response.status_code == 202, response.text
    records = repository.list_batch_runs_scoped(response.json()['batch_id'], principal=Principal())
    assert len(records) == 2 and {record.config['split_seed'] for record in records} == {73}


def test_repository_concurrency_scope_and_deleted_tombstone(tmp_path):
    repository = RunRepository(tmp_path / 'runs.sqlite3')
    repository.initialize()
    def create(principal=Principal()):
        return repository.create_queued(dataset_id='d', config={}, principal=principal,
                                        request_key='key', request_fingerprint='fingerprint')
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(lambda _: create(), range(12)))
    assert len({record.run_id for record in records}) == 1
    assert create(Principal(owner_id='other')).run_id != records[0].run_id
    repository.cancel_scoped(records[0].run_id, principal=Principal(), now=datetime.now(timezone.utc))
    repository.delete_terminal_scoped(records[0].run_id, principal=Principal())
    with pytest.raises(SubmissionConflict, match='删除'):
        create()
