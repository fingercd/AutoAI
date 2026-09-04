from pathlib import Path
from datetime import datetime, timezone

import pytest

from backend.app.datasets.repository import DatasetRepository
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunRepository
from backend.app.runs.status_projection import project_status
from backend.app.runs.submission import (
    DatasetUnavailable, RunSubmissionError, RunSubmissionRequest, RunSubmissionService,
    WorkerContractMismatch,
)


def test_submission_creates_snapshot_and_queued_projection(tmp_path):
    storage = tmp_path / 'storage'; uploads = storage / 'uploads'; uploads.mkdir(parents=True)
    source = uploads / 'data.csv'; source.write_text('Index,Label,Sample_ID,0\n1,A,S1,1\n', encoding='utf-8')
    datasets = DatasetRepository(storage / 'datasets.sqlite3', storage_root=storage); datasets.initialize()
    dataset = datasets.register(source, original_name='teacher.csv', principal=Principal())
    runs = RunRepository(storage / 'runs.sqlite3'); runs.initialize()
    root = storage / 'runs'
    service = RunSubmissionService(
        run_repository=runs, dataset_repository=datasets,
        run_dir=lambda run_id: root / run_id, status_projector=project_status,
    )
    result = service.submit(RunSubmissionRequest(
        dataset_id=dataset.dataset_id, legacy_data_path=None, test_dataset_id=None,
        test_legacy_data_path=None, raw_config={'model_type': 'logistic_regression'},
        principal=Principal(), submission_source='agent',
    ))
    assert result.record.state == 'queued'
    assert result.dataset_snapshot['name'] == 'teacher.csv'
    assert len(result.dataset_snapshot['sha256']) == 64
    assert (root / result.record.run_id / 'status.json').is_file()
    assert result.effective_config['submission_source'] == 'agent'


def test_submission_hides_cross_principal_dataset_and_rejects_incompatible_worker(tmp_path):
    storage = tmp_path / 'storage'; uploads = storage / 'uploads'; uploads.mkdir(parents=True)
    source = uploads / 'data.csv'; source.write_text('Index,Label,Sample_ID,0\n1,A,S1,1\n')
    datasets = DatasetRepository(storage / 'datasets.sqlite3', storage_root=storage); datasets.initialize()
    dataset = datasets.register(source, original_name='data.csv', principal=Principal('a', 'tenant'))
    runs = RunRepository(storage / 'runs.sqlite3'); runs.initialize()
    service = RunSubmissionService(
        run_repository=runs, dataset_repository=datasets,
        run_dir=lambda run_id: storage / 'runs' / run_id, status_projector=project_status,
    )
    request = RunSubmissionRequest(
        dataset_id=dataset.dataset_id, legacy_data_path=None, test_dataset_id=None,
        test_legacy_data_path=None, raw_config={'model_type': 'logistic_regression'},
        principal=Principal('b', 'tenant'), submission_source='agent',
    )
    with pytest.raises(DatasetUnavailable): service.submit(request)

    runs.record_worker_heartbeat(
        worker_id='legacy', now=datetime.now(timezone.utc), contract_version=None,
    )
    with pytest.raises(WorkerContractMismatch):
        service.submit(RunSubmissionRequest(
            **{**request.__dict__, 'principal': Principal('a', 'tenant')}
        ))


def test_agent_submission_key_is_durable_and_payload_bound(tmp_path):
    storage = tmp_path / 'storage'; uploads = storage / 'uploads'; uploads.mkdir(parents=True)
    source = uploads / 'data.csv'; source.write_text('Index,Label,Sample_ID,0\n1,A,S1,1\n')
    datasets = DatasetRepository(storage / 'datasets.sqlite3', storage_root=storage); datasets.initialize()
    dataset = datasets.register(source, original_name='display-only.csv', principal=Principal())
    runs = RunRepository(storage / 'runs.sqlite3'); runs.initialize()
    service = RunSubmissionService(
        run_repository=runs, dataset_repository=datasets,
        run_dir=lambda run_id: storage / 'runs' / run_id, status_projector=project_status,
    )
    request = RunSubmissionRequest(
        dataset_id=dataset.dataset_id, legacy_data_path=None, test_dataset_id=None,
        test_legacy_data_path=None, raw_config={'model_type': 'logistic_regression'},
        principal=Principal(), submission_source='agent', submission_key='reservation-1',
    )

    first = service.submit(request)
    second = service.submit(request)

    assert first.record.run_id == second.record.run_id
    assert first.submission_payload_hash == second.submission_payload_hash
    assert len(first.submission_payload_hash) == 64
    assert len(runs.list()) == 1

    changed = RunSubmissionRequest(
        **{**request.__dict__, 'raw_config': {'model_type': 'logistic_regression', 'normalization': 'minmax'}}
    )
    with pytest.raises(RunSubmissionError) as conflict:
        service.submit(changed)
    assert conflict.value.code == 'agent_submission_key_conflict'
    assert conflict.value.retryable is False
    assert len(runs.list()) == 1


def test_human_submission_does_not_write_agent_mapping(tmp_path):
    storage = tmp_path / 'storage'; uploads = storage / 'uploads'; uploads.mkdir(parents=True)
    source = uploads / 'data.csv'; source.write_text('Index,Label,Sample_ID,0\n1,A,S1,1\n')
    datasets = DatasetRepository(storage / 'datasets.sqlite3', storage_root=storage); datasets.initialize()
    dataset = datasets.register(source, original_name='data.csv', principal=Principal())
    runs = RunRepository(storage / 'runs.sqlite3'); runs.initialize()
    service = RunSubmissionService(
        run_repository=runs, dataset_repository=datasets,
        run_dir=lambda run_id: storage / 'runs' / run_id, status_projector=project_status,
    )

    result = service.submit(RunSubmissionRequest(
        dataset_id=dataset.dataset_id, legacy_data_path=None, test_dataset_id=None,
        test_legacy_data_path=None, raw_config={'model_type': 'logistic_regression'},
        principal=Principal(), submission_source='human',
    ))

    assert result.record.state == 'queued'
    assert runs.lookup_submission_mapping('anything', principal=Principal()).status == 'missing'
