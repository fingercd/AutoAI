from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest
from fastapi.testclient import TestClient

from backend.app.datasets.repository import DatasetIntegrityError, DatasetRepository
from backend.app.runs.artifacts import RunArtifactWriter
from backend.app.runs.contracts import Principal
from backend.app.runs.migration import import_legacy_runs
from backend.app.runs.repository import RunNotFound, RunRepository
from backend.app.runs.worker import RunWorker


def test_run_metadata_snapshot_scope_and_duration_are_persistent(tmp_path) -> None:
    repository = RunRepository(tmp_path / 'runs.sqlite3')
    repository.initialize()
    owner = Principal(owner_id='owner-a', tenant_id='tenant-a')
    other = Principal(owner_id='owner-b', tenant_id='tenant-a')
    created = repository.create_queued(
        dataset_id='dataset-a',
        config={'model_type': 'pls_da'},
        dataset_snapshot={'name': 'dataset.csv', 'sha256': 'a' * 64},
        principal=owner,
    )
    assert created.created_at
    assert created.dataset_snapshot['sha256'] == 'a' * 64
    assert repository.get_scoped(created.run_id, principal=owner).run_id == created.run_id
    with pytest.raises(RunNotFound):
        repository.get_scoped(created.run_id, principal=other)

    started = datetime(2026, 7, 16, 1, 2, 3, tzinfo=timezone.utc)
    claim = repository.claim_next(worker_id='worker-a', now=started)
    assert claim is not None and claim.started_at == started.isoformat()
    finished = repository.finish_success(
        claim.run_id,
        claim_token=claim.claim_token or '',
        now=started + timedelta(seconds=7.25),
    )
    assert finished.finished_at == (started + timedelta(seconds=7.25)).isoformat()


def test_dataset_hash_is_recomputed_before_training(tmp_path) -> None:
    storage = tmp_path / 'storage'
    upload = storage / 'uploads' / 'data.csv'
    upload.parent.mkdir(parents=True)
    upload.write_text('Index,Name,XXX,Intensity,Label,Sample_ID\n', encoding='utf-8')
    repository = DatasetRepository(tmp_path / 'datasets.sqlite3', storage_root=storage)
    repository.initialize()
    record = repository.register(upload, original_name='data.csv', principal=Principal())

    assert repository.verify_integrity(record, expected_sha256=record.sha256) == record.sha256
    upload.write_text('changed', encoding='utf-8')
    with pytest.raises(DatasetIntegrityError):
        repository.verify_integrity(record, expected_sha256=record.sha256)


def test_worker_records_heartbeat_and_structured_error(tmp_path) -> None:
    repository = RunRepository(tmp_path / 'runs.sqlite3')
    repository.initialize()
    created = repository.create_queued(dataset_id=None, config={'model_type': 'svm'})
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)

    def explode(_):
        raise ValueError('invalid labels')

    worker = RunWorker(
        repository=repository,
        worker_id='worker-a',
        execute=explode,
        now=lambda: now,
    )
    assert worker.run_once() is True
    failed = repository.get(created.run_id)
    assert failed.error_details == {
        'code': 'invalid_training_data_or_config',
        'stage': 'training_validation',
        'message': 'invalid labels',
        'type': 'ValueError',
        'retryable': False,
    }
    health = repository.worker_health(now=now + timedelta(seconds=1), stale_seconds=15)
    assert health['available'] is True
    assert health['workers'][0]['worker_id'] == 'worker-a'


def test_worker_does_not_persist_absolute_paths_in_public_error_fields(tmp_path) -> None:
    repository = RunRepository(tmp_path / 'runs.sqlite3')
    repository.initialize()
    created = repository.create_queued(dataset_id=None, config={'model_type': 'svm'})

    def missing(_):
        raise FileNotFoundError(r'D:\secret\teacher-data.csv')

    worker = RunWorker(
        repository=repository,
        worker_id='worker-safe-error',
        execute=missing,
        now=lambda: datetime.now(timezone.utc),
    )
    assert worker.run_once() is True
    failed = repository.get(created.run_id)
    assert failed.error == '训练数据文件不可用，请重新上传或检查服务器存储'
    assert failed.error_details['message'] == failed.error
    assert r'D:\secret' not in json.dumps(failed.error_details, ensure_ascii=False)
    from backend.app.routers.runs import _summary_projection
    assert r'D:\secret' not in json.dumps(_summary_projection(failed), ensure_ascii=False)


def test_scoped_cancel_and_delete_never_cross_principal_boundaries(tmp_path) -> None:
    repository = RunRepository(tmp_path / 'runs.sqlite3')
    repository.initialize()
    owner = Principal(owner_id='owner-a', tenant_id='tenant')
    other = Principal(owner_id='owner-b', tenant_id='tenant')
    queued = repository.create_queued(dataset_id=None, config={'model_type': 'svm'}, principal=owner)

    with pytest.raises(RunNotFound):
        repository.cancel_scoped(queued.run_id, now=datetime.now(timezone.utc), principal=other)
    assert repository.get(queued.run_id).state == 'queued'

    cancelled = repository.cancel_scoped(queued.run_id, now=datetime.now(timezone.utc), principal=owner)
    assert cancelled.state == 'cancelled'
    with pytest.raises(RunNotFound):
        repository.delete_terminal_scoped(cancelled.run_id, principal=other)
    assert repository.exists(cancelled.run_id)
    repository.delete_terminal_scoped(cancelled.run_id, principal=owner)
    assert not repository.exists(cancelled.run_id)


def test_legacy_migration_supports_dry_run_and_explicit_unowned_rebind(tmp_path) -> None:
    run_root = tmp_path / 'runs'
    status_dir = run_root / 'legacy-run'
    status_dir.mkdir(parents=True)
    (status_dir / 'status.json').write_text(
        json.dumps({'run_id': 'legacy-run', 'status': 'success', 'config': {'model_type': 'pls_da'}}),
        encoding='utf-8',
    )
    repository = RunRepository(tmp_path / 'runs.sqlite3')
    repository.initialize()

    assert import_legacy_runs(run_root=run_root, repository=repository, dry_run=True) == 1
    assert not repository.exists('legacy-run')
    assert import_legacy_runs(run_root=run_root, repository=repository) == 1
    assert import_legacy_runs(run_root=run_root, repository=repository) == 0

    owner = Principal(owner_id='server-owner', tenant_id='server-tenant')
    assert import_legacy_runs(
        run_root=run_root,
        repository=repository,
        principal=owner,
        dry_run=True,
        rebind_unowned=True,
    ) == 1
    assert import_legacy_runs(
        run_root=run_root,
        repository=repository,
        principal=owner,
        rebind_unowned=True,
    ) == 1
    assert repository.get_scoped('legacy-run', principal=owner).owner_id == 'server-owner'
    assert import_legacy_runs(
        run_root=run_root,
        repository=repository,
        principal=owner,
        rebind_unowned=True,
    ) == 0


def test_legacy_migration_skips_corrupt_status_and_cli_rejects_partial_scope(tmp_path, monkeypatch) -> None:
    from backend.app.runs import migration

    run_root = tmp_path / 'runs'
    broken = run_root / 'broken'
    broken.mkdir(parents=True)
    (broken / 'status.json').write_text('{broken', encoding='utf-8')
    repository = RunRepository(tmp_path / 'runs.sqlite3')
    repository.initialize()
    warnings: list[str] = []

    assert import_legacy_runs(
        run_root=run_root,
        repository=repository,
        warning_sink=warnings.append,
    ) == 0
    assert warnings and '跳过损坏状态文件' in warnings[0]

    monkeypatch.setattr(
        'sys.argv',
        ['migration', '--run-root', str(run_root), '--database', str(tmp_path / 'cli.sqlite3'), '--owner-id', 'only-owner'],
    )
    with pytest.raises(SystemExit) as exc_info:
        migration.main()
    assert exc_info.value.code == 2


def test_config_download_policy_allows_sanitized_config_but_blocks_server_paths(tmp_path) -> None:
    safe = RunArtifactWriter(tmp_path / 'safe')
    safe.write_json('config.json', {'model_type': 'pls_da', 'dataset_name': 'data.csv'})
    safe_manifest = safe.finalize(run_id='safe')
    assert safe_manifest['artifacts']['config.json']['downloadable'] is True

    unsafe = RunArtifactWriter(tmp_path / 'unsafe')
    unsafe.write_json('config.json', {'model_type': 'pls_da', 'data_path': r'D:\private\data.csv'})
    unsafe_manifest = unsafe.finalize(run_id='unsafe')
    assert unsafe_manifest['artifacts']['config.json']['downloadable'] is False
    assert unsafe_manifest['artifacts']['config.json']['download_reason']


def test_delete_cleanup_failure_restores_run_scope_metadata_and_artifacts(tmp_path, monkeypatch) -> None:
    from backend.app.main import app
    from backend.app.routers import runs

    repository = RunRepository(tmp_path / 'runs.sqlite3')
    repository.initialize()
    owner = Principal(owner_id='owner', tenant_id='tenant')
    record = repository.import_legacy(
        run_id='restore-delete-run',
        state='succeeded',
        config={'model_type': 'pls_da', 'dataset_name': 'data.csv'},
        dataset_id='dataset-1',
        legacy_data_path=None,
        principal=owner,
        dataset_snapshot={'name': 'data.csv', 'sha256': 'a' * 64},
        created_at='2026-07-16T00:00:00+00:00',
        started_at='2026-07-16T00:01:00+00:00',
        finished_at='2026-07-16T00:02:00+00:00',
        manifest_name='manifest.json',
    )
    run_root = tmp_path / 'runs'
    run_dir = run_root / record.run_id
    run_dir.mkdir(parents=True)
    (run_dir / 'manifest.json').write_text(json.dumps({'run_id': record.run_id, 'artifacts': {}}), encoding='utf-8')
    monkeypatch.setattr(runs, 'get_run_repository', lambda: repository)
    monkeypatch.setattr(runs, 'get_run_dir', lambda run_id: run_root / run_id)
    real_rmtree = runs.shutil.rmtree
    calls = 0

    def fail_quarantine_cleanup(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_rmtree(path)
        raise PermissionError('locked')

    monkeypatch.setattr(runs.shutil, 'rmtree', fail_quarantine_cleanup)
    from backend.app.http.principal import get_principal
    app.dependency_overrides[get_principal] = lambda: owner
    try:
        response = TestClient(app).delete(f'/api/training/runs/{record.run_id}')
    finally:
        app.dependency_overrides.pop(get_principal, None)

    assert response.status_code == 500
    restored = repository.get_scoped(record.run_id, principal=owner)
    assert restored.dataset_snapshot == record.dataset_snapshot
    assert restored.created_at == record.created_at
    assert restored.started_at == record.started_at
    assert restored.finished_at == record.finished_at
    assert restored.manifest_name == 'manifest.json'
    assert run_dir.is_dir()
