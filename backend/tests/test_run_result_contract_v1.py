from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from fastapi.testclient import TestClient
import pytest

from backend.app.http.principal import get_principal
from backend.app.main import app
from backend.app.routers import runs as runs_router
from backend.app.runs.artifacts import RunArtifactWriter
from backend.app.runs.contracts import Principal, RunRecord
from backend.app.runs.repository import RunRepository


def _patch_run_storage(monkeypatch, tmp_path) -> tuple[RunRepository, object]:
    run_root = tmp_path / 'runs'
    repository = RunRepository(tmp_path / 'runs.sqlite3')
    repository.initialize()
    monkeypatch.setattr(runs_router, 'get_run_repository', lambda: repository)
    monkeypatch.setattr(runs_router, 'get_run_dir', lambda run_id: run_root / run_id)
    return repository, run_root


def _write_ready_artifacts(
    run_dir,
    *,
    run_id: str,
    strategy: str = 'stratified_holdout',
    metrics: dict | None = None,
    cv_summary: dict | None = None,
    model_type: str = 'pls_da',
    model_family: str = 'traditional_ml',
) -> None:
    metrics = metrics or {
        'train': {'accuracy': 0.95, 'macro_precision': 0.95, 'macro_recall': 0.95, 'macro_f1': 0.95},
        'valid': {'accuracy': 0.90, 'macro_precision': 0.90, 'macro_recall': 0.90, 'macro_f1': 0.90},
        'test': {
            'accuracy': 0.75,
            'balanced_accuracy': 0.75,
            'macro_precision': 0.75,
            'macro_recall': 0.75,
            'macro_f1': 0.75,
            'confusion_matrix': [[3, 1], [1, 3]],
            'classification_report': {
                'A': {'precision': 0.75, 'recall': 0.75, 'f1-score': 0.75, 'support': 4},
                'B': {'precision': 0.75, 'recall': 0.75, 'f1-score': 0.75, 'support': 4},
            },
        },
    }
    writer = RunArtifactWriter(run_dir)
    writer.write_json('status.json', {
        'run_id': run_id,
        'state': 'running',
        'status': 'running',
        'evaluation_strategy': strategy,
        'fold_count': 2 if strategy == 'leave_one_sample_id_cv' else 1,
        'model_type': model_type,
        'model_family': model_family,
        'label_names': ['A', 'B'],
        'sample_count': 24,
    })
    writer.write_json('config.json', {'model_type': model_type, 'data_path': r'D:\private\dataset.csv'})
    writer.write_json('model_metadata.json', {
        'model_type': model_type,
        'model_family': model_family,
        'architecture_version': 'docx-classification-v2',
        'feature_count': 160,
    })
    writer.write_json('label_map.json', {'0': 'A', '1': 'B'})
    writer.write_json('split.json', {'folds': []})
    writer.write_json('metrics.json', metrics)
    writer.write_json('cv_metrics.json', {
        'strategy': strategy,
        'fold_count': 2 if strategy == 'leave_one_sample_id_cv' else 1,
        'metrics': metrics,
        'cv_summary': cv_summary or {},
        'folds': [],
    })
    writer.write_bytes('fold_metrics.csv', b'fold_index,test_macro_f1\n1,0.75\n')
    writer.write_bytes('predictions.csv', b'Index,true_label,pred_label\n1,A,A\n')
    writer.write_bytes(
        'history.csv',
        b'fold_index,epoch,train_loss,valid_accuracy\n1,1,,0.9\n',
    )
    writer.write_bytes('model.pt' if model_family == 'deep_learning' else 'model.pkl', b'private-model-object')
    writer.finalize(run_id=run_id, metadata={'model_type': model_type, 'model_family': model_family})


def _create_succeeded_run(
    repository: RunRepository,
    run_root,
    *,
    principal: Principal = Principal(),
    strategy: str = 'stratified_holdout',
    metrics: dict | None = None,
    cv_summary: dict | None = None,
    model_type: str = 'pls_da',
    model_family: str = 'traditional_ml',
) -> RunRecord:
    queued = repository.create_queued(
        dataset_id='dataset-1',
        config={
            'model_type': model_type,
            'dataset_name': 'teacher-data.csv',
            'evaluation_strategy': strategy,
            'data_path': r'D:\private\dataset.csv',
        },
        dataset_snapshot={
            'name': 'teacher-data.csv',
            'sha256': 'a' * 64,
            'curve_count': 24,
            'sample_id_count': 8,
            'class_count': 2,
            'feature_count': 160,
        },
        principal=principal,
    )
    started = datetime(2026, 7, 16, 2, 0, tzinfo=timezone.utc)
    claim = repository.claim_next(worker_id='test-worker', now=started)
    assert claim is not None and claim.run_id == queued.run_id and claim.claim_token
    _write_ready_artifacts(
        run_root / queued.run_id,
        run_id=queued.run_id,
        strategy=strategy,
        metrics=metrics,
        cv_summary=cv_summary,
        model_type=model_type,
        model_family=model_family,
    )
    return repository.finish_success(
        queued.run_id,
        claim_token=claim.claim_token,
        now=started + timedelta(seconds=12.5),
    )


def test_result_v1_is_refreshable_traceable_and_contains_only_real_analysis(tmp_path, monkeypatch) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    record = _create_succeeded_run(repository, run_root)
    client = TestClient(app)

    first = client.get(f'/api/training/runs/{record.run_id}/result')
    refreshed = client.get(f'/api/training/runs/{record.run_id}/result')

    assert first.status_code == 200
    assert refreshed.json() == first.json()
    payload = first.json()
    assert payload['schema_version'] == 'run-result-v1'
    assert payload['run']['run_id'] == record.run_id
    assert payload['run']['state'] == 'succeeded'
    assert payload['run']['result_state'] == 'ready'
    assert payload['run']['started_at']
    assert payload['run']['finished_at']
    assert payload['run']['duration_seconds'] == pytest.approx(12.5)
    assert payload['dataset'] == {
        'dataset_id': 'dataset-1',
        'name': 'teacher-data.csv',
        'sha256': 'a' * 64,
        'curve_count': 24,
        'sample_id_count': 8,
        'class_count': 2,
        'feature_count': 160,
        'test_curve_count': None,
    }
    assert payload['evaluation']['primary_aggregation'] == 'direct'
    assert payload['metrics']['primary']['macro_f1'] == pytest.approx(0.75)
    assert payload['analysis']['prediction_distribution'] == {
        'labels': ['A', 'B'],
        'true_counts': [4, 4],
        'predicted_counts': [4, 4],
    }
    assert payload['analysis']['history']['available'] is False
    assert '参数选择审计' in payload['analysis']['history']['reason']
    assert 'traditional' in payload['analysis']['training_audit']
    assert payload['analysis']['roc']['available'] is False
    assert payload['analysis']['precision_recall']['available'] is False
    assert r'D:\private' not in first.text

    artifacts = {item['name']: item for item in payload['artifacts']}
    assert artifacts['metrics.json']['downloadable'] is True
    assert artifacts['metrics.json']['download_url'].endswith('/artifact/metrics.json')
    assert artifacts['config.json']['downloadable'] is False
    assert artifacts['model.pkl']['downloadable'] is False
    assert all('path' not in item for item in payload['artifacts'])


def test_deep_result_exposes_real_epoch_history(tmp_path, monkeypatch) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    record = _create_succeeded_run(
        repository,
        run_root,
        model_type='cnn1d',
        model_family='deep_learning',
    )

    payload = TestClient(app).get(f'/api/training/runs/{record.run_id}/result').json()

    assert payload['model']['family'] == 'deep_learning'
    assert payload['analysis']['history']['available'] is True
    assert payload['analysis']['history']['rows'][0]['epoch'] == 1
    assert 'deep_training' in payload['analysis']['training_audit']


def test_result_v1_separates_pooled_oof_from_fold_mean(tmp_path, monkeypatch) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    pooled = {
        'accuracy': 1.0,
        'macro_precision': 1.0,
        'macro_recall': 1.0,
        'macro_f1': 1.0,
        'confusion_matrix': [[4, 0], [0, 4]],
        'classification_report': {},
    }
    fold_mean = {
        'train': {'accuracy': 0.9, 'macro_f1': 0.9},
        'valid': {'accuracy': 0.8, 'macro_f1': 0.8},
        'test': {'accuracy': 1.0, 'macro_f1': 0.5},
    }
    fold_std = {
        'train': {'accuracy': 0.01, 'macro_f1': 0.01},
        'valid': {'accuracy': 0.02, 'macro_f1': 0.02},
        'test': {'accuracy': 0.0, 'macro_f1': 0.0},
    }
    raw_metrics = {
        'train': fold_mean['train'],
        'valid': fold_mean['valid'],
        'test': pooled,
    }
    record = _create_succeeded_run(
        repository,
        run_root,
        strategy='leave_one_sample_id_cv',
        metrics=raw_metrics,
        cv_summary={
            'primary_test_aggregation': 'pooled_out_of_fold',
            'pooled_test': pooled,
            'fold_mean': fold_mean,
            'fold_std': fold_std,
        },
    )

    payload = TestClient(app).get(f'/api/training/runs/{record.run_id}/result').json()

    assert payload['evaluation']['primary_aggregation'] == 'pooled_oof'
    assert payload['metrics']['primary']['macro_f1'] == pytest.approx(1.0)
    assert payload['metrics']['splits']['test']['aggregation'] == 'pooled_oof'
    assert payload['metrics']['splits']['test']['values']['macro_f1'] == pytest.approx(1.0)
    assert payload['metrics']['fold_mean']['test']['macro_f1'] == pytest.approx(0.5)
    assert payload['metrics']['splits']['train']['aggregation'] == 'fold_mean'
    assert payload['metrics']['splits']['valid']['fold_std']['macro_f1'] == pytest.approx(0.02)


@pytest.mark.parametrize(
    ('state', 'expected_result_state'),
    [('queued', 'pending'), ('failed', 'failed'), ('cancelled', 'cancelled')],
)
def test_result_v1_has_explicit_non_success_states(
    tmp_path,
    monkeypatch,
    state: str,
    expected_result_state: str,
) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    record = repository.import_legacy(
        run_id=f'{state}-run',
        state=state,
        config={'model_type': 'svm', 'dataset_name': 'legacy.csv'},
        dataset_id=None,
        legacy_data_path=None,
    )
    (run_root / record.run_id).mkdir(parents=True)

    response = TestClient(app).get(f'/api/training/runs/{record.run_id}/result')

    assert response.status_code == 200
    assert response.json()['run']['result_state'] == expected_result_state


def test_failed_result_preserves_structured_error_without_exposing_traceback(tmp_path, monkeypatch) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    queued = repository.create_queued(dataset_id=None, config={'model_type': 'svm'})
    now = datetime.now(timezone.utc)
    claim = repository.claim_next(worker_id='worker', now=now)
    assert claim is not None and claim.claim_token
    failed = repository.finish_failure(
        queued.run_id,
        claim_token=claim.claim_token,
        now=now + timedelta(seconds=2),
        error='数据集标签无效',
        error_details={
            'code': 'invalid_training_data_or_config',
            'stage': 'training_validation',
            'message': '数据集标签无效',
            'retryable': False,
            'type': 'ValueError',
            'traceback': 'must-not-leak',
        },
    )
    (run_root / failed.run_id).mkdir(parents=True)

    response = TestClient(app).get(f'/api/training/runs/{failed.run_id}/result')

    assert response.status_code == 200
    assert response.json()['run']['error'] == {
        'code': 'invalid_training_data_or_config',
        'stage': 'training_validation',
        'message': '数据集标签无效',
        'retryable': False,
        'type': 'ValueError',
    }
    assert 'must-not-leak' not in response.text


def test_succeeded_result_distinguishes_missing_corrupt_and_partial_artifacts(tmp_path, monkeypatch) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    missing = repository.import_legacy(
        run_id='missing-manifest',
        state='succeeded',
        config={'model_type': 'pls_da'},
        dataset_id=None,
        legacy_data_path=None,
    )
    (run_root / missing.run_id).mkdir(parents=True)
    corrupt = repository.import_legacy(
        run_id='corrupt-manifest',
        state='succeeded',
        config={'model_type': 'pls_da'},
        dataset_id=None,
        legacy_data_path=None,
    )
    corrupt_dir = run_root / corrupt.run_id
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / 'manifest.json').write_text('{not-json', encoding='utf-8')
    partial = repository.import_legacy(
        run_id='partial-result',
        state='succeeded',
        config={'model_type': 'pls_da'},
        dataset_id=None,
        legacy_data_path=None,
    )
    partial_writer = RunArtifactWriter(run_root / partial.run_id)
    partial_writer.write_json('metrics.json', {'test': {'accuracy': 0.5}})
    partial_writer.finalize(run_id=partial.run_id)
    client = TestClient(app)

    missing_payload = client.get(f'/api/training/runs/{missing.run_id}/result').json()
    corrupt_payload = client.get(f'/api/training/runs/{corrupt.run_id}/result').json()
    partial_payload = client.get(f'/api/training/runs/{partial.run_id}/result').json()

    assert missing_payload['run']['result_state'] == 'missing_manifest'
    assert corrupt_payload['run']['result_state'] == 'corrupt_manifest'
    assert partial_payload['run']['result_state'] == 'partial'
    assert any('Manifest 缺失' in warning for warning in missing_payload['warnings'])
    assert any('Manifest 损坏' in warning for warning in corrupt_payload['warnings'])
    assert any('部分结果文件缺失' in warning for warning in partial_payload['warnings'])


def test_summary_projection_is_scoped_paginated_and_does_not_return_full_results(tmp_path, monkeypatch) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    for index in range(3):
        record = repository.create_queued(
            dataset_id=f'dataset-{index}',
            config={
                'model_type': 'pls_da',
                'dataset_name': f'dataset-{index}.csv',
                'data_path': rf'D:\private\dataset-{index}.csv',
            },
            dataset_snapshot={'name': f'dataset-{index}.csv'},
        )
        (run_root / record.run_id).mkdir(parents=True)
    client = TestClient(app)

    first = client.get('/api/training/runs', params={'projection': 'summary', 'limit': 2})
    second = client.get(
        '/api/training/runs',
        params={'projection': 'summary', 'limit': 2, 'cursor': first.json()['next_cursor']},
    )

    assert first.status_code == 200
    assert len(first.json()['items']) == 2
    assert first.json()['next_cursor']
    assert len(second.json()['items']) == 1
    assert second.json()['next_cursor'] is None
    combined_ids = [item['run_id'] for item in first.json()['items'] + second.json()['items']]
    assert len(set(combined_ids)) == 3
    assert all('metrics' not in item and 'config' not in item for item in first.json()['items'])
    assert r'D:\private' not in first.text


def test_run_routes_hide_other_principals_and_local_legacy_runs(tmp_path, monkeypatch) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    owner_a = Principal(owner_id='owner-a', tenant_id='tenant')
    owner_b = Principal(owner_id='owner-b', tenant_id='tenant')
    run_a = repository.create_queued(dataset_id=None, config={'model_type': 'pls_da'}, principal=owner_a)
    run_b = repository.create_queued(dataset_id=None, config={'model_type': 'svm'}, principal=owner_b)
    local_run = repository.create_queued(dataset_id=None, config={'model_type': 'pca_lda'})
    for record in (run_a, run_b, local_run):
        (run_root / record.run_id).mkdir(parents=True)
    app.dependency_overrides[get_principal] = lambda: owner_a
    try:
        client = TestClient(app)
        listing = client.get('/api/training/runs', params={'projection': 'summary'})

        assert [item['run_id'] for item in listing.json()['items']] == [run_a.run_id]
        assert client.get(f'/api/training/runs/{run_a.run_id}').status_code == 200
        assert client.get(f'/api/training/runs/{run_b.run_id}').status_code == 404
        assert client.get(f'/api/training/runs/{local_run.run_id}').status_code == 404
        assert client.get(f'/api/training/runs/{run_b.run_id}/result').status_code == 404
        assert client.get(f'/api/training/runs/{local_run.run_id}/artifact/metrics.json').status_code == 404
    finally:
        app.dependency_overrides.pop(get_principal, None)


def test_download_requires_success_integrity_and_catalog_but_keeps_local_legacy_compatibility(
    tmp_path,
    monkeypatch,
) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    succeeded = _create_succeeded_run(repository, run_root)
    queued = repository.create_queued(dataset_id=None, config={'model_type': 'pls_da'})
    queued_writer = RunArtifactWriter(run_root / queued.run_id)
    queued_writer.write_json('metrics.json', {'test': {'accuracy': 0.5}})
    queued_writer.finalize(run_id=queued.run_id)
    legacy_dir = run_root / 'legacy-no-db'
    legacy_dir.mkdir(parents=True)
    (legacy_dir / 'legacy.txt').write_text('legacy', encoding='utf-8')
    (legacy_dir / 'manifest.json').write_text(
        json.dumps({'artifacts': {'legacy.txt': {'downloadable': True}}}),
        encoding='utf-8',
    )
    client = TestClient(app)

    assert client.get(f'/api/training/runs/{succeeded.run_id}/artifact/metrics.json').status_code == 200
    assert client.get(f'/api/training/runs/{succeeded.run_id}/artifact/config.json').status_code == 403
    assert client.get(f'/api/training/runs/{succeeded.run_id}/artifact/model.pkl').status_code == 403
    assert client.get(f'/api/training/runs/{queued.run_id}/artifact/metrics.json').status_code == 409
    assert client.get('/api/training/runs/legacy-no-db/artifact/legacy.txt').text == 'legacy'

    (run_root / succeeded.run_id / 'metrics.json').write_text('{"tampered":true}', encoding='utf-8')
    tampered = client.get(f'/api/training/runs/{succeeded.run_id}/artifact/metrics.json')
    assert tampered.status_code == 409
    assert tampered.json()['detail'] == '文件完整性校验失败'
