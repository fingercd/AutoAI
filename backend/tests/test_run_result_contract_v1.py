from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

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
        'train': {
            'accuracy': 0.95,
            'macro_precision': 0.95,
            'macro_recall': 0.95,
            'macro_f1': 0.95,
            'confusion_matrix': [[9, 1], [0, 10]],
            'classification_report': {
                'A': {'precision': 1.0, 'recall': 0.9, 'f1-score': 0.9474, 'support': 10},
                'B': {'precision': 0.9091, 'recall': 1.0, 'f1-score': 0.9524, 'support': 10},
            },
        },
        'valid': {
            'accuracy': 0.90,
            'macro_precision': 0.90,
            'macro_recall': 0.90,
            'macro_f1': 0.90,
            'confusion_matrix': [[5, 1], [0, 4]],
            'classification_report': {
                'A': {'precision': 1.0, 'recall': 0.8333, 'f1-score': 0.9091, 'support': 6},
                'B': {'precision': 0.8, 'recall': 1.0, 'f1-score': 0.8889, 'support': 4},
            },
        },
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
        'sample_feature_importance': {
            'status': 'ready',
            'artifact': 'sample_feature_importance.json',
            'csv_artifact': 'sample_feature_importance.csv',
            'method': 'sample_occlusion_log_loss',
            'sample_count': 8,
        },
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
    writer.write_json('sample_feature_importance.json', {
        'status': 'ready',
        'method': 'sample_occlusion_log_loss',
        'sample_count': 1,
        'samples': [],
    })
    writer.write_bytes(
        'sample_feature_importance.csv',
        b'sample_id,start_index,end_index,importance\nsample-1,0,1,0.5\n',
    )
    if model_family == 'deep_learning':
        writer.write_bytes(
            'history.csv',
            b'fold_index,epoch,train_loss,valid_accuracy\n1,1,0.4,0.9\n',
        )
    writer.write_bytes('model.pt' if model_family == 'deep_learning' else 'model.pkl', b'private-model-object')
    writer.finalize(
        run_id=run_id,
        metadata={
            'model_type': model_type,
            'model_family': model_family,
            'evaluation_strategy': strategy,
        },
    )


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
    config_strategy_field: str = 'evaluation_strategy',
) -> RunRecord:
    queued = repository.create_queued(
        dataset_id='dataset-1',
        config={
            'model_type': model_type,
            'dataset_name': 'teacher-data.csv',
            config_strategy_field: strategy,
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
    assert set(payload['analysis']['splits']) == {'train', 'valid', 'test'}
    assert payload['analysis']['splits']['train']['aggregation'] == 'direct'
    assert payload['analysis']['splits']['train']['confusion_matrix'] == [[9, 1], [0, 10]]
    assert payload['analysis']['splits']['valid']['prediction_distribution'] == {
        'labels': ['A', 'B'],
        'true_counts': [6, 4],
        'predicted_counts': [5, 5],
    }
    assert payload['analysis']['splits']['test']['classification_report']['A']['support'] == 4
    assert payload['analysis']['history']['available'] is False
    assert '参数选择审计' in payload['analysis']['history']['reason']
    assert 'traditional' in payload['analysis']['training_audit']
    assert payload['analysis']['roc']['available'] is False
    assert payload['analysis']['precision_recall']['available'] is False
    assert 'global' not in payload['explainability']
    assert payload['explainability']['samples']['artifact'] == 'sample_feature_importance.json'
    assert r'D:\private' not in first.text

    artifacts = {item['name']: item for item in payload['artifacts']}
    assert artifacts['metrics.json']['downloadable'] is True
    assert artifacts['metrics.json']['download_url'].endswith('/artifact/metrics.json')
    assert artifacts['config.json']['downloadable'] is False
    assert artifacts['model.pkl']['downloadable'] is False
    assert artifacts['sample_feature_importance.json']['downloadable'] is True
    assert 'feature_importance.json' not in artifacts
    assert 'feature_importance.csv' not in artifacts
    assert artifacts['history.csv']['applicable'] is False
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
    assert payload['analysis']['splits']['train']['aggregation'] == 'pooled_cross_fold'
    assert payload['analysis']['splits']['valid']['aggregation'] == 'pooled_cross_fold'
    assert payload['analysis']['splits']['test']['aggregation'] == 'pooled_oof'


@pytest.mark.parametrize('strategy', ['stratified_holdout', 'external_test_holdout'])
def test_summary_projection_includes_dataset_training_time_duration_and_test_macro_f1(
    tmp_path,
    monkeypatch,
    strategy: str,
) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    record = _create_succeeded_run(repository, run_root, strategy=strategy)

    def unexpected_full_projection(*_args, **_kwargs):
        raise AssertionError('summary 列表不应构造完整 result projection')

    monkeypatch.setattr(runs_router, 'project_run_result', unexpected_full_projection)

    response = TestClient(app).get(
        '/api/training/runs',
        params={'projection': 'summary', 'limit': 20},
    )

    assert response.status_code == 200
    summary = next(item for item in response.json()['items'] if item['run_id'] == record.run_id)
    assert summary['dataset_name'] == 'teacher-data.csv'
    assert summary['created_at']
    assert summary['started_at']
    assert summary['finished_at']
    assert summary['duration_seconds'] == pytest.approx(12.5)
    assert summary['test_macro_f1'] == pytest.approx(0.75)
    assert 'config' not in summary


def test_summary_projection_uses_cv_pooled_test_macro_f1_not_fold_mean(tmp_path, monkeypatch) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    pooled_test = {'accuracy': 0.88, 'macro_f1': 0.88}
    record = _create_succeeded_run(
        repository,
        run_root,
        strategy='leave_one_sample_id_cv',
        config_strategy_field='split_mode',
        metrics={
            'test': {
                'accuracy': 0.61,
                'macro_f1': 0.61,
                'aggregation': 'pooled_out_of_fold',
            },
        },
        cv_summary={
            'primary_test_aggregation': 'pooled_out_of_fold',
            'pooled_test': pooled_test,
            'fold_mean': {'test': {'accuracy': 0.42, 'macro_f1': 0.42}},
        },
    )
    client = TestClient(app)

    summary_response = client.get('/api/training/runs', params={'projection': 'summary'})
    result_response = client.get(f'/api/training/runs/{record.run_id}/result')

    assert summary_response.status_code == 200
    assert result_response.status_code == 200
    summary = next(item for item in summary_response.json()['items'] if item['run_id'] == record.run_id)
    result = result_response.json()
    assert summary['test_macro_f1'] == pytest.approx(0.88)
    assert summary['test_macro_f1'] == pytest.approx(result['metrics']['primary']['macro_f1'])
    assert summary['test_macro_f1'] != pytest.approx(0.42)
    assert summary['test_macro_f1'] != pytest.approx(0.61)
    assert 'metrics' not in summary


def test_summary_projection_returns_null_for_invalid_or_ambiguous_test_macro_f1(
    tmp_path,
    monkeypatch,
) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    invalid_metrics = [
        {'test': {'macro_f1': None}},
        {'test': {'macro_f1': float('nan')}},
        {'test': {'macro_f1': float('inf')}},
        {'test': {'macro_f1': -0.01}},
        {'test': {'macro_f1': 1.01}},
        {'test': {'macro_f1': 10**400}},
        {'test': {'macro_f1': '0.75'}},
        {'test': {'macro_f1': True}},
        {'macro_f1': 0.99, 'valid': {'macro_f1': 0.99}},
    ]
    run_ids = {
        _create_succeeded_run(repository, run_root, metrics=metrics).run_id
        for metrics in invalid_metrics
    }

    response = TestClient(app).get(
        '/api/training/runs',
        params={'projection': 'summary', 'limit': 20},
    )

    assert response.status_code == 200
    summaries = [item for item in response.json()['items'] if item['run_id'] in run_ids]
    assert len(summaries) == len(run_ids)
    assert all(item['test_macro_f1'] is None for item in summaries)


def test_summary_projection_rejects_invalid_cv_pooled_value_without_using_fold_mean(
    tmp_path,
    monkeypatch,
) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    record = _create_succeeded_run(
        repository,
        run_root,
        strategy='leave_one_sample_id_cv',
        metrics={
            'test': {
                'macro_f1': 0.77,
                'aggregation': 'pooled_out_of_fold',
            },
        },
        cv_summary={
            'primary_test_aggregation': 'pooled_out_of_fold',
            'pooled_test': {'macro_f1': None},
            'fold_mean': {'test': {'macro_f1': 0.99}},
        },
    )

    response = TestClient(app).get('/api/training/runs', params={'projection': 'summary'})

    summary = next(item for item in response.json()['items'] if item['run_id'] == record.run_id)
    assert summary['test_macro_f1'] is None


def test_summary_projection_requires_manifest_size_and_sha256_integrity(tmp_path, monkeypatch) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    missing = _create_succeeded_run(repository, run_root)
    damaged = _create_succeeded_run(repository, run_root)
    unverifiable = _create_succeeded_run(repository, run_root)

    (run_root / missing.run_id / 'metrics.json').unlink()
    damaged_metrics = run_root / damaged.run_id / 'metrics.json'
    original = damaged_metrics.read_bytes()
    tampered = original.replace(b'"macro_f1": 0.75', b'"macro_f1": 0.95', 1)
    assert tampered != original
    assert len(tampered) == len(original)
    damaged_metrics.write_bytes(tampered)
    manifest_path = run_root / unverifiable.run_id / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['artifacts']['metrics.json'].pop('size_bytes')
    manifest['artifacts']['metrics.json'].pop('sha256')
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    response = TestClient(app).get('/api/training/runs', params={'projection': 'summary'})

    summaries = {item['run_id']: item for item in response.json()['items']}
    assert summaries[missing.run_id]['result_state'] == 'partial'
    assert summaries[missing.run_id]['test_macro_f1'] is None
    # descriptors 的快速大小检查仍会认为 ready；标量读取器必须由 SHA-256 拒绝同大小篡改。
    assert summaries[damaged.run_id]['result_state'] == 'ready'
    assert summaries[damaged.run_id]['test_macro_f1'] is None
    assert summaries[unverifiable.run_id]['result_state'] == 'ready'
    assert summaries[unverifiable.run_id]['test_macro_f1'] is None


@pytest.mark.parametrize('state', ['queued', 'running', 'failed', 'cancelled'])
def test_summary_projection_never_reads_metrics_for_non_success_states(
    tmp_path,
    monkeypatch,
    state: str,
) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    record = repository.import_legacy(
        run_id=f'{state}-summary-run',
        state=state,
        config={'model_type': 'pls_da', 'evaluation_strategy': 'stratified_holdout'},
        dataset_id=None,
        legacy_data_path=None,
    )
    _write_ready_artifacts(run_root / record.run_id, run_id=record.run_id)

    response = TestClient(app).get('/api/training/runs', params={'projection': 'summary'})

    summary = next(item for item in response.json()['items'] if item['run_id'] == record.run_id)
    assert summary['test_macro_f1'] is None


def test_create_run_rejects_live_incompatible_worker_before_queueing(tmp_path, monkeypatch) -> None:
    repository, _run_root = _patch_run_storage(monkeypatch, tmp_path)
    repository.record_worker_heartbeat(
        worker_id='legacy-worker',
        now=datetime.now(timezone.utc),
        contract_version=None,
    )
    monkeypatch.setattr(
        runs_router,
        'resolve_training_data_reference',
        lambda payload, principal: SimpleNamespace(
            dataset_id='dataset-1',
            legacy_path=None,
            dataset_name='teacher-data.csv',
            test_dataset_id=None,
            test_legacy_path=None,
            test_dataset_name=None,
        ),
    )
    monkeypatch.setattr(
        runs_router,
        '_dataset_snapshot_for_reference',
        lambda **kwargs: {
            'dataset_id': 'dataset-1',
            'name': 'teacher-data.csv',
            'sha256': 'a' * 64,
        },
    )

    response = TestClient(app).post(
        '/api/training/runs',
        json={'dataset_id': 'dataset-1', 'config': {'model_type': 'pls_da'}},
    )

    assert response.status_code == 503
    assert response.json()['detail']['code'] == 'worker_contract_mismatch'
    assert repository.list() == []


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


def test_legacy_manifest_status_size_changes_do_not_make_result_partial(tmp_path, monkeypatch) -> None:
    repository, run_root = _patch_run_storage(monkeypatch, tmp_path)
    record = repository.import_legacy(
        run_id='legacy-volatile-status',
        state='succeeded',
        config={'model_type': 'pls_da', 'dataset_name': 'legacy.csv'},
        dataset_id=None,
        legacy_data_path=None,
    )
    run_dir = run_root / record.run_id
    run_dir.mkdir(parents=True)
    (run_dir / 'status.json').write_text(
        json.dumps({'state': 'succeeded', 'sample_count': 90}),
        encoding='utf-8',
    )
    (run_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'run_id': record.run_id,
                'artifacts': {
                    'status.json': {
                        'downloadable': False,
                        'size_bytes': 1,
                        'sha256': '0' * 64,
                    },
                },
            }
        ),
        encoding='utf-8',
    )

    payload = TestClient(app).get(f'/api/training/runs/{record.run_id}/result').json()
    descriptors = {item['name']: item for item in payload['artifacts']}

    assert payload['run']['result_state'] == 'ready'
    assert descriptors['status.json']['integrity'] == 'volatile'
    assert any('旧版 Worker' in warning for warning in payload['warnings'])


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
