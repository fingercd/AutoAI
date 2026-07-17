import json
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.app.routers import runs as runs_router
from backend.app.runs.contracts import RunRecord
from backend.app.runs.status_projection import project_status


def _record(*, state: str = 'running', version: int = 2) -> RunRecord:
    return RunRecord(
        run_id='run-1',
        state=state,
        version=version,
        dataset_id='ds-1',
        legacy_data_path=None,
        config={'model_type': 'cnn1d'},
        progress={},
        claim_token='claim-1' if state == 'running' else None,
        worker_id='worker-1' if state == 'running' else None,
        lease_expires_at=None,
    )


def test_dataset_name_for_pre_snapshot_run_is_recovered_from_dataset_repository(monkeypatch):
    class Repository:
        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            return None

        def resolve_system(self, dataset_id, *, legacy_path):
            assert dataset_id == 'ds-1'
            assert legacy_path is None
            return SimpleNamespace(original_name='teacher_dataset.csv')

    monkeypatch.setattr(runs_router, 'DatasetRepository', Repository)

    assert runs_router._dataset_name_for_record(_record()) == 'teacher_dataset.csv'


def test_project_status_preserves_result_fields_while_record_state_stays_authoritative(tmp_path):
    run_dir = tmp_path / 'run-1'
    run_dir.mkdir()
    (run_dir / 'status.json').write_text(
        json.dumps(
            {
                'run_id': 'stale-id',
                'status': 'success',
                'state': 'succeeded',
                'version': 1,
                'dataset_id': 'stale-dataset',
                'metrics': {'test': {'accuracy': 0.875}},
                'history': [{'epoch': 1, 'valid_accuracy': 0.75}],
                'actual_epochs': 1,
            }
        ),
        encoding='utf-8',
    )

    payload = project_status(run_dir, _record(), completed_folds=1)

    assert payload['metrics']['test']['accuracy'] == 0.875
    assert payload['history'][0]['epoch'] == 1
    assert payload['actual_epochs'] == 1
    assert payload['run_id'] == 'run-1'
    assert payload['status'] == 'running'
    assert payload['state'] == 'running'
    assert payload['version'] == 2
    assert payload['dataset_id'] == 'ds-1'
    assert payload['completed_folds'] == 1


@pytest.mark.parametrize(
    'stored_status',
    [
        {'run_id': 'run-1', 'status': 'success', 'state': 'succeeded', 'version': 3, 'dataset_id': 'ds-1'},
        {'run_id': 'run-1', 'status': 'running', 'state': 'running', 'version': 2, 'dataset_id': 'ds-1'},
        {
            'run_id': 'run-1',
            'status': 'success',
            'state': 'succeeded',
            'version': 3,
            'dataset_id': 'ds-1',
            'history': [
                {'fold_index': 1, 'epoch': 1, 'train_loss': 0.5, 'valid_accuracy': 0.75},
                {'fold_index': 1, 'epoch': 2, 'train_loss': 0.3, 'valid_accuracy': 0.8},
            ],
        },
        None,
    ],
)
def test_projection_recovers_legacy_success_status_from_artifacts_and_persists_it(tmp_path, monkeypatch, stored_status):
    run_dir = tmp_path / 'run-1'
    run_dir.mkdir()
    if stored_status is not None:
        (run_dir / 'status.json').write_text(json.dumps(stored_status), encoding='utf-8')
    metrics = {
        'train': {'accuracy': 0.9},
        'valid': {'accuracy': 0.8},
        'test': {'accuracy': 0.7, 'macro_f1': 0.65, 'confusion_matrix': [[3, 1], [1, 3]]},
    }
    (run_dir / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    (run_dir / 'config.json').write_text(
        json.dumps({'model_type': 'cnn1d', 'epochs': 5, 'fold_count': 1, 'evaluation_strategy': 'stratified_holdout'}),
        encoding='utf-8',
    )
    pd.DataFrame(
        [
            {'fold_index': 1, 'epoch': 1, 'train_loss': 0.5, 'valid_accuracy': 0.75},
            {'fold_index': 1, 'epoch': 2, 'train_loss': 0.3, 'valid_accuracy': 0.8},
        ]
    ).to_csv(run_dir / 'history.csv', index=False)
    (run_dir / 'sample_feature_importance.json').write_text(json.dumps({'status': 'ready'}), encoding='utf-8')
    (run_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'created_at': '2026-07-11T10:00:00+00:00',
                'metadata': {'model_type': 'cnn1d', 'model_family': 'deep_learning'},
                'artifacts': {
                    'model.pt': {'downloadable': True},
                    'sample_feature_importance.json': {'downloadable': True},
                    'sample_feature_importance.csv': {'downloadable': True},
                },
            }
        ),
        encoding='utf-8',
    )
    monkeypatch.setattr(runs_router, 'get_run_dir', lambda _: run_dir)
    record = _record(state='succeeded', version=3)

    payload = runs_router._projection(record)

    assert payload['metrics'] == metrics
    assert len(payload['history']) == 2
    assert payload['actual_epochs'] == 2
    assert payload['target_epochs'] == 5
    assert payload['model_type'] == 'cnn1d'
    assert payload['model_family'] == 'deep_learning'
    assert payload['model_artifact'] == 'model.pt'
    assert payload['completed_at'] == '2026-07-11T10:00:00+00:00'
    assert payload['sample_feature_importance']['artifact'] == 'sample_feature_importance.json'
    assert 'feature_importance' not in payload
    persisted = json.loads((run_dir / 'status.json').read_text(encoding='utf-8'))
    assert persisted['metrics'] == metrics
    assert persisted['actual_epochs'] == 2
