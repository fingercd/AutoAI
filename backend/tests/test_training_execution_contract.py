import json

from backend.app.runs.contracts import RunRecord
from backend.app.runs.execution import TrainingExecution


def test_execution_checks_repository_cancellation_before_artifact_commit(tmp_path):
    calls = []

    class Repository:
        def assert_active(self, run_id: str, *, claim_token: str, now):
            calls.append((run_id, claim_token))

    record = RunRecord(
        run_id='run-1', state='running', version=1, dataset_id='ds-1', legacy_data_path=None,
        config={'model_type': 'pls_da'}, progress={}, claim_token='claim-1', worker_id='worker-1', lease_expires_at=None,
    )
    execution = TrainingExecution(repository=Repository(), run_dir=tmp_path / 'run-1')
    assert execution.cancel_check(record) is None
    assert calls == [('run-1', 'claim-1')]


def test_execution_projects_complete_training_result_before_manifest(tmp_path):
    class Repository:
        def assert_active(self, run_id: str, *, claim_token: str, now):
            return None

    class Execution(TrainingExecution):
        def _run_legacy_training(self, record, *, data_path, test_data_path):
            return {
                'run_id': record.run_id,
                'status': 'success',
                'metrics': {'train': {'accuracy': 0.9}, 'valid': {'accuracy': 0.8}, 'test': {'accuracy': 0.7}},
                'history': [{'epoch': 1, 'valid_accuracy': 0.8}],
                'actual_epochs': 1,
                'sample_count': 12,
                'completed_at': '2026-07-11T10:00:00+00:00',
                'model_type': 'pls_da',
                'model_family': 'traditional_ml',
                'model_artifact': 'model.pkl',
            }

    record = RunRecord(
        run_id='run-1', state='running', version=1, dataset_id='ds-1', legacy_data_path=None,
        config={'model_type': 'pls_da'}, progress={}, claim_token='claim-1', worker_id='worker-1', lease_expires_at=None,
    )
    run_dir = tmp_path / 'run-1'
    execution = Execution(repository=Repository(), run_dir=run_dir)

    result = execution.execute(record, data_path=tmp_path / 'data.csv')

    assert result == {'manifest_name': 'manifest.json'}
    status = json.loads((run_dir / 'status.json').read_text(encoding='utf-8'))
    assert status['status'] == 'running'
    assert status['metrics']['test']['accuracy'] == 0.7
    assert status['actual_epochs'] == 1
    assert status['sample_count'] == 12
    assert status['completed_at'] == '2026-07-11T10:00:00+00:00'
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert 'status.json' in manifest['artifacts']
