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
