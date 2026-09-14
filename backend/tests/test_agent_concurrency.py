from concurrent.futures import ThreadPoolExecutor

from backend.app.agent.contracts import AgentDomainError
from backend.app.agent.contracts import CreateAgentExperimentRequest, CreateAgentSessionRequest
from backend.app.agent.repository import AgentSessionRepository
from backend.app.agent.service import AgentService
from backend.app.datasets.repository import DatasetRepository
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunRepository
from backend.app.runs.status_projection import project_status
from backend.app.runs.submission import RunSubmissionError, RunSubmissionService


def test_twenty_concurrent_reservations_cannot_exceed_max_runs_one(tmp_path):
    repo = AgentSessionRepository(tmp_path / 'agent.sqlite3'); repo.initialize()
    session, _ = repo.create_session(
        dataset_id='ds-1', selection_metric='macro_f1', allowed_models=['logistic_regression'],
        max_runs=1, seed=42, evaluation_config={}, modules=[], context_policy={},
        client_request_id=None, payload_hash='session', principal=Principal(),
    )

    def reserve(index):
        try:
            item, _ = repo.reserve_experiment(
                session_id=session.session_id, action_json={'candidate': index}, rationale=None,
                parent_run_id=None, config_hash=f'hash-{index}', client_request_id=f'req-{index}',
                payload_hash=f'payload-{index}', active_run_ids=set(), principal=Principal(),
            )
            return item.state
        except AgentDomainError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(reserve, range(20)))
    assert results.count('reserved') == 1
    assert all(result in {'reserved', 'agent_active_run_exists', 'agent_run_budget_exhausted'} for result in results)
    assert repo.count_budget_scoped(session_id=session.session_id, principal=Principal()) == 1


class _FailedSubmission:
    def submit(self, _request):
        raise RunSubmissionError('worker_contract_mismatch', 'worker unavailable', status_code=503)


def _service(tmp_path, submission=None):
    storage = tmp_path / 'storage'; uploads = storage / 'uploads'; uploads.mkdir(parents=True)
    data = uploads / 'data.csv'; data.write_text('Index,Label,Sample_ID,0,1\n1,A,A1,1,2\n2,A,A2,2,3\n3,A,A3,3,4\n4,B,B1,4,5\n5,B,B2,5,6\n6,B,B3,6,7\n')
    datasets = DatasetRepository(storage / 'datasets.sqlite3', storage_root=storage); datasets.initialize()
    dataset = datasets.register(data, original_name='data.csv', principal=Principal())
    sessions = AgentSessionRepository(storage / 'agent.sqlite3'); sessions.initialize()
    runs = RunRepository(storage / 'runs.sqlite3'); runs.initialize()
    session, _ = sessions.create_session(
        dataset_id=dataset.dataset_id, selection_metric='macro_f1',
        allowed_models=['logistic_regression'], max_runs=1, seed=42,
        evaluation_config={'split_mode': 'stratified_holdout', 'split_train': 8,
                           'split_valid': 1, 'split_test': 1}, modules=[], context_policy={},
        client_request_id=None, payload_hash='session', principal=Principal(),
    )
    actual = RunSubmissionService(
        run_repository=runs, dataset_repository=datasets,
        run_dir=lambda run_id: storage / 'runs' / run_id, status_projector=project_status,
    )
    return AgentService(
        session_repository=sessions, run_repository=runs, dataset_repository=datasets,
        submission_service=submission or actual, run_root=storage / 'runs',
    ), sessions, runs, session


def test_run_creation_failure_releases_reservation(tmp_path):
    service, sessions, _runs, session = _service(tmp_path, _FailedSubmission())
    try:
        service.create_experiment(
            session_id=session.session_id,
            payload=CreateAgentExperimentRequest(model_type='logistic_regression'),
            principal=Principal(),
        )
    except AgentDomainError as exc:
        assert exc.code == 'worker_contract_mismatch'
    assert sessions.count_budget_scoped(session_id=session.session_id, principal=Principal()) == 0


def test_binding_failure_cancels_never_started_run_and_releases_budget(tmp_path, monkeypatch):
    service, sessions, runs, session = _service(tmp_path)
    monkeypatch.setattr(sessions, 'bind_experiment', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('bind')))
    try:
        service.create_experiment(
            session_id=session.session_id,
            payload=CreateAgentExperimentRequest(model_type='logistic_regression'),
            principal=Principal(),
        )
    except AgentDomainError as exc:
        assert exc.code == 'agent_experiment_binding_failed'
    assert runs.list()[0].state == 'cancelled'
    assert sessions.count_budget_scoped(session_id=session.session_id, principal=Principal()) == 0


def test_compensation_failure_keeps_budget_reserved(tmp_path, monkeypatch):
    from backend.app.runs.repository import InvalidRunTransition

    service, sessions, runs, session = _service(tmp_path)
    monkeypatch.setattr(sessions, 'bind_experiment', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('bind')))
    monkeypatch.setattr(
        runs,
        'cancel_queued_unstarted_scoped',
        lambda *args, **kwargs: (_ for _ in ()).throw(InvalidRunTransition('claimed')),
    )
    try:
        service.create_experiment(
            session_id=session.session_id,
            payload=CreateAgentExperimentRequest(model_type='logistic_regression'),
            principal=Principal(),
        )
    except AgentDomainError:
        pass
    reservations = sessions.list_experiments_scoped(session.session_id, principal=Principal())
    assert reservations[0].state == 'compensation_required'
    assert sessions.count_budget_scoped(session_id=session.session_id, principal=Principal()) == 1


def test_other_principal_cannot_create_session_from_dataset(tmp_path):
    service, _sessions, _runs, session = _service(tmp_path)
    payload = CreateAgentSessionRequest(
        dataset_id=session.dataset_id, selection_metric='macro_f1',
        allowed_models=['logistic_regression'], max_runs=1,
    )
    try:
        service.create_session(payload, principal=Principal('other', 'tenant'))
    except AgentDomainError as exc:
        assert exc.code == 'dataset_unavailable'
        assert exc.status_code == 404
    else:
        raise AssertionError('cross-principal dataset must be invisible')
