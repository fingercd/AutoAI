"""Review-only deterministic concurrent lost-bind recovery reproducer."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from backend.tests.test_agent_model_sessions import api, session
from backend.app.agent.contracts import AgentDomainError
from backend.app.agent.contracts import CreateAgentExperimentRequestV2
from backend.app.runs.contracts import Principal


def test_concurrent_lost_bind_replay_returns_same_run(api, monkeypatch):
    from backend.app.routers.agent import _agent_service

    created = session(api)
    service = _agent_service('agent-session-v2')
    original_submit = service.submissions.submit

    class LostProcess(BaseException):
        pass

    def crash_after_run_creation(request):
        original_submit(request)
        raise LostProcess()

    payload = CreateAgentExperimentRequestV2(
        model_type='logistic_regression', client_request_id='concurrent-lost-bind')
    principal = Principal()
    monkeypatch.setattr(service.submissions, 'submit', crash_after_run_creation)
    with pytest.raises(LostProcess):
        service.create_experiment(session_id=created['session_id'], payload=payload,
                                  principal=principal)
    monkeypatch.setattr(service.submissions, 'submit', original_submit)
    assert len(service.runs.list()) == 1
    assert service.sessions.list_experiments_scoped(
        created['session_id'], principal=principal)[0].state == 'reserved'

    barrier = Barrier(2)
    original_bind = service.sessions.bind_experiment

    def simultaneous_bind(*args, **kwargs):
        # Both requests have read the SAME durable reserved+mapped state.
        barrier.wait(timeout=15)
        return original_bind(*args, **kwargs)

    monkeypatch.setattr(service.sessions, 'bind_experiment', simultaneous_bind)

    def replay(_):
        try:
            return service.create_experiment(session_id=created['session_id'],
                                              payload=payload, principal=principal)
        except AgentDomainError as error:
            return {'unexpected_error': error.code, 'http_status': error.status_code}

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(replay, range(2)))

    assert len(service.runs.list()) == 1
    assert all('unexpected_error' not in item for item in outcomes), outcomes
    assert outcomes[0]['run_id'] == outcomes[1]['run_id']
    assert all(item['idempotent_replay'] for item in outcomes)
