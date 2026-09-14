"""A retirement response ends a prepared task without reselecting a model."""
import pytest

from backend.tests.test_agent_v2 import api
from backend.app.agent.contracts import AgentDomainError
from agent_poc.tests.test_review_recovery import components
from agent_poc.orchestration.graph import Nodes
from agent_poc.orchestration.runtime import start_task, resume_task, TaskInterrupted


def test_retirement_during_prepared_resume_preserves_reason(api, tmp_path, monkeypatch):
    from backend.app.agent import policy_v2
    from backend.app.runs.repository import RunRepository
    wire, runtime, llm, client, dataset = components(api, 'cnn1d')
    original = Nodes.submit
    def interrupt(self, state):
        raise KeyboardInterrupt()
    monkeypatch.setattr(Nodes, 'submit', interrupt)
    with pytest.raises(TaskInterrupted):
        start_task(runtime, dataset_id=dataset, allowed_models=['cnn1d'],
            model_configs={'cnn1d':{'epochs':2}}, storage=tmp_path/'checkpoint',
            thread_id='retired', client=client(), llm=llm)
    monkeypatch.setattr(Nodes, 'submit', original)
    def retired(*args, **kwargs):
        raise AgentDomainError('model_retired', 'Model has been retired', status_code=409)
    monkeypatch.setattr(policy_v2, 'admit_action', retired)
    before = len(wire.requests)
    state = resume_task(runtime, storage=tmp_path/'checkpoint', thread_id='retired',
        client=client(), llm=llm)
    assert state['lifecycle']['status'] == 'needs_attention'
    assert 'model_retired' in str(state['lifecycle'])
    assert len(wire.requests) == before + 1 == state['budget']['api_calls']['actual']
    assert not RunRepository(api[1]/'runs.sqlite3').list()
    # No silent re-selection, retry or implicit inspection on the next resume.
    assert resume_task(runtime, storage=tmp_path/'checkpoint', thread_id='retired',
        client=client(), llm=llm) == state
    assert len(wire.requests) == before + 1
