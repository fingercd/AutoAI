from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_poc.clients.llm_client import LLMClient
from agent_poc.budget import BudgetController, BudgetExceeded
from agent_poc.schemas import RequestHumanDecision, RunExperimentDecision
from agent_poc.state import AgentBudgetConfig, ModelConfig


class FakeOpenAI:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.contents)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _model():
    return ModelConfig(
        key='qwen35_9b',
        base_url='http://127.0.0.1:8101/v1',
        served_model_name='qwen35_9b',
        model_path='/server-only/model',
    )


def test_llm_client_repairs_once_then_validates_with_pydantic():
    fake = FakeOpenAI([
        'not-json',
        '{"decision":"RUN_EXPERIMENT","model_type":"svm","rationale":"ok"}',
    ])
    client = LLMClient(_model(), client=fake)
    decision = client.decide([{'role': 'user', 'content': '{}'}])
    assert isinstance(decision, RunExperimentDecision)
    assert client.call_count == 2
    assert client.repair_count == 1
    assert len(fake.calls) == 2


def test_second_invalid_json_becomes_request_human_without_submission():
    fake = FakeOpenAI(['{}', '{"decision":"RUN_EXPERIMENT","extra":true}'])
    client = LLMClient(_model(), client=fake)
    decision = client.decide([{'role': 'user', 'content': '{}'}])
    assert isinstance(decision, RequestHumanDecision)
    assert client.call_count == 2
    assert client.repair_count == 1


def test_structured_repair_debits_llm_and_shared_retry_budget():
    fake = FakeOpenAI([
        'not-json',
        '{"decision":"RUN_EXPERIMENT","model_type":"svm","rationale":"ok"}',
    ])
    controller = BudgetController(AgentBudgetConfig(
        max_model_fits=10,
        max_llm_calls=2,
        max_api_calls=10,
        max_wall_clock_seconds=30,
        max_retry_attempts=1,
    ))
    client = LLMClient(_model(), client=fake, budget_controller=controller)
    assert isinstance(
        client.decide([{'role': 'user', 'content': '{}'}]),
        RunExperimentDecision,
    )
    assert controller.safe_usage()['llm_call_count'] == 2
    assert controller.safe_usage()['retry_attempt_count'] == 1


def test_exhausted_repair_budget_prevents_second_completion():
    fake = FakeOpenAI(['not-json', 'must-not-be-consumed'])
    controller = BudgetController(AgentBudgetConfig(
        max_model_fits=10,
        max_llm_calls=2,
        max_api_calls=10,
        max_wall_clock_seconds=30,
        max_retry_attempts=0,
    ))
    client = LLMClient(_model(), client=fake, budget_controller=controller)
    with pytest.raises(BudgetExceeded) as result:
        client.decide([{'role': 'user', 'content': '{}'}])
    assert result.value.dimension == 'max_retry_attempts'
    assert len(fake.calls) == 1
    assert controller.safe_usage()['llm_call_count'] == 1
