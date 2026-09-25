"""No-Run rejection is distinct from a lost response or an unknown mapping."""
from dataclasses import replace
import time

import pytest

from backend.tests.test_agent_model_sessions import api
from backend.tests.test_step8_guard_integration import HEADERS
from backend.app.runs.repository import RunRepository
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig
from agent_poc.orchestration.runtime import RuntimeConfig, start_task, resume_task, _default_budget_policy
from agent_poc.tests.test_knowledge_graph import KnowledgeProvider
from agent_poc.tests.test_review_recovery import CountTransport


@pytest.mark.parametrize('fault', ['lost_rejection', 'lost_created_run', 'unknown_mapping', 'close_budget'])
def test_admission_recovery_preserves_uncertainty_and_usage(api, tmp_path, budget_config, monkeypatch, fault):
    from backend.app import training
    from backend.app.runs.guard import GuardError
    wire, storage, dataset = api
    if fault in ('lost_rejection', 'close_budget'):
        def reject(*args, **kwargs):
            raise GuardError('guard_resource_limit', stage='admission')
        monkeypatch.setattr(training, 'prepare_training_inputs', reject)

    class FaultTransport(CountTransport):
        lost = False
        def request(self, method, url, **kwargs):
            response = super().request(method, url, **kwargs)
            if method == 'POST' and url.endswith('/experiments') and not self.lost:
                self.lost = True
                if fault != 'close_budget':
                    raise TimeoutError('response lost')
            if fault == 'unknown_mapping' and self.lost and method == 'GET' and '/sessions/' in url:
                raise TimeoutError('mapping cannot be inspected')
            return response

    config = LLMConfig('http://scripted.invalid/v1', 'fixture',
        prompt_version='agent-decision-budget-v1', **budget_config)
    runtime = RuntimeConfig('http://backend.invalid', 'local', config)
    transport = FaultTransport(wire)
    client = AutoAIClient(runtime.backend_url, transport=transport, api_version='v2',
        execution_profile='train-evidence-recipes-v1', protocol_revision='agent-recipes-revision-v6',
        processing_mode='fixed', search_mode='fixed', max_trials=1, max_retries=0)
    provider = KnowledgeProvider('json_action')
    llm = LLMAdapter(config, transport=provider)
    ticks = [time.time()]
    extra = {}
    if fault == 'close_budget':
        policy = _default_budget_policy(task_id='limited-close', started_at=ticks[0],
            timeout_seconds=3600, allowed_models=['logistic_regression'], model_configs={},
            max_trials=1, max_llm_calls=6, max_api_calls=60, max_repair_attempts=2,
            max_operation_attempts=3, max_output_tokens=1024)
        policy = replace(policy, finalization_api_calls=0, limits={**policy.limits, 'api_calls':5})
        extra = dict(task_id='limited-close', max_api_calls=5, budget_policy=policy.as_dict())
    state = start_task(runtime, dataset_id=dataset, allowed_models=['logistic_regression'],
        processing_mode='fixed', search_mode='fixed', max_trials=1, decision_mode='recipe_id',
        knowledge=False, budget_awareness='off', fail_fast_guard='on', storage=tmp_path/'graph',
        thread_id='recovery', client=client, llm=llm, wait=False, clock=lambda:ticks[0], **extra)
    if fault == 'close_budget':
        assert state['lifecycle']['status'] == 'needs_attention', state['lifecycle']
        assert state['guard']['reports'][0]['checks'][0]['reason_code'] == 'guard_resource_limit'
        assert len(transport.requests) == 5
        assert state['budget']['api_calls']['actual'] == 5
    else:
        assert state['lifecycle']['next_action'] == 'inspect_session', state['lifecycle']
        assert state['recovery']['pending_operation']['status'] == 'unknown'
        assert not state['guard']['reports']  # A lost response is not a trusted rejection.
        if fault in ('lost_created_run', 'unknown_mapping'):
            ticks[0] += 3
            state = resume_task(runtime, storage=tmp_path/'graph', thread_id='recovery',
                                client=client, llm=llm, clock=lambda:ticks[0], wait=False)
            assert state['lifecycle']['next_action'] == ('observe' if fault == 'lost_created_run' else 'inspect_session')
            assert sum(method == 'POST' and url.endswith('/experiments') for method,url in transport.requests) == 1
        assert not any(url.endswith('/terminate') for _,url in transport.requests)
    backend = wire.get('/api/agent/v2/sessions/'+state['identity']['session_id'], headers=HEADERS).json()
    assert backend['state'] == 'open'
    assert len(provider.contexts) == 1 and state['budget']['llm_calls']['actual'] == 1
    assert state['budget']['api_calls']['actual'] == len(transport.requests)
    assert len(RunRepository(storage/'runs.sqlite3').list()) == (1 if fault in ('lost_created_run','unknown_mapping') else 0)
