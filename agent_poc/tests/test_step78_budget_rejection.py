"""R-S78-01: confirmed unaffordable proposals close, unknown effects stay held."""
import sqlite3
import time

import httpx
import pytest

from backend.tests.test_agent_model_sessions import api
from backend.app.agent.budget import BudgetPolicy
from backend.app.runs.repository import RunRepository
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.graph import Nodes, RECIPE_BUDGET_REJECTION
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig
from agent_poc.orchestration.runtime import (
    RuntimeConfig, start_task, resume_task, TaskInterrupted,
    _default_budget_policy,
)
from agent_poc.tests.test_review_recovery import CountTransport


class UnaffordableProvider:
    """Controlled response, not a real model quality experiment."""
    def __init__(self, protocol='json_action'):
        self.protocol = protocol
        self.contexts = []

    def request(self, method, url, *, headers, json: dict, timeout):
        import json as codec
        context = codec.loads(json['messages'][-1]['content'])
        self.contexts.append(context)
        recipe = next(r for r in context['recipes'] if r['model_id'] == 'resnet1d')
        args = {**context['bindings'], 'recipe_id': recipe['recipe_id'], 'knowledge_refs': []}
        action = dict(tool_name='submit_ml_experiment', arguments=args,
                      rationale='Choose the frozen four epoch recipe.')
        message = (dict(role='assistant', content=codec.dumps(action))
                   if self.protocol == 'json_action' else
                   dict(role='assistant', content=action['rationale'], tool_calls=[
                       dict(id='unaffordable-choice', type='function', function=dict(
                           name=action['tool_name'], arguments=codec.dumps(args)))]))
        return httpx.Response(200, json=dict(id='confirmed-unaffordable-response',
            choices=[dict(message=message, finish_reason='stop')],
            usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30)))


def setup_task(api, tmp_path, budget_config, *, version=8, awareness='off',
               guard='on', protocol='json_action'):
    test_client, storage, dataset = api
    config = LLMConfig('http://scripted.invalid/v1', 'fixture',
        prompt_version='agent-decision-budget-v1', protocol=protocol, **budget_config)
    runtime = RuntimeConfig('http://backend.invalid', 'local', config)
    wire = CountTransport(test_client)
    revision = f'agent-recipes-revision-v{version - 2}'
    client = AutoAIClient(runtime.backend_url, transport=wire, api_version='v2',
        execution_profile='train-evidence-recipes-v1', protocol_revision=revision,
        processing_mode='fixed', search_mode='fixed', max_trials=1, max_retries=0)
    provider = UnaffordableProvider(protocol)
    llm = LLMAdapter(config, transport=provider)
    models = ['cnn1d', 'resnet1d']
    configs = {'cnn1d': {'epochs': 2}, 'resnet1d': {'epochs': 4}}
    standard = _default_budget_policy(task_id='reject-budget', started_at=time.time(),
        timeout_seconds=3600, allowed_models=models, model_configs=configs, max_trials=1,
        max_llm_calls=6, max_api_calls=60, max_repair_attempts=2,
        max_operation_attempts=3, max_output_tokens=config.max_tokens)
    policy = BudgetPolicy(task_id=standard.task_id, started_at=standard.started_at,
        deadline_at=standard.deadline_at, work_deadline_at=standard.work_deadline_at,
        limits={**standard.limits, 'training_epochs': 2},
        max_output_tokens_per_call=standard.max_output_tokens_per_call)
    kwargs = dict(dataset_id=dataset, allowed_models=models, model_configs=configs,
        processing_mode='fixed', search_mode='fixed', max_trials=1,
        decision_mode='recipe_id', knowledge=False, budget_awareness=awareness,
        budget_policy=policy.as_dict(), storage=tmp_path / 'graph', thread_id='rejection',
        client=client, llm=llm, wait=False)
    if version == 8:
        kwargs['fail_fast_guard'] = guard
    return runtime, kwargs, wire, provider


def resume(runtime, kwargs):
    return resume_task(runtime, **{k: kwargs[k] for k in
        ('storage', 'thread_id', 'client', 'llm', 'wait')})


@pytest.mark.parametrize('version,guard', [(7, 'on'), (8, 'on'), (8, 'off')])
@pytest.mark.parametrize('protocol', ['json_action', 'native_tools'])
@pytest.mark.parametrize('crash', [False, True])
def test_confirmed_rejection_terminates_and_replays_without_spending(
        api, tmp_path, budget_config, monkeypatch, version, guard, protocol, crash):
    runtime, kwargs, wire, provider = setup_task(api, tmp_path, budget_config,
        version=version, guard=guard, protocol=protocol)
    if crash:
        original = Nodes.choose
        def interrupt_after_response(self, state):
            result = original(self, state)
            assert result['decision']['reason_code'] == RECIPE_BUDGET_REJECTION
            raise KeyboardInterrupt
        with monkeypatch.context() as patch:
            patch.setattr(Nodes, 'choose', interrupt_after_response)
            with pytest.raises(TaskInterrupted):
                start_task(runtime, **kwargs)
        assert len(provider.contexts) == 1
        state = resume(runtime, kwargs)
    else:
        state = start_task(runtime, **kwargs)
    assert state['lifecycle']['status'] == 'completed', state['lifecycle']
    assert state['lifecycle']['reason_code'] == 'budget_exhausted'
    assert state['decision']['status'] == 'invalid'
    assert state['decision']['reason_code'] == RECIPE_BUDGET_REJECTION
    assert state['decision']['response_id'] == 'confirmed-unaffordable-response'
    assert state['execution']['submission_content'] is None
    assert state['execution']['run_id'] is None
    assert state['finalization']['backend_session_state'] == 'terminated'
    assert state['finalization']['selected_run_id'] is None
    assert not state['recovery']['needs_human_review']
    assert len(provider.contexts) == 1
    assert 'budget_card' not in provider.contexts[0]
    assert not RunRepository(api[1] / 'runs.sqlite3').list()
    assert not any(path.endswith('/experiments') for _, path, _ in wire.calls)
    assert state['budget']['llm_calls']['actual'] == 1
    assert state['budget']['input_tokens']['actual'] == 20
    assert state['budget']['output_tokens']['actual'] == 10
    with sqlite3.connect(kwargs['storage'] / 'calls.sqlite') as db:
        row = db.execute("SELECT status,proposal_json FROM orchestration_calls_v1 WHERE kind='llm'").fetchone()
        assert row[0] == 'confirmed'
        assert state['decision']['recipe_id'] in row[1]
    calls_before = len(wire.requests)
    assert resume(runtime, kwargs) == state
    assert len(wire.requests) == calls_before
    assert len(provider.contexts) == 1


@pytest.mark.parametrize('version,guard', [(7, 'on'), (8, 'on'), (8, 'off')])
@pytest.mark.parametrize('protocol', ['json_action', 'native_tools'])
def test_awareness_on_rejects_unaffordable_output_at_protocol_boundary(
        api, tmp_path, budget_config, version, guard, protocol):
    runtime, kwargs, wire, provider = setup_task(api, tmp_path, budget_config,
        version=version, guard=guard, awareness='on', protocol=protocol)
    state = start_task(runtime, **kwargs)
    # On mode's schema only permits affordable recipes: this remains a protocol
    # error, not a confirmed valid proposal at the graph affordability boundary.
    assert state['lifecycle']['status'] == 'waiting'
    assert state['lifecycle']['reason_code'] == 'llm_output_invalid'
    assert state['decision']['reason_code'] != RECIPE_BUDGET_REJECTION
    assert 'budget_card' in provider.contexts[0]
    assert state['budget']['llm_calls']['actual'] == 1
    assert state['budget']['input_tokens']['actual'] == 20
    assert not any(path.endswith(('/terminate', '/experiments')) for _, path, _ in wire.calls)


@pytest.mark.parametrize('failure', ['active_run', 'mapping_unknown', 'api_budget', 'llm_unknown'])
def test_uncertainty_or_insufficient_close_budget_never_claims_termination(
        api, tmp_path, budget_config, monkeypatch, failure):
    from agent_poc.orchestration.persistence import PersistenceError
    runtime, kwargs, wire, provider = setup_task(api, tmp_path, budget_config)
    original_call = Nodes.call
    original_choose = Nodes.choose
    def call(self, state, name, arguments, **extra):
        if state['lifecycle']['stage'] == 'terminate' and name == 'inspect_ml_session':
            if failure == 'api_budget':
                raise PersistenceError('budget_exhausted_api_calls')
            response = original_call(self, state, name, arguments, **extra)
            if failure in ('active_run', 'mapping_unknown'):
                response['experiments'] = [dict(binding_state=(
                    'bound' if failure == 'active_run' else 'compensation_required'),
                    state='running' if failure == 'active_run' else None)]
            return response
        if name == 'reconcile_ml_session' and failure == 'mapping_unknown':
            return dict(items=[dict(state='compensation_required',
                requires_manual_review=True, resolution_code='mapping_unknown')])
        return original_call(self, state, name, arguments, **extra)
    monkeypatch.setattr(Nodes, 'call', call)
    if failure == 'llm_unknown':
        # Dispatch happened but its result could not be recorded, so no known proposal.
        def unknown(self, state):
            def lose_response(*args, **kwargs):
                raise PersistenceError('response_outcome_unknown')
            self.deps.llm._transport.request = lose_response
            return original_choose(self, state)
        monkeypatch.setattr(Nodes, 'choose', unknown)
    state = start_task(runtime, **kwargs)
    if failure == 'llm_unknown':
        assert state['lifecycle']['status'] == 'waiting', state['lifecycle']
        assert state['lifecycle']['reason_code'] == 'llm_unavailable'
        assert state['recovery']['pending_operation']['status'] == 'unknown'
        assert state['budget']['input_tokens']['unknown_pending'] == 1
    else:
        assert state['lifecycle']['status'] == 'needs_attention', state['lifecycle']
        assert state['recovery']['needs_human_review']
    assert state['finalization']['backend_session_state'] != 'terminated'
    assert not any(path.endswith('/terminate') for _, path, _ in wire.calls)
    assert not RunRepository(api[1] / 'runs.sqlite3').list()
    session = kwargs['client'].inspect_ml_session(state['identity']['session_id'])
    assert session['state'] == 'open'
    if failure != 'llm_unknown':
        assert state['decision']['reason_code'] == RECIPE_BUDGET_REJECTION
        assert state['budget']['llm_calls']['actual'] == 1
    else:
        assert state['decision']['reason_code'] != RECIPE_BUDGET_REJECTION


@pytest.mark.parametrize('resolution', ['released', 'empty'])
def test_reconciled_rejection_keeps_original_decision_and_confirms_choose(
        api, tmp_path, budget_config, monkeypatch, resolution):
    runtime, kwargs, wire, provider = setup_task(api, tmp_path, budget_config)
    original_call, original_execute = Nodes.call, Nodes.execute
    inspected = []
    choose_operations = []
    def call(self, state, name, arguments, **extra):
        if state['lifecycle']['stage'] == 'terminate' and name == 'inspect_ml_session':
            response = original_call(self, state, name, arguments, **extra)
            if not inspected:
                response['experiments'] = [dict(binding_state='reserved', state=None)]
            inspected.append(True)
            return response
        if name == 'reconcile_ml_session':
            return dict(items=[] if resolution == 'empty' else [dict(
                state='released', requires_manual_review=False,
                resolution_code='verified_no_submission')])
        return original_call(self, state, name, arguments, **extra)
    def execute(self, action, state):
        result = original_execute(self, action, state)
        if action == 'choose':
            choose_operations.append(result['recovery']['pending_operation'])
        return result
    monkeypatch.setattr(Nodes, 'call', call)
    monkeypatch.setattr(Nodes, 'execute', execute)
    state = start_task(runtime, **kwargs)
    assert state['finalization']['backend_session_state'] == 'terminated', state['lifecycle']
    assert len(choose_operations) == 1
    assert choose_operations[0]['status'] == 'confirmed'
    assert state['decision']['reason_code'] == RECIPE_BUDGET_REJECTION
    assert len(provider.contexts) == 1
    assert not any(path.endswith('/experiments') for _, path, _ in wire.calls)
