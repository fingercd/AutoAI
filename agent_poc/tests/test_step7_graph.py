"""The v7 Graph creates a real v5 Session with the exact frozen policy."""

import pytest

from backend.tests.test_agent_model_sessions import api
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig
from agent_poc.orchestration.runtime import RuntimeConfig, start_task
from agent_poc.tests.test_knowledge_graph import KnowledgeProvider
from agent_poc.tests.test_review_recovery import CountTransport


def test_v7_graph_creates_budgeted_session(api, tmp_path, budget_config):
    test_client, _, dataset = api
    llm_config = LLMConfig('http://scripted.invalid/v1', 'fixture',
        prompt_version='agent-decision-budget-v1', **budget_config)
    runtime = RuntimeConfig('http://backend.invalid', 'local', llm_config)
    transport = CountTransport(test_client)
    client = AutoAIClient(runtime.backend_url, transport=transport,
        api_version='v2', execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v5', processing_mode='fixed',
        search_mode='fixed', max_trials=1, max_retries=0)
    llm = LLMAdapter(llm_config, transport=KnowledgeProvider('json_action'))
    state = start_task(runtime, dataset_id=dataset,
        allowed_models=['logistic_regression'], processing_mode='fixed',
        search_mode='fixed', max_trials=1, decision_mode='recipe_id',
        knowledge=False, budget_awareness='off', storage=tmp_path / 'graph',
        thread_id='v7-graph', client=client, llm=llm, wait=False)
    assert state['versions']['state'] == 'agent-state-v7'
    assert state['identity']['session_id']
    request = next(body for method, path, body in transport.calls
                   if method == 'POST' and path == '/api/agent/v2/sessions')
    assert request['budget_awareness'] == 'off'
    assert request['budget_policy'] == state['task']['budget_policy']
    assert 'budget_card' not in llm._transport.contexts[0]
    if state['decision']['status'] != 'ready':
        from agent_poc.orchestration.graph import Nodes
        from agent_poc.orchestration.projection import recipe_selection_context, validate_context
        preparation = Nodes.preparation(state)
        context = recipe_selection_context(task=state['task'], session_id=state['identity']['session_id'],
            preparation=preparation, context_policy=state['module_policy']['context_policy'],
            model_configs=state['capabilities']['frozen_snapshot']['model_configs'])
        validate_context('submit', context)
    assert state['decision']['status'] == 'ready', state['lifecycle']


def test_v7_zero_fit_budget_terminates_without_training(api, tmp_path, budget_config):
    import time
    from backend.app.agent.budget import BudgetPolicy
    from agent_poc.orchestration.runtime import _default_budget_policy

    test_client, _, dataset = api
    llm_config = LLMConfig('http://scripted.invalid/v1', 'fixture',
        prompt_version='agent-decision-budget-v1', **budget_config)
    runtime = RuntimeConfig('http://backend.invalid', 'local', llm_config)
    transport = CountTransport(test_client)
    client = AutoAIClient(runtime.backend_url, transport=transport,
        api_version='v2', execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v5', processing_mode='fixed',
        search_mode='fixed', max_trials=1, max_retries=0)
    provider = KnowledgeProvider('json_action')
    llm = LLMAdapter(llm_config, transport=provider)
    standard = _default_budget_policy(task_id='zero-fit-task', started_at=time.time(),
        timeout_seconds=3600, allowed_models=['logistic_regression'],
        model_configs=None, max_trials=1, max_llm_calls=6,
        max_api_calls=60, max_repair_attempts=2,
        max_operation_attempts=3, max_output_tokens=llm_config.max_tokens)
    policy = BudgetPolicy(task_id=standard.task_id, started_at=standard.started_at,
        deadline_at=standard.deadline_at, work_deadline_at=standard.work_deadline_at,
        limits={**standard.limits, 'model_fits':0},
        max_output_tokens_per_call=standard.max_output_tokens_per_call)
    state = start_task(runtime, dataset_id=dataset,
        allowed_models=['logistic_regression'], processing_mode='fixed',
        search_mode='fixed', max_trials=1, decision_mode='recipe_id',
        knowledge=False, budget_awareness='off', budget_policy=policy.as_dict(),
        storage=tmp_path / 'zero-fit', thread_id='v7-zero-fit', client=client,
        llm=llm, wait=False)
    assert state['lifecycle']['status'] == 'completed', state['lifecycle']
    assert state['lifecycle']['reason_code'] == 'budget_exhausted'
    assert state['execution']['run_id'] is None
    assert provider.contexts == []
    assert any(path.endswith('/terminate') for _, path, _ in transport.calls)


@pytest.mark.parametrize('protocol',['json_action','native_tools'])
def test_v7_budget_awareness_can_stop_before_training(api, tmp_path, budget_config, protocol):
    import httpx
    test_client, _, dataset = api
    llm_config = LLMConfig('http://scripted.invalid/v1', 'fixture',
        prompt_version='agent-decision-budget-v1', protocol=protocol, **budget_config)
    runtime = RuntimeConfig('http://backend.invalid', 'local', llm_config)
    transport = CountTransport(test_client)
    client = AutoAIClient(runtime.backend_url, transport=transport,
        api_version='v2', execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v5', processing_mode='fixed',
        search_mode='fixed', max_trials=1, max_retries=0)
    class StopProvider(KnowledgeProvider):
        def request(self, method, url, *, headers, json: dict, timeout):
            context = __import__('json').loads(json['messages'][-1]['content'])
            self.contexts.append(context)
            assert context['budget_card']['recipes'][0]['feasible']
            assert context['allowed_actions'] == ['submit_ml_experiment','stop_ml_session']
            action = {'tool_name':'stop_ml_session',
                      'arguments':{'session_id':context['bindings']['session_id']},
                      'rationale':'Stop before spending the remaining training budget.'}
            message = ({'role':'assistant','content':__import__('json').dumps(action)}
                       if protocol=='json_action' else
                       {'role':'assistant','content':action['rationale'],
                        'tool_calls':[{'id':'stop-choice','type':'function','function':{
                            'name':'stop_ml_session',
                            'arguments':__import__('json').dumps(action['arguments'])}}]})
            return httpx.Response(200, json={'choices':[{'message':message,'finish_reason':'stop'}],
                'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}})
    provider = StopProvider(protocol)
    state = start_task(runtime, dataset_id=dataset,
        allowed_models=['logistic_regression'], processing_mode='fixed',
        search_mode='fixed', max_trials=1, decision_mode='recipe_id',
        knowledge=False, budget_awareness='on', storage=tmp_path / 'stop',
        thread_id='v7-stop', client=client, llm=LLMAdapter(llm_config, transport=provider),
        wait=False)
    assert state['lifecycle']['status'] == 'completed', state['lifecycle']
    assert state['decision']['kind'] == 'budget_stop'
    assert state['execution']['run_id'] is None
    assert state['finalization']['backend_session_state'] == 'terminated'
    assert any(path.endswith('/terminate') for _, path, _ in transport.calls)


def test_v7_replays_saved_proposal_without_new_llm_call(api, tmp_path, budget_config, monkeypatch):
    from agent_poc.orchestration.graph import Nodes
    from agent_poc.orchestration.runtime import resume_task, TaskInterrupted
    test_client, _, dataset = api
    llm_config = LLMConfig('http://scripted.invalid/v1', 'fixture',
        prompt_version='agent-decision-budget-v1', **budget_config)
    runtime = RuntimeConfig('http://backend.invalid', 'local', llm_config)
    transport = CountTransport(test_client)
    client = AutoAIClient(runtime.backend_url, transport=transport,
        api_version='v2', execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v5', processing_mode='fixed',
        search_mode='fixed', max_trials=1, max_retries=0)
    provider = KnowledgeProvider('json_action')
    llm = LLMAdapter(llm_config, transport=provider)
    original = Nodes.choose
    def interrupted(self, state):
        original(self, state)
        raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr(Nodes, 'choose', interrupted)
        with pytest.raises(TaskInterrupted):
            start_task(runtime, dataset_id=dataset,
                allowed_models=['logistic_regression'], processing_mode='fixed',
                search_mode='fixed', max_trials=1, decision_mode='recipe_id',
                knowledge=False, budget_awareness='on', storage=tmp_path / 'replay',
                thread_id='v7-replay', client=client, llm=llm, wait=False)
    assert len(provider.contexts) == 1
    state = resume_task(runtime, storage=tmp_path / 'replay',
        thread_id='v7-replay', client=client, llm=llm, wait=False)
    assert state['decision']['status'] == 'ready', state['lifecycle']
    assert len(provider.contexts) == 1


def test_v7_work_deadline_terminates_before_new_decision(api, tmp_path,
                                                          budget_config, monkeypatch):
    import time
    from agent_poc.orchestration.graph import Nodes
    from agent_poc.orchestration.runtime import resume_task, read_status, TaskInterrupted
    test_client, _, dataset = api
    llm_config = LLMConfig('http://scripted.invalid/v1', 'fixture',
        prompt_version='agent-decision-budget-v1', **budget_config)
    runtime = RuntimeConfig('http://backend.invalid','local',llm_config)
    transport = CountTransport(test_client)
    client = AutoAIClient(runtime.backend_url, transport=transport,
        api_version='v2', execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v5', processing_mode='fixed',
        search_mode='fixed', max_trials=1, max_retries=0)
    provider = KnowledgeProvider('json_action')
    llm = LLMAdapter(llm_config, transport=provider)
    now = [time.time()]
    checkpoint = tmp_path / 'deadline'
    with monkeypatch.context() as patch:
        def interrupt_before_choice(self, state):
            raise KeyboardInterrupt
        patch.setattr(Nodes,'choose',interrupt_before_choice)
        with pytest.raises(TaskInterrupted):
            start_task(runtime, dataset_id=dataset,
                allowed_models=['logistic_regression'], processing_mode='fixed',
                search_mode='fixed', max_trials=1, decision_mode='recipe_id',
                knowledge=False, budget_awareness='on', storage=checkpoint,
                thread_id='v7-deadline', client=client, llm=llm,
                clock=lambda:now[0], wait=False)
    before = read_status(storage=checkpoint, thread_id='v7-deadline')
    now[0] = before['task']['budget_policy']['work_deadline_at'] + 1
    state = resume_task(runtime, storage=checkpoint, thread_id='v7-deadline',
        client=client, llm=llm, clock=lambda:now[0], wait=False)
    assert state['lifecycle']['status']=='completed', state['lifecycle']
    assert state['lifecycle']['reason_code']=='deadline'
    assert provider.contexts == []


@pytest.mark.parametrize('awareness,final_action',
    [('on','finalize'),('off','finalize'),('on','stop')])
def test_v7_training_and_finalization_closes_session(api, tmp_path, budget_config,
                                                     awareness, final_action):
    import httpx
    from datetime import datetime, timezone
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.worker import RunWorker, execute_with_budget_supervision
    from backend.app.runs.status_projection import project_status
    from agent_poc.orchestration.runtime import resume_task

    test_client, storage, dataset = api
    llm_config = LLMConfig('http://scripted.invalid/v1', 'fixture',
        prompt_version='agent-decision-budget-v1', **budget_config)
    runtime = RuntimeConfig('http://backend.invalid','local',llm_config)
    transport = CountTransport(test_client)
    client = AutoAIClient(runtime.backend_url, transport=transport,
        api_version='v2', execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v5', processing_mode='fixed',
        search_mode='fixed', max_trials=1, max_retries=0)
    class FinalProvider(KnowledgeProvider):
        def request(self, method, url, *, headers, json: dict, timeout):
            context = __import__('json').loads(json['messages'][-1]['content'])
            if context['phase']=='finalize' and final_action=='stop':
                assert context['allowed_actions']==['finalize_ml_session','stop_ml_session']
                self.contexts.append(context)
                action = {'tool_name':'stop_ml_session',
                    'arguments':{'session_id':context['bindings']['session_id']},
                    'rationale':'Stop after observing the valid result.'}
                return httpx.Response(200,json={'choices':[{'message':{'role':'assistant',
                    'content':__import__('json').dumps(action)},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}})
            return super().request(method,url,headers=headers,json=json,timeout=timeout)
    provider = FinalProvider('json_action')
    llm = LLMAdapter(llm_config, transport=provider)
    checkpoint = tmp_path / f'training-{awareness}-{final_action}'
    first = start_task(runtime, dataset_id=dataset,
        allowed_models=['logistic_regression'], processing_mode='fixed',
        search_mode='fixed', max_trials=1, decision_mode='recipe_id',
        knowledge=False, budget_awareness=awareness, storage=checkpoint,
        thread_id=f'v7-training-{awareness}-{final_action}', client=client, llm=llm, wait=False)
    assert first['lifecycle']['status'] == 'waiting', first['lifecycle']
    repo = RunRepository(storage / 'runs.sqlite3')
    worker = RunWorker(repository=repo, worker_id='v7-training',
        execute=lambda run: execute_with_budget_supervision(run, repository=repo,
            agent_database=storage / 'agent.sqlite3'),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60,
        project_status=lambda run:project_status(storage / 'runs' / run.run_id, run))
    assert worker.run_once()
    assert repo.get(first['execution']['run_id']).state == 'succeeded'
    final = resume_task(runtime, storage=checkpoint, thread_id=f'v7-training-{awareness}-{final_action}',
        client=client, llm=llm, wait=True)
    assert final['lifecycle']['status'] == 'completed', final['lifecycle']
    assert final['finalization']['backend_session_state'] == (
        'terminated' if final_action=='stop' else 'finalized')
    assert len(provider.contexts) == 2
    assert ('budget_card' in provider.contexts[0]) == (awareness == 'on')
    from scripts.task_cost_report import build_report
    summary = __import__('json').loads(build_report(
        journal=checkpoint / 'calls.sqlite', agent_db=storage / 'agent.sqlite3',
        runs_db=storage / 'runs.sqlite3',
        run_dir=storage / 'runs' / first['execution']['run_id'],
        run_id=first['execution']['run_id'],
        thread_id=f'v7-training-{awareness}-{final_action}',
        session_id=first['identity']['session_id'],
        task_id=first['identity']['task_id'],owner_id=None,tenant_id=None,
    )['cost_summary.json'])
    assert summary['schema_version'] == 'task-cost-report-v1'
    assert summary['report_status'] == 'settled', summary['budget_dimensions']
    assert summary['budget_dimensions']['model_fits']['known_actual'] >= 1
