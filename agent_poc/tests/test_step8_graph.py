from backend.tests.test_agent_model_sessions import api
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig
from agent_poc.orchestration.runtime import RuntimeConfig, start_task
from agent_poc.tests.test_knowledge_graph import KnowledgeProvider
from agent_poc.tests.test_review_recovery import CountTransport
import pytest


def test_v8_graph_freezes_guard_and_submits(api, tmp_path, budget_config, monkeypatch):
    test_client, _, dataset = api
    config = LLMConfig('http://scripted.invalid/v1','fixture', prompt_version='agent-decision-budget-v1', **budget_config)
    runtime = RuntimeConfig('http://backend.invalid','local',config)
    transport = CountTransport(test_client)
    client = AutoAIClient(runtime.backend_url, transport=transport, api_version='v2',
        execution_profile='train-evidence-recipes-v1', protocol_revision='agent-recipes-revision-v6',
        processing_mode='fixed', search_mode='fixed', max_trials=1, max_retries=0)
    llm = LLMAdapter(config, transport=KnowledgeProvider('json_action'))
    state = start_task(runtime, dataset_id=dataset, allowed_models=['logistic_regression'],
        processing_mode='fixed', search_mode='fixed', max_trials=1, decision_mode='recipe_id',
        knowledge=False, budget_awareness='off', fail_fast_guard='on', storage=tmp_path/'graph',
        thread_id='v8-graph', client=client, llm=llm, wait=False)
    assert state['versions']['state'] == 'agent-state-v8'
    assert state['identity']['session_id'], state['lifecycle']
    assert state['execution']['run_id'], state['lifecycle']
    assert state['module_policy']['fail_fast_guard']['enabled'] is True
    assert 'budget_card' not in llm._transport.contexts[0]
    from agent_poc.orchestration.state import apply_patch
    with pytest.raises(ValueError):
        apply_patch(state, {'module_policy':{'fail_fast_guard':{'enabled':False}}})
    import copy
    from agent_poc.orchestration.state import validate_state
    corrupt = copy.deepcopy(state)
    corrupt['task']['guard_policy']['rules_digest'] = '0' * 64
    with pytest.raises(ValueError):
        validate_state(corrupt)
    from backend.app.runs import guard
    from agent_poc.orchestration.runtime import read_status, resume_task, RuntimeErrorCode
    monkeypatch.setattr(guard, 'RULES_DIGEST', 'f' * 64)
    assert read_status(storage=tmp_path/'graph', thread_id='v8-graph') == state
    before = len(transport.requests)
    with pytest.raises(RuntimeErrorCode, match='guard_check_unavailable'):
        resume_task(runtime, storage=tmp_path/'graph', thread_id='v8-graph', client=client, llm=llm)
    assert len(transport.requests) == before


@pytest.mark.parametrize('outcome', ['success','damaged','prefit_failure'])
def test_v8_real_run_finalization_or_no_candidate_termination(api, tmp_path, budget_config, outcome, monkeypatch):
    from datetime import datetime, timezone
    from backend.app.datasets.repository import DatasetRepository
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.worker import RunWorker, execute_with_budget_supervision
    from agent_poc.orchestration.runtime import resume_task
    test_client, storage, dataset = api
    config = LLMConfig('http://scripted.invalid/v1','fixture', prompt_version='agent-decision-budget-v1', **budget_config)
    runtime = RuntimeConfig('http://backend.invalid','local',config)
    transport = CountTransport(test_client)
    client = AutoAIClient(runtime.backend_url, transport=transport, api_version='v2',
        execution_profile='train-evidence-recipes-v1', protocol_revision='agent-recipes-revision-v6',
        processing_mode='fixed', search_mode='fixed', max_trials=1, max_retries=0)
    provider = KnowledgeProvider('json_action')
    llm = LLMAdapter(config, transport=provider)
    root = tmp_path/'graph'
    state = start_task(runtime, dataset_id=dataset, allowed_models=['logistic_regression'],
        processing_mode='fixed', search_mode='fixed', max_trials=1, decision_mode='recipe_id',
        knowledge=False, budget_awareness='off', fail_fast_guard='off', storage=root,
        thread_id='v8-complete', client=client, llm=llm, wait=False)
    run_id = state['execution']['run_id']
    if outcome == 'prefit_failure':
        source = DatasetRepository(storage/'datasets.sqlite3',storage_root=storage).resolve_system(dataset,legacy_path=None).path
        source.write_bytes(source.read_bytes()+b'\n')
    repo = RunRepository(storage/'runs.sqlite3')
    runner = RunWorker(repository=repo, worker_id='graph-guard', now=lambda:datetime.now(timezone.utc),
        execute=lambda r:execute_with_budget_supervision(r, repository=repo, agent_database=storage/'agent.sqlite3'))
    assert runner.run_once()
    if outcome == 'damaged':
        target = storage/'runs'/run_id/'model_metadata.json'
        content = target.read_bytes(); target.write_bytes(b'!'+content[1:])
    final = resume_task(runtime, storage=root,thread_id='v8-complete',client=client,llm=llm,wait=True)
    assert final['lifecycle']['status'] == 'completed', final['lifecycle']
    assert final['finalization']['backend_session_state'] == ('finalized' if outcome=='success' else 'terminated')
    assert final['guard']['reports']
    assert len(provider.contexts) == (2 if outcome=='success' else 1)
    if outcome == 'success':
        from backend.app.runs import guard
        from agent_poc.orchestration.runtime import read_status
        before = len(transport.requests)
        monkeypatch.setattr(guard, 'RULES_DIGEST', 'f' * 64)
        assert read_status(storage=root, thread_id='v8-complete') == final
        assert resume_task(runtime, storage=root, thread_id='v8-complete', client=client, llm=llm) == final
        assert len(transport.requests) == before
        historical = client.inspect_ml_session(final['identity']['session_id'])
        assert historical['selected_run_id'] == run_id and historical['best_run_id'] is None
