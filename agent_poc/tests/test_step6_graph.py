"""Revision v4 Graph reaches one real Run and replays without extra cost."""
from datetime import datetime, timezone
import pytest

from backend.tests.test_agent_model_sessions import api
from backend.tests.test_finite_search import low_feature_api, HEADERS
from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker
from backend.app.runs.execution import execute_claimed_run
from agent_poc.tests.test_review_recovery import CountTransport
from agent_poc.tests.test_knowledge_graph import KnowledgeProvider
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig
from agent_poc.orchestration.runtime import RuntimeConfig, start_task, resume_task
from agent_poc.orchestration.persistence import CallJournal


def test_graph_default_low_dimension_session_request_replays(monkeypatch,tmp_path,budget_config):
    test_client,_,dataset=low_feature_api(monkeypatch,tmp_path,2)
    llm_config=LLMConfig('http://scripted.invalid/v1','fixture',
        prompt_version='agent-decision-search-v1',**budget_config)
    runtime=RuntimeConfig('http://backend.invalid','local',llm_config)
    wire=CountTransport(test_client)
    client=AutoAIClient(runtime.backend_url,transport=wire,
        api_version='v2',execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v4',processing_mode='fixed',
        search_mode='fixed',max_trials=1,max_retries=0)
    llm=LLMAdapter(llm_config,transport=KnowledgeProvider('json_action'))
    state=start_task(runtime,dataset_id=dataset,allowed_models=['pls_da'],
        processing_mode='fixed',search_mode='fixed',max_trials=1,
        decision_mode='recipe_id',knowledge=False,
        storage=tmp_path/'graph',thread_id='small-pls-graph',client=client,llm=llm,
        wait=False)
    assert state['identity']['session_id']
    request=next(body for method,path,body in wire.calls
                 if method=='POST' and path=='/api/agent/v2/sessions')
    assert not request.get('model_configs')
    replay=test_client.post('/api/agent/v2/sessions',headers=HEADERS,json=request)
    assert replay.status_code==201,replay.text
    assert replay.json()['idempotent_replay'] is True
    assert replay.json()['session_id']==state['identity']['session_id']
    assert replay.json()['locked_config']['capability_snapshot']['model_configs']['pls_da']['pls_components']==2


def test_bounded_search_graph_finalizes_one_run_and_replays(api,tmp_path,budget_config):
    test_client,storage,dataset=api
    llm_config=LLMConfig('http://scripted.invalid/v1','fixture',
        prompt_version='agent-decision-search-v1',**budget_config)
    runtime=RuntimeConfig('http://backend.invalid','local',llm_config)
    client=AutoAIClient(runtime.backend_url,transport=CountTransport(test_client),
        api_version='v2',execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v4',processing_mode='fixed',
        search_mode='bounded',max_trials=2,max_retries=0)
    llm=LLMAdapter(llm_config,transport=KnowledgeProvider('json_action'))
    checkpoints=tmp_path/'graph'
    state=start_task(runtime,dataset_id=dataset,allowed_models=['logistic_regression'],
        processing_mode='fixed',search_mode='bounded',max_trials=2,
        decision_mode='recipe_id',knowledge=False,
        storage=checkpoints,thread_id='step6-bounded',client=client,llm=llm,
        wait=False)
    assert state['lifecycle']['status']=='waiting',state['lifecycle']
    run_id=state['execution']['run_id']
    assert run_id
    repo=RunRepository(storage/'runs.sqlite3');repo.initialize()
    worker=RunWorker(repository=repo,worker_id='step6-graph',
        execute=lambda record:execute_claimed_run(record,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60)
    assert worker.run_once()
    assert repo.get(run_id).state=='succeeded'
    completed=resume_task(runtime,storage=checkpoints,thread_id='step6-bounded',
        client=client,llm=llm,wait=True)
    assert completed['lifecycle']['status']=='completed',completed['lifecycle']
    assert completed['finalization']['selected_run_id']==run_id
    replay=resume_task(runtime,storage=checkpoints,thread_id='step6-bounded',
        client=client,llm=llm,wait=False)
    assert replay['execution']['run_id']==run_id
    assert len(repo.list())==1
    journal=CallJournal(checkpoints/'calls.sqlite','step6-bounded')
    try:
        requests=journal.export_llm_requests()
        assert len(requests)==2
        assert all(row['request_duration_seconds'] is not None for row in requests)
        assert all(row['model_id']=='fixture' and row['protocol']=='json_action'
                   for row in requests)
        assert all(row['task_id']==completed['identity']['task_id'] and
                   row['session_id']==completed['identity']['session_id'] for row in requests)
        waits=journal.export_monitor_waits()
        assert waits and all(row['run_id']==run_id for row in waits)
        assert all(row['session_id']==completed['identity']['session_id'] for row in waits)
        assert all(row['status']=='succeeded' for row in waits)
    finally:
        journal.close()


@pytest.mark.parametrize('search_mode,max_trials', [('fixed',1), ('bounded',2)])
def test_xgboost_graph_worker_finalize_and_replay(api,tmp_path,budget_config,search_mode,max_trials):
    test_client,storage,dataset=api
    llm_config=LLMConfig('http://scripted.invalid/v1','fixture',
        prompt_version='agent-decision-search-v1',**budget_config)
    runtime=RuntimeConfig('http://backend.invalid','local',llm_config)
    wire=CountTransport(test_client)
    client=AutoAIClient(runtime.backend_url,transport=wire,
        api_version='v2',execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v4',processing_mode='fixed',
        search_mode=search_mode,max_trials=max_trials,max_retries=0)
    llm=LLMAdapter(llm_config,transport=KnowledgeProvider('json_action'))
    checkpoints=tmp_path/'graph'
    state=start_task(runtime,dataset_id=dataset,allowed_models=['xgboost'],
        processing_mode='fixed',search_mode=search_mode,max_trials=max_trials,
        decision_mode='recipe_id',knowledge=False,
        storage=checkpoints,thread_id=f'xgb-{search_mode}',client=client,llm=llm,
        wait=False)
    assert state['lifecycle']['status']=='waiting',state['lifecycle']
    run_id=state['execution']['run_id']
    assert run_id
    repo=RunRepository(storage/'runs.sqlite3');repo.initialize()
    assert type(repo.get(run_id).config['xgboost_min_child_weight']) is int
    worker=RunWorker(repository=repo,worker_id='xgb-graph',
        execute=lambda record:execute_claimed_run(record,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60)
    assert worker.run_once()
    assert repo.get(run_id).state=='succeeded',repo.get(run_id).error
    completed=resume_task(runtime,storage=checkpoints,thread_id=f'xgb-{search_mode}',
        client=client,llm=llm,wait=True)
    assert completed['lifecycle']['status']=='completed',completed['lifecycle']
    assert completed['finalization']['selected_run_id']==run_id
    replay=resume_task(runtime,storage=checkpoints,thread_id=f'xgb-{search_mode}',
        client=client,llm=llm,wait=False)
    assert replay['execution']['run_id']==run_id
    assert len(repo.list())==1
