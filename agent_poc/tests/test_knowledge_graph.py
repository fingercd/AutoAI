import json
from datetime import datetime, timezone

import httpx
import pytest

from backend.tests.test_agent_model_sessions import api
from agent_poc.tests.test_review_recovery import CountTransport
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig
from agent_poc.orchestration.runtime import RuntimeConfig, start_task, resume_task, read_status, TaskInterrupted
from agent_poc.orchestration.graph import Nodes


@pytest.mark.parametrize('version,expected',[
    ('agent-state-v1','6560b5a49b1c5dd22e3072843091961c2704ebeff9e93e7a5af284b98a960db6'),
    ('agent-state-v2','525dce3d3d03e89c998ad1770ef994157f5107c1a579254e7d5428e7aea5edf5'),
    ('agent-state-v3','ffcb0ef79ad1706a92687e71eebea9b2bc1094b848e75f61342d0de22a5cd1f6'),
])
def test_historical_wire_golden_from_third_step(version,expected):
    # Independently generated from the 10fd4e66 third-step source, not from v4.
    from agent_poc.orchestration.state import new_state,fingerprint
    state=new_state(dataset_id='fixed-data',allowed_models=['logistic_regression'],backend_fingerprint='a'*64,
        principal_fingerprint='b'*64,llm_config_fingerprint='c'*64,task_id='fixed-task',thread_id='fixed-thread',
        wire_version=version,now=1000.0)
    assert fingerprint(state)==expected


class CheckedTransport(CountTransport):
    def request(self,method,url,**kwargs):
        response=super().request(method,url,**kwargs)
        if url.endswith('/sessions') and response.status_code==201:
            from agent_poc.clients.knowledge import KnowledgeCreatedSessionResponse
            try:
                KnowledgeCreatedSessionResponse.model_validate(response.json())
            except ValueError as exc:
                print(exc)
                raise
        return response


@pytest.mark.parametrize('query',[
    {'query_mode':'train_template','user_text':None,'domain':None},
    {'query_mode':'user_text','user_text':'Small samples with many measured features.','domain':None},
])
def test_explicit_query_reaches_prepared_graph(api,tmp_path,monkeypatch,budget_config,query):
    wire=CheckedTransport(api[0])
    cfg=LLMConfig('http://scripted.invalid/v1','fixture',prompt_version='agent-decision-knowledge-v1', **budget_config)
    runtime=RuntimeConfig('http://backend.invalid','local',cfg)
    client=AutoAIClient(runtime.backend_url,transport=wire,api_version='v2',execution_profile='train-evidence-recipes-v1',protocol_revision='agent-recipes-revision-v2',max_retries=0)
    def stop(self,state):raise KeyboardInterrupt()
    monkeypatch.setattr(Nodes,'submit',stop)
    with pytest.raises(TaskInterrupted):
        start_task(runtime,dataset_id=api[2],allowed_models=['logistic_regression'],knowledge=True,
            knowledge_query=query,storage=tmp_path/'query-checkpoint',thread_id='explicit-query',
            client=client,llm=LLMAdapter(cfg,transport=KnowledgeProvider('json_action')))
    state=read_status(storage=tmp_path/'query-checkpoint',thread_id='explicit-query')
    assert state['task']['knowledge_query']==query
    assert state['identity']['session_id']
    assert state['knowledge']['snapshot']['schema_version']=='knowledge-snapshot-rag-v1'
    assert state['execution']['submission_content'] is not None


class KnowledgeProvider:
    """Controlled protocol fixture; never evidence of a real LLM call."""
    def __init__(self,protocol):self.protocol=protocol;self.contexts=[]
    def request(self,method,url,*,headers,json:dict,timeout):
        import json as codec
        context=codec.loads(json['messages'][-1]['content']);self.contexts.append(context)
        args=dict(context['bindings'])
        if context['phase']=='submit':
            args.update(recipe_id=context['recipes'][0]['recipe_id'],knowledge_refs=context['knowledge']['provided_entry_ids'][:1])
        tool=context['allowed_actions'][0];rationale='Compare the frozen baseline using the provided limitations.'
        if self.protocol=='json_action':
            message=dict(role='assistant',content=codec.dumps(dict(tool_name=tool,arguments=args,rationale=rationale)))
        else:
            message=dict(role='assistant',content=rationale,tool_calls=[dict(id='choice',type='function',
                function=dict(name=tool,arguments=codec.dumps(args)))])
        return httpx.Response(200,json=dict(choices=[dict(message=message,finish_reason='stop')],
            usage=dict(prompt_tokens=20,completion_tokens=10,total_tokens=30)))


@pytest.mark.parametrize('protocol',['json_action','native_tools'])
@pytest.mark.parametrize('refs',[[],['unknown'],['km_grouped_repeats','km_grouped_repeats']])
def test_llm_reference_boundary(api,protocol,refs):
    from backend.tests.test_knowledge_session import create
    from agent_poc.orchestration.projection import recipe_selection_context
    from agent_poc.orchestration.llm import LLMError
    created,_=create(api);locked=created['locked_config']
    context=recipe_selection_context(task=locked,session_id=created['session_id'],preparation=locked['preparation'],
        context_policy={**locked['context_policy'],'projection':'agent-context-knowledge-v1'})
    class AlteredProvider(KnowledgeProvider):
        def request(self,*args,**kwargs):
            response=super().request(*args,**kwargs);body=response.json();message=body['choices'][0]['message']
            if self.protocol=='json_action':
                action=json.loads(message['content']);action['arguments']['knowledge_refs']=refs
                message['content']=json.dumps(action)
            else:
                action=json.loads(message['tool_calls'][0]['function']['arguments']);action['knowledge_refs']=refs
                message['tool_calls'][0]['function']['arguments']=json.dumps(action)
            return httpx.Response(200,json=body)
    cfg=LLMConfig('http://scripted.invalid/v1','fixture',protocol=protocol,prompt_version='agent-decision-knowledge-v1')
    adapter=LLMAdapter(cfg,transport=AlteredProvider(protocol))
    if refs:
        with pytest.raises(LLMError) as error:adapter.propose('submit',context)
        assert error.value.code=='llm_output_invalid'
        assert error.value.usage.total_tokens==30
    else:
        assert adapter.propose('submit',context).arguments['knowledge_refs']==[]


@pytest.mark.parametrize('decision_mode',[None,'recipe_id','structured_config'])
@pytest.mark.parametrize('enabled',[True,False])
@pytest.mark.parametrize('protocol',['json_action','native_tools'])
def test_knowledge_graph_and_prepared_restart(api,tmp_path,monkeypatch,budget_config,enabled,protocol,decision_mode):
    from backend.app.runs.execution import execute_claimed_run
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.worker import RunWorker
    from backend.app.runs.status_projection import project_status
    client_api,storage,dataset=api;wire=CheckedTransport(client_api)
    cfg=LLMConfig('http://scripted.invalid/v1','fixture',protocol=protocol,prompt_version='agent-decision-knowledge-v1', **budget_config)
    runtime=RuntimeConfig('http://backend.invalid','local',cfg)
    from backend.tests.test_agent_decision_modes import ExpressionProvider
    provider=ExpressionProvider(protocol);adapter=LLMAdapter(cfg,transport=provider)
    def client():return AutoAIClient(runtime.backend_url,transport=wire,api_version='v2',execution_profile='train-evidence-recipes-v1',
                                    protocol_revision='agent-recipes-revision-v2',max_retries=0)
    checkpoint=tmp_path/'checkpoint';original=Nodes.submit
    def stop(self,state):raise KeyboardInterrupt()
    monkeypatch.setattr(Nodes,'submit',stop)
    with pytest.raises(TaskInterrupted):
        start_task(runtime,dataset_id=dataset,allowed_models=['logistic_regression'],storage=checkpoint,thread_id='knowledge',
                   client=client(),llm=adapter,knowledge=enabled,decision_mode=decision_mode)
    before=read_status(storage=checkpoint,thread_id='knowledge')
    assert before['knowledge']['status']==('ready' if enabled else 'disabled')
    if enabled:
        assert before['versions']['knowledge']['version']=='knowledge-snapshot-rag-v1'
    assert bool(before['decision']['evidence_refs'])==enabled
    assert before['recovery']['pending_operation']['status']=='prepared'
    monkeypatch.setenv('AUTOAI_KNOWLEDGE_BUNDLE',str(tmp_path/'missing-publication.json'))
    monkeypatch.setattr(Nodes,'submit',original)
    first=resume_task(runtime,storage=checkpoint,thread_id='knowledge',client=client(),llm=adapter)
    assert first['lifecycle']['status']=='waiting',first['lifecycle']
    assert first['identity']['startup_config_fingerprint']==before['identity']['startup_config_fingerprint']
    repo=RunRepository(storage/'runs.sqlite3');run=first['execution']['run_id']
    worker=RunWorker(repository=repo,worker_id='knowledge',execute=lambda r:execute_claimed_run(r,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60,project_status=lambda r:project_status(storage/'runs'/r.run_id,r))
    assert worker.run_once()
    assert repo.get(run).state=='succeeded'
    final=resume_task(runtime,storage=checkpoint,thread_id='knowledge',client=client(),llm=adapter,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    assert len(repo.list())==1 and len(provider.contexts)==2
    assert final['budget']['api_calls']['actual']==len(wire.requests)
    assert final['knowledge']==before['knowledge']


@pytest.mark.parametrize('boundary',['session_response','llm_journal','journal_missing','journal_corrupt','run_response'])
def test_new_revision_lost_response_and_journal_replay(api,tmp_path,monkeypatch,budget_config,boundary):
    import time
    from backend.app.runs.repository import RunRepository
    from agent_poc.orchestration.state import fingerprint
    class LostResponse(CountTransport):
        lost=False
        def request(self,method,url,**kwargs):
            response=super().request(method,url,**kwargs)
            selected=(boundary=='session_response' and url.endswith('/sessions') or
                      boundary=='run_response' and url.endswith('/experiments'))
            if selected and method=='POST' and 200<=response.status_code<300 and not self.lost:
                self.lost=True
                raise TimeoutError('response lost after server commit')
            return response
    wire=LostResponse(api[0]);cfg=LLMConfig('http://scripted.invalid/v1','fixture',prompt_version='agent-decision-knowledge-v1', **budget_config)
    runtime=RuntimeConfig('http://backend.invalid','local',cfg);provider=KnowledgeProvider('json_action');adapter=LLMAdapter(cfg,transport=provider)
    def client():return AutoAIClient(runtime.backend_url,transport=wire,api_version='v2',max_retries=0,
        execution_profile='train-evidence-recipes-v1',protocol_revision='agent-recipes-revision-v2')
    clock=[time.time()];checkpoint=tmp_path/'checkpoint'
    original=Nodes.llm_call
    if boundary in ('llm_journal','journal_missing','journal_corrupt'):
        def crash(self,*args,**kwargs):
            original(self,*args,**kwargs)
            raise KeyboardInterrupt()
        monkeypatch.setattr(Nodes,'llm_call',crash)
        with pytest.raises(TaskInterrupted):
            start_task(runtime,dataset_id=api[2],allowed_models=['logistic_regression'],knowledge=True,
                client=client(),llm=adapter,storage=checkpoint,thread_id='lost',clock=lambda:clock[0])
        first=read_status(storage=checkpoint,thread_id='lost')
        monkeypatch.setattr(Nodes,'llm_call',original)
        if boundary.startswith('journal_'):
            import sqlite3
            with sqlite3.connect(checkpoint/'calls.sqlite') as connection:
                connection.execute("UPDATE orchestration_calls_v1 SET proposal_json=? WHERE kind='llm' AND status='confirmed'",
                    (None if boundary=='journal_missing' else '{}',))
    else:
        first=start_task(runtime,dataset_id=api[2],allowed_models=['logistic_regression'],knowledge=True,
            client=client(),llm=adapter,storage=checkpoint,thread_id='lost',clock=lambda:clock[0])
        assert wire.lost
    monkeypatch.setenv('AUTOAI_KNOWLEDGE_BUNDLE',str(tmp_path/'missing.json'))
    clock[0]+=3
    recovered=resume_task(runtime,client=client(),llm=adapter,storage=checkpoint,thread_id='lost',clock=lambda:clock[0])
    if boundary.startswith('journal_'):
        assert recovered['lifecycle']['status']=='needs_attention',recovered['lifecycle']
        assert len(provider.contexts)==1
        assert not RunRepository(api[1]/'runs.sqlite3').list()
        return
    assert recovered['execution']['run_id'],recovered['lifecycle']
    assert len(RunRepository(api[1]/'runs.sqlite3').list())==1
    assert len(provider.contexts)==1
    assert fingerprint(provider.contexts[0]['knowledge'])==recovered['knowledge']['snapshot']['projection_digest']
    assert recovered['identity']['startup_config_fingerprint']==first['identity']['startup_config_fingerprint']
    assert recovered['budget']['api_calls']['actual']==len(wire.requests)
def test_prompt_budget_drops_whole_cards_and_preserves_recipes(api, monkeypatch):
    from backend.tests.test_knowledge_session import create
    from agent_poc.orchestration.projection import recipe_selection_context
    from agent_poc.orchestration.llm import LLMError
    created, _ = create(api)
    locked = created['locked_config']
    context = recipe_selection_context(task=locked, session_id=created['session_id'],
        preparation=locked['preparation'], context_policy={**locked['context_policy'], 'projection':'agent-context-knowledge-v1'})
    config = LLMConfig('http://fixture.invalid/v1', 'fixture', max_tokens=64,
        prompt_version='agent-decision-knowledge-v1', tokenizer_path='fixture', context_window=350)
    # Synthetic tokenizer budget fixture; actual tokenizer is exercised in real runs.
    monkeypatch.setattr(LLMConfig, 'public_config', lambda self: {'context_version':'agent-context-knowledge-v1'})
    adapter = LLMAdapter(config, transport=KnowledgeProvider('json_action'))
    monkeypatch.setattr(adapter, '_prompt_tokens', lambda request:
        100 + 100 * len(json.loads(request['messages'][-1]['content'])['knowledge']['entries']))
    prepared = adapter.prepare_context('submit', context)
    assert len(prepared['knowledge']['entries']) == 1
    assert prepared['knowledge']['entries'][0] == context['knowledge']['entries'][0]
    assert prepared['recipes'] == context['recipes']
    assert len(context['knowledge']['entries']) == 2
    monkeypatch.setattr(adapter, '_prompt_tokens', lambda request: 400)
    with pytest.raises(LLMError, match='llm_context_too_long'):
        adapter.prepare_context('submit', context)
