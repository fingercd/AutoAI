"""Scripted protocol tests with genuine isolated backend submission and workers."""
from datetime import datetime, timezone
import json
import httpx
import pytest
from backend.tests.test_agent_model_sessions import api
from agent_poc.tests.test_review_recovery import CountTransport
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMAdapter,LLMConfig
from agent_poc.orchestration.runtime import RuntimeConfig,start_task,resume_task,TaskInterrupted,read_status
from agent_poc.orchestration.graph import Nodes


class RecipeDecision:
    def __init__(self,model,protocol):self.model=model;self.protocol=protocol;self.contexts=[]
    def request(self,method,url,*,headers,json,timeout):
        import json as codec
        context=codec.loads(json['messages'][-1]['content']);self.contexts.append(context)
        args=dict(context['bindings'])
        if context['phase']=='submit':
            args['recipe_id']=next(r['recipe_id'] for r in context['recipes'] if r['model_id']==self.model)
        rationale='Select the permitted classification recipe.'
        tool=context['allowed_actions'][0]
        if self.protocol=='json_action':
            message={'role':'assistant','content':codec.dumps(dict(tool_name=tool,arguments=args,rationale=rationale))}
        else:
            message={'role':'assistant','content':rationale,'tool_calls':[{'id':'choice','type':'function',
                'function':{'name':tool,'arguments':codec.dumps(args)}}]}
        return httpx.Response(200,json={'choices':[{'message':message,'finish_reason':'stop'}],
            'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}})


@pytest.mark.parametrize('model',['logistic_regression','cnn1d'])
@pytest.mark.parametrize('protocol',['json_action','native_tools'])
@pytest.mark.parametrize('interrupt',[False,True])
def test_recipe_graph_recovery(api,tmp_path,monkeypatch,model,protocol,interrupt):
    from backend.app.runs.execution import execute_claimed_run
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.worker import RunWorker
    from backend.app.runs.status_projection import project_status
    api_client,storage,dataset=api;wire=CountTransport(api_client)
    cfg=LLMConfig('http://scripted.invalid/v1','fixture',protocol=protocol,prompt_version='agent-decision-recipes-v1')
    runtime=RuntimeConfig('http://backend.invalid','local',cfg)
    provider=RecipeDecision(model,protocol);adapter=LLMAdapter(cfg,transport=provider)
    def client():return AutoAIClient(runtime.backend_url,transport=wire,api_version='v2',execution_profile='train-evidence-recipes-v1',max_retries=0)
    checkpoint=tmp_path/'checkpoint'
    kwargs=dict(dataset_id=dataset,allowed_models=[model],model_configs={model:{'epochs':2}} if model=='cnn1d' else {},
        storage=checkpoint,thread_id='recipes',client=client(),llm=adapter,evidence_context=False,risk_context=False)
    original=Nodes.submit
    if interrupt:
        def stop(self,state):raise KeyboardInterrupt()
        monkeypatch.setattr(Nodes,'submit',stop)
        with pytest.raises(TaskInterrupted):start_task(runtime,**kwargs)
        before=read_status(storage=checkpoint,thread_id='recipes')
        assert before['recovery']['pending_operation']['status']=='prepared'
        assert before['execution']['submission_content']['recipe_id']
        monkeypatch.setattr(Nodes,'submit',original)
        first=resume_task(runtime,storage=checkpoint,thread_id='recipes',client=client(),llm=adapter)
    else:first=start_task(runtime,**kwargs)
    assert first['lifecycle']['status']=='waiting',first['lifecycle']
    assert first['recipes']['catalog']['recipes']
    assert 'train_statistics' not in provider.contexts[0] and 'train_risks' not in provider.contexts[0]
    assert set(first['execution']['submission_content'])=={'session_id','recipe_id','rationale','client_request_id'}
    repo=RunRepository(storage/'runs.sqlite3');run=first['execution']['run_id']
    worker=RunWorker(repository=repo,worker_id='recipes',execute=lambda r:execute_claimed_run(r,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60,project_status=lambda r:project_status(storage/'runs'/r.run_id,r))
    assert worker.run_once()
    assert repo.get(run).state=='succeeded',repo.get(run).error
    final=resume_task(runtime,storage=checkpoint,thread_id='recipes',client=client(),llm=adapter,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    assert final['finalization']['selected_run_id']==run
    assert len(repo.list())==1
    assert final['budget']['api_calls']['actual']==len(wire.requests)
    assert len(provider.contexts)==2


def test_negotiated_client_uses_frozen_recipe_wire(api):
    from backend.app.recipes import PROFILE
    from urllib.parse import urlsplit
    from agent_poc.clients.autoai_client import AutoAIClient
    class Transport:
        def request(self,method,url,*,headers,json,timeout):
            response=api[0].request(method,urlsplit(url).path,headers=headers,json=json)
            if method=='POST' and url.endswith('/sessions') and response.status_code==201:
                from agent_poc.clients.preparation import RecipeCreatedSessionResponse
                RecipeCreatedSessionResponse.model_validate(response.json())
            return response
    client=AutoAIClient('http://local.invalid',transport=Transport(),api_version='v2',execution_profile=PROFILE)
    client.inspect_ml_capabilities()
    result=client.start_ml_session(dataset_id=api[2],allowed_models=['logistic_regression'],max_runs=1,selection_metric='macro_f1',context_policy={'source_role':'development','case_write':False,'evidence':False,'risks':False})
    preparation=result['locked_config']['preparation']
    client.restore_frozen_session(result['session_id'],result['locked_config']['capability_snapshot'],preparation)
    response=client.submit_ml_experiment(result['session_id'],recipe_id=preparation['catalog']['recipes'][0]['recipe_id'],rationale='Fixed candidate.',client_request_id='selection')
    assert response['binding_state']=='bound'


@pytest.mark.parametrize('decision_mode',['recipe_id','structured_config'])
@pytest.mark.parametrize('interrupt',[False,True])
def test_processing_recipe_graph_keeps_nondefault_member(api,tmp_path,budget_config,decision_mode,interrupt,monkeypatch):
    api_client,storage,dataset=api
    wire=CountTransport(api_client)
    config=LLMConfig('http://scripted.invalid/v1','fixture',protocol='json_action',
        prompt_version='agent-decision-processing-v1',max_tokens=64,**budget_config)
    runtime=RuntimeConfig('http://backend.invalid','local',config)

    class ProcessingDecision(RecipeDecision):
        def request(self,method,url,*,headers,json,timeout):
            import json as codec
            context=codec.loads(json['messages'][-1]['content'])
            self.contexts.append(context)
            args=dict(context['bindings'])
            if context['phase']=='submit':
                recipe=next(r for r in context['recipes'] if
                    r['model_id']=='logistic_regression' and
                    r['normalization']=='area' and
                    r['class_balance']=='class_weight')
                if decision_mode=='recipe_id':
                    args.update(recipe_id=recipe['recipe_id'],knowledge_refs=[])
                else:
                    args.update(model_id=recipe['model_id'],normalization='area',
                        class_balance='class_weight',model_params=context['fixed_model_params'][recipe['model_id']],
                        knowledge_refs=[])
            rationale='Select finite processing.'
            message={'role':'assistant','content':codec.dumps(dict(
                tool_name=context['allowed_actions'][0],arguments=args,rationale=rationale))}
            return httpx.Response(200,json={'choices':[{'message':message,'finish_reason':'stop'}],
                'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}})

    provider=ProcessingDecision('logistic_regression','json_action')
    adapter=LLMAdapter(config,transport=provider)
    client=AutoAIClient(runtime.backend_url,transport=wire,api_version='v2',
        execution_profile='train-evidence-recipes-v1',protocol_revision='agent-recipes-revision-v3',
        decision_mode=decision_mode,processing_mode='dynamic',max_retries=0)
    options=dict(storage=tmp_path/'processing-state',thread_id='processing',
        dataset_id=dataset,allowed_models=['logistic_regression'],processing_mode='dynamic',
        decision_mode=decision_mode,knowledge=False,client=client,llm=adapter)
    if interrupt:
        original=Nodes.submit
        def stop(self,state):raise KeyboardInterrupt()
        monkeypatch.setattr(Nodes,'submit',stop)
        with pytest.raises(TaskInterrupted):start_task(runtime,**options)
        pending=read_status(storage=options['storage'],thread_id='processing')
        assert pending['recovery']['pending_operation']['status']=='prepared'
        assert pending['execution']['submission_content']['recipe_id']
        monkeypatch.setattr(Nodes,'submit',original)
        state=resume_task(runtime,storage=options['storage'],thread_id='processing',client=client,llm=adapter)
    else:
        state=start_task(runtime,**options)
    assert state['lifecycle']['status']=='waiting',state['lifecycle']
    assert state['versions']['state']=='agent-state-v5'
    from backend.app.runs.repository import RunRepository
    record=RunRepository(storage/'runs.sqlite3').get(state['execution']['run_id'])
    assert (record.config['normalization'],record.config['class_balance'])==('area','class_weight')
    assert len(provider.contexts[0]['recipes'])==8
    from backend.app.runs.execution import execute_claimed_run
    from backend.app.runs.worker import RunWorker
    from backend.app.runs.status_projection import project_status
    repo=RunRepository(storage/'runs.sqlite3')
    worker=RunWorker(repository=repo,worker_id='processing',execute=lambda r:execute_claimed_run(r,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60,
        project_status=lambda r:project_status(storage/'runs'/r.run_id,r))
    assert worker.run_once()
    assert repo.get(record.run_id).state=='succeeded',repo.get(record.run_id).error
    resumed=AutoAIClient(runtime.backend_url,transport=wire,api_version='v2',
        execution_profile='train-evidence-recipes-v1',protocol_revision='agent-recipes-revision-v3',
        decision_mode=decision_mode,processing_mode='dynamic',max_retries=0)
    final=resume_task(runtime,storage=tmp_path/'processing-state',thread_id='processing',
        client=resumed,llm=adapter,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    assert final['finalization']['selected_run_id']==record.run_id
