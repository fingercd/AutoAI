"""V2 graph wiring uses real Session/Run HTTP and training, scripted decisions."""
from datetime import datetime, timezone
from dataclasses import replace
import json
import pytest
import httpx
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMConfig, LLMAdapter
from agent_poc.orchestration.runtime import RuntimeConfig,start_task,resume_task,read_status
from agent_poc.orchestration.state import validate_state
from agent_poc.tests.test_orchestration_integration import BackendTransport
from backend.tests.test_agent_model_sessions import api
from backend.app.model_catalog import MODEL_DECLARATIONS, model_availability


class ScriptedDecision:
    def __init__(self,model,protocol='json_action'):
        self.model=model
        self.protocol=protocol
        self.requests=[]
    def request(self,method,url,*,headers,json,timeout):
        import json as codec
        self.requests.append(json)
        context=codec.loads(json['messages'][-1]['content'])
        args=dict(context['bindings'])
        if context['phase']=='submit':
            args['model_type']=self.model
            assert self.model in context['capabilities']['models']
            rationale='Select the permitted classification model.'
        else:
            rationale='Finalize the validated candidate.'
        tool=context['allowed_actions'][0]
        if self.protocol=='json_action':
            message={'role':'assistant','content':codec.dumps(dict(tool_name=tool,arguments=args,rationale=rationale))}
        else:
            message={'role':'assistant','content':rationale,'tool_calls':[{'id':'tool-call','type':'function',
                'function':{'name':tool,'arguments':codec.dumps(args)}}]}
        return httpx.Response(200,json={'model':'scripted','choices':[{'index':0,'finish_reason':'stop','message':message}],
            'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}})


@pytest.mark.parametrize('model',[m.id for m in MODEL_DECLARATIONS if model_availability(m.id)[0]])
def test_all_available_graph_models(api,tmp_path,model,_lose_response=False):
    from backend.app.runs.execution import execute_claimed_run
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.worker import RunWorker
    from backend.app.runs.status_projection import project_status
    transport,storage,dataset=api
    llm_config=LLMConfig('http://scripted.invalid/v1','scripted',prompt_version='agent-decision-step2-v1')
    runtime=RuntimeConfig('http://backend.invalid','local',llm_config)
    provider=ScriptedDecision(model)
    adapter=LLMAdapter(llm_config,transport=provider)
    class FaultTransport(BackendTransport):
        lost=False
        def request(self,method,url,**kwargs):
            response=super().request(method,url,**kwargs)
            if _lose_response and not self.lost and method=='POST' and url.endswith('/experiments') and response.status_code==202:
                self.lost=True
                raise TimeoutError('injected response loss after durable Run creation')
            return response
    wire=FaultTransport(transport)
    def client(): return AutoAIClient(runtime.backend_url,transport=wire,api_version='v2',max_retries=0)
    configs={model:{'epochs':2}} if next(m for m in MODEL_DECLARATIONS if m.id==model).execution_family=='deep_learning' else {}
    checkpoint=tmp_path/'graph'
    first=start_task(runtime,dataset_id=dataset,allowed_models=[model],model_configs=configs,storage=checkpoint,
        thread_id='v2',client=client(),llm=adapter,wait=False)
    assert first['lifecycle']['status']=='waiting',first['lifecycle']
    assert first['execution']['submission_content']['model_params']==first['capabilities']['frozen_snapshot']['model_configs'][model]
    run=first['execution']['run_id']
    repo=RunRepository(storage/'runs.sqlite3')
    if _lose_response:
        assert wire.lost
        assert run is None  # The durable response was lost before graph binding.
        created=repo.list()
        assert len(created)==1
        assert created[0].state=='queued'
        run=created[0].run_id
    worker=RunWorker(repository=repo,worker_id='v2-graph-worker',execute=lambda r:execute_claimed_run(r,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60,
        project_status=lambda r:project_status(storage/'runs'/r.run_id,r))
    assert worker.run_once()
    assert repo.get(run).state=='succeeded',repo.get(run).error
    final=resume_task(runtime,storage=checkpoint,thread_id='v2',client=client(),llm=adapter,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    assert final['finalization']['selected_run_id']==run
    assert validate_state(final).versions.state=='agent-state-v2'
    assert len(final)==21
    assert read_status(storage=checkpoint,thread_id='v2')==final
    assert resume_task(runtime,storage=checkpoint,thread_id='v2',client=client(),llm=adapter)==final
    assert len(repo.list())==1


@pytest.mark.parametrize('protocol',['json_action','native_tools'])
def test_new_model_proposals_bind_fixed_params(protocol):
    from agent_poc.orchestration.projection import selection_context
    context=selection_context(task=dict(allowed_models=['cnn1d'],seed=42,selection_metric='macro_f1',
        evaluation_config=dict(mode='stratified_holdout',train_weight=8,validation_weight=1,heldout_weight=1)),
        models=['cnn1d'],session_id='s',client_request_id='e',model_configs={'cnn1d':{'epochs':2}})
    provider=ScriptedDecision('cnn1d',protocol)
    adapter=LLMAdapter(LLMConfig('http://scripted.invalid/v1','scripted',protocol=protocol,prompt_version='agent-decision-step2-v1'),transport=provider)
    proposal=adapter.propose('submit',context)
    assert proposal.arguments['model_params']=={'epochs':2}
    assert 'model_params' not in json.dumps(provider.requests[0].get('tools',[]))


@pytest.mark.parametrize('model',['logistic_regression','cnn1d'])
def test_v2_graph_recovers_post_response_loss(api,tmp_path,model):
    test_all_available_graph_models(api,tmp_path,model,_lose_response=True)
