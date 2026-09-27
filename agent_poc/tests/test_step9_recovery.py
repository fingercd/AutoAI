"""Confirmed no-Run events preserve termination and durable proposals."""
from dataclasses import replace
import json,sqlite3
import pytest
from backend.tests.test_agent_model_sessions import api
from agent_poc.tests.test_step78_budget_rejection import setup_task,UnaffordableProvider
from agent_poc.tests.test_step9_graph import DiagnosisProvider
from agent_poc.orchestration.llm import LLMAdapter
from agent_poc.orchestration.runtime import start_task,resume_task,TaskInterrupted


@pytest.mark.parametrize('capacity',[False,True])
@pytest.mark.parametrize('protocol',['json_action','native_tools'])
@pytest.mark.parametrize('kind',['admission','budget'])
@pytest.mark.parametrize('crash',[False,True])
def test_no_run_closes_before_explanation_and_never_reposts(api,tmp_path,budget_config,monkeypatch,protocol,kind,crash,capacity):
    from backend.app.datasets.repository import DatasetRepository
    from backend.app.runs.repository import RunRepository
    from agent_poc.orchestration.graph import Nodes
    runtime,kwargs,wire,old=setup_task(api,tmp_path,budget_config,version=9,protocol=protocol)
    runtime=replace(runtime,llm_config=replace(runtime.llm_config,diagnosis_phase=True))
    kwargs.update(feedback_diagnosis='on',fail_fast_guard='on')
    chooser=UnaffordableProvider(protocol)
    provider=DiagnosisProvider(protocol)
    choose_request=provider.request
    changed=[]
    def request(method,url,*,headers,json,timeout):
        import json as codec
        ctx=codec.loads(json['messages'][-1]['content'])
        if ctx['phase']=='diagnose':
            result=choose_request(method,url,headers=headers,json=json,timeout=timeout)
        elif kind=='budget':
            result=chooser.request(method,url,headers=headers,json=json,timeout=timeout)
        else:
            result=choose_request(method,url,headers=headers,json=json,timeout=timeout)
            if not changed:
                p=DatasetRepository(api[1]/'datasets.sqlite3',storage_root=api[1]).resolve_system(api[2],legacy_path=None).path
                p.write_bytes(p.read_bytes()+b'\n');changed.append(True)
        return result
    provider.request=request
    kwargs['llm']=LLMAdapter(runtime.llm_config,transport=provider)
    if capacity:
        original_tokens=kwargs['llm']._prompt_tokens
        monkeypatch.setattr(kwargs['llm'],'_prompt_tokens',lambda request:1000000 if request['messages'][-1]['content'].startswith('{') and json.loads(request['messages'][-1]['content']).get('phase')=='diagnose' else original_tokens(request))
    if kind=='admission':
        kwargs['allowed_models']=['cnn1d'];kwargs['model_configs']={'cnn1d':{'epochs':2}}
    def closed():

        with sqlite3.connect(api[1]/'agent.sqlite3') as db:
            assert db.execute('SELECT terminated_at FROM agent_sessions_v1').fetchone()[0] is not None
    provider.before_diagnosis=closed
    original=Nodes.termination_confirmed
    def interrupt(self,state,response):
        result=original(self,state,response)
        assert result['lifecycle']['stage']=='post_termination_diagnosis'
        raise KeyboardInterrupt
    if crash:
        with monkeypatch.context() as patch:
            patch.setattr(Nodes,'termination_confirmed',interrupt)
            with pytest.raises(TaskInterrupted):start_task(runtime,**kwargs)
        final=resume_task(runtime,**{k:kwargs[k] for k in ('storage','thread_id','client','llm','wait')})
    else:final=start_task(runtime,**kwargs)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    assert final['execution']['run_id'] is None
    assert not RunRepository(api[1]/'runs.sqlite3').list()
    assert final['diagnosis']['report']['status']==('unavailable' if capacity else 'ready'),final['diagnosis']
    if capacity:
        assert final['diagnosis']['report']['reason_code']=='diagnosis_context_too_long'
        assert not [c for c in provider.contexts if c['phase']=='diagnose']
    else:
        assert ('budget_rejected' if kind=='budget' else 'data_invalid') in final['diagnosis']['report']['problem_codes']
    if kind=='budget':assert final['decision']['reason_code']=='recipe_training_budget_exceeded'
    before=(len(wire.requests),len(provider.contexts),len(chooser.contexts))
    again=resume_task(runtime,**{k:kwargs[k] for k in ('storage','thread_id','client','llm','wait')})
    assert again==final
    assert before==(len(wire.requests),len(provider.contexts),len(chooser.contexts))
    posts=[row for row in wire.calls if row[0]=='POST' and row[1].endswith('/terminate')]
    assert len(posts)==1
