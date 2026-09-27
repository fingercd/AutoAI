"""Scripted provider tests. These are not real Qwen evidence."""
from datetime import datetime, timezone
import json
import sqlite3

import httpx
import pytest

from backend.tests.test_agent_model_sessions import api
from agent_poc.tests.test_review_recovery import CountTransport
from agent_poc.tests.test_knowledge_graph import KnowledgeProvider
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMAdapter, LLMConfig
from agent_poc.orchestration.runtime import RuntimeConfig, start_task, resume_task, read_status


class DiagnosisProvider(KnowledgeProvider):
    def __init__(self,protocol, *, invalid=False, unknown=False, before_diagnosis=None):
        super().__init__(protocol)
        self.invalid=invalid
        self.unknown=unknown
        self.before_diagnosis=before_diagnosis

    def request(self,method,url,*,headers,json:dict,timeout):
        import json as codec
        context=codec.loads(json['messages'][-1]['content'])
        if context['phase']!='diagnose':
            return super().request(method,url,headers=headers,json=json,timeout=timeout)
        if self.before_diagnosis:
            self.before_diagnosis()
        self.contexts.append(context)
        proposal=dict(schema_version='diagnosis-proposal-v1', input_digest=context['input_digest'],
            assessment='no_issue_identified',hypotheses=[],suggestions=[])
        if self.invalid:
            proposal['facts']=[{'value':'forged'}]
        if self.protocol=='json_action':
            message=dict(role='assistant',content=codec.dumps(proposal))
        else:
            message=dict(role='assistant',content=None,tool_calls=[dict(id='diagnostic-call',type='function',
                function=dict(name='report_feedback_diagnosis',arguments=codec.dumps(proposal)))])
        payload=dict(choices=[dict(message=message,finish_reason='stop')])
        if not self.unknown:
            payload['usage']=dict(prompt_tokens=20,completion_tokens=10,total_tokens=30)
        return httpx.Response(200,json=payload)


def started(api,tmp_path,budget_config, *, protocol='json_action', diagnosis='on', awareness='off', guard='on', invalid=False, unknown=False, max_llm_calls=6, full_pool=False):
    test_client, storage, dataset=api
    allowed=['logistic_regression']
    processing='fixed'
    if full_pool:
        from agent_poc.tests.test_step9_acceptance_repairs import full_state
        allowed=full_state()['task']['allowed_models']
        processing='dynamic'
    config=LLMConfig('http://scripted.invalid/v1','fixture',protocol=protocol,
        prompt_version='agent-decision-budget-v1',diagnosis_phase=True,**budget_config)
    runtime=RuntimeConfig('http://backend.invalid','local',config)
    transport=CountTransport(test_client)
    client=AutoAIClient(runtime.backend_url,transport=transport,api_version='v2',
        execution_profile='train-evidence-recipes-v1',protocol_revision='agent-recipes-revision-v7',
        processing_mode=processing,search_mode='fixed',max_trials=1,max_retries=0)
    provider=DiagnosisProvider(protocol,invalid=invalid,unknown=unknown)
    llm=LLMAdapter(config,transport=provider)
    root=tmp_path/'graph'
    state=start_task(runtime,dataset_id=dataset,allowed_models=allowed,
        processing_mode=processing,search_mode='fixed',max_trials=1,decision_mode='recipe_id',
        knowledge=False,budget_awareness=awareness,fail_fast_guard=guard,feedback_diagnosis=diagnosis,
        storage=root,thread_id='step9-graph',client=client,llm=llm,wait=False,max_llm_calls=max_llm_calls)
    assert state['execution']['run_id'],state['lifecycle']
    return state,runtime,transport,client,provider,llm,root


@pytest.mark.parametrize('outcome',['success','failed','damaged'])
@pytest.mark.parametrize('protocol',['json_action','native_tools'])
def test_terminal_routes_and_replay(api,tmp_path,budget_config,outcome,protocol):
    from backend.app.datasets.repository import DatasetRepository
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.worker import RunWorker,execute_with_budget_supervision
    state,runtime,transport,client,provider,llm,root=started(api,tmp_path,budget_config,protocol=protocol)
    _,storage,dataset=api
    run_id=state['execution']['run_id']
    if outcome=='failed':
        source=DatasetRepository(storage/'datasets.sqlite3',storage_root=storage).resolve_system(dataset,legacy_path=None).path
        source.write_bytes(source.read_bytes()+b'\n')
    repo=RunRepository(storage/'runs.sqlite3')
    worker=RunWorker(repository=repo,worker_id='step9-fixture',now=lambda:datetime.now(timezone.utc),
        execute=lambda r:execute_with_budget_supervision(r,repository=repo,agent_database=storage/'agent.sqlite3'))
    assert worker.run_once()
    if outcome=='damaged':
        target=storage/'runs'/run_id/'model_metadata.json'
        original=target.read_bytes();target.write_bytes(b'!'+original[1:])
    def check_closed():
        with sqlite3.connect(storage/'agent.sqlite3') as db:
            # Confirm through the existing scoped client; this is test inspection, not a Graph call.
            pass
        assert client.inspect_ml_session(state['identity']['session_id'])['state']=='terminated'
    if outcome!='success':
        provider.before_diagnosis=check_closed
    final=resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    assert final['diagnosis']['report']['status']=='ready',final['diagnosis']
    assert final['finalization']['backend_session_state']==('finalized' if outcome=='success' else 'terminated')
    contexts=[c for c in provider.contexts if c['phase']=='diagnose']
    assert len(contexts)==1
    if outcome=='success':
        assert any(f['code']=='macro_f1_gap' for f in contexts[0]['facts'])
    else:
        assert not any(f['code'].endswith('_gap') for f in contexts[0]['facts'])
    assert all(c['phase']!='finalize' or 'suggestion_catalog' not in c for c in provider.contexts)
    before=(len(transport.requests),len(provider.contexts))
    assert read_status(storage=root,thread_id='step9-graph')==final
    assert resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm)==final
    assert (len(transport.requests),len(provider.contexts))==before
    with sqlite3.connect(root/'calls.sqlite') as db:
        assert db.execute('SELECT count(*) FROM diagnosis_inputs_v1').fetchone()[0]==1
        assert db.execute('SELECT count(*) FROM diagnosis_reports_v1').fetchone()[0]==1
        assert db.execute("SELECT budget_phase FROM orchestration_calls_v1 WHERE name='diagnose'").fetchone()[0]=='work'


def finish_worker(api):
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.worker import RunWorker,execute_with_budget_supervision
    storage=api[1]
    repo=RunRepository(storage/'runs.sqlite3')
    worker=RunWorker(repository=repo,worker_id='step9-tests',now=lambda:datetime.now(timezone.utc),
        execute=lambda r:execute_with_budget_supervision(r,repository=repo,agent_database=storage/'agent.sqlite3'))
    assert worker.run_once()


@pytest.mark.parametrize('diagnosis',['on','off'])
@pytest.mark.parametrize('awareness',['on','off'])
@pytest.mark.parametrize('guard',['on','off'])
def test_switch_matrix_and_readonly_export(api,tmp_path,budget_config,diagnosis,awareness,guard,monkeypatch):
    import hashlib
    from scripts.export_diagnosis import export_diagnosis
    from backend.app.agent import diagnosis as rules
    state,runtime,wire,client,provider,llm,root=started(api,tmp_path,budget_config,
        diagnosis=diagnosis,awareness=awareness,guard=guard)
    finish_worker(api)
    final=resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    contexts=[c for c in provider.contexts if c['phase']=='diagnose']
    assert len(contexts)==(1 if diagnosis=='on' else 0)
    if contexts:
        assert (contexts[0]['call_cost'] is not None)==(awareness=='on')
        assert (contexts[0]['training_cost'] is not None)==(awareness=='on')
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.sqlite')}
    count=(len(wire.requests),len(provider.contexts))
    monkeypatch.setattr(rules,'rules_digest',lambda:'f'*64)
    assert resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm)==final
    payload=export_diagnosis(storage=root,thread_id='step9-graph',output=tmp_path/'export')
    assert payload['freshness']=='snapshot_only'
    assert payload['current_eligibility']=='unverified'
    assert payload['status']==('ready' if diagnosis=='on' else 'disabled')
    assert set(p.name for p in (tmp_path/'export').iterdir())=={'diagnosis.json','diagnosis.md'}
    assert (len(wire.requests),len(provider.contexts))==count
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.sqlite')}


@pytest.mark.parametrize('mode',['invalid','unknown','tail'])
def test_optional_failure_preserves_finalize(api,tmp_path,budget_config,mode):
    state,runtime,wire,client,provider,llm,root=started(api,tmp_path,budget_config,
        invalid=mode=='invalid',unknown=mode=='unknown',max_llm_calls=4 if mode=='tail' else 6)
    finish_worker(api)
    final=resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    assert final['finalization']['backend_session_state']=='finalized'
    contexts=[c for c in provider.contexts if c['phase']=='diagnose']
    assert len(contexts)=={'invalid':2,'unknown':1,'tail':0}[mode]
    if mode!='unknown':
        assert final['diagnosis']['report']['status']=='unavailable'
    with sqlite3.connect(root/'calls.sqlite') as db:
        rows=db.execute("SELECT status,input_tokens,output_tokens FROM orchestration_calls_v1 WHERE name='diagnose'").fetchall()
        assert len(rows)==len(contexts)
        if mode=='unknown':assert rows[0][1:] == (None,None)


@pytest.mark.parametrize('window',['input_before','input_after','reserved','dispatch_after','response_before_finish','confirmed','report_after'])
def test_diagnostic_crash_replay(api,tmp_path,budget_config,monkeypatch,window):
    from agent_poc.orchestration.persistence import CallJournal
    from agent_poc.orchestration.runtime import TaskInterrupted
    state,runtime,wire,client,provider,llm,root=started(api,tmp_path,budget_config)
    finish_worker(api)
    method={'input_before':'freeze_diagnosis_input','response_before_finish':'finish','input_after':'freeze_diagnosis_input','reserved':'begin','dispatch_after':'mark_dispatched',
            'confirmed':'finish','report_after':'save_diagnosis_report'}[window]
    original=getattr(CallJournal,method)
    fired=[]
    def crash(self,*a,**kw):
        diagnostic=(method in ('freeze_diagnosis_input','save_diagnosis_report') or kw.get('name')=='diagnose'
            or (a and isinstance(a[0],int) and any(row['id']==a[0] and row['name']=='diagnose' for row in self.snapshot())))
        if diagnostic and not fired and window in ('input_before','response_before_finish'):
            fired.append(True)
            raise KeyboardInterrupt
        result=original(self,*a,**kw)
        if diagnostic and not fired:
            fired.append(True)
            raise KeyboardInterrupt
        return result
    with monkeypatch.context() as patch:
        patch.setattr(CallJournal,method,crash)
        with pytest.raises(TaskInterrupted):
            resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm,wait=True)
    calls_before=len([c for c in provider.contexts if c['phase']=='diagnose'])
    final=resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    calls_after=len([c for c in provider.contexts if c['phase']=='diagnose'])
    assert calls_after==(0 if window=='dispatch_after' else 1)
    if window=='response_before_finish':assert calls_before==calls_after==1
    if window in ('confirmed','report_after'):assert calls_before==calls_after
    assert final['execution']['run_id']==state['execution']['run_id']
    with sqlite3.connect(root/'calls.sqlite') as db:
        assert db.execute('SELECT count(*) FROM diagnosis_inputs_v1').fetchone()[0]==1
        assert db.execute('SELECT count(*) FROM diagnosis_reports_v1').fetchone()[0]==1


def test_evidence_changes_after_diagnosis_closes_first(api,tmp_path,budget_config,monkeypatch):
    state,runtime,wire,client,provider,llm,root=started(api,tmp_path,budget_config)
    finish_worker(api)
    changed=[]
    def damage():
        if not changed:
            p=api[1]/'runs'/state['execution']['run_id']/'model_metadata.json'
            b=p.read_bytes();p.write_bytes(b'!'+b[1:]);changed.append(True)
        else:assert client.inspect_ml_session(state['identity']['session_id'])['state']=='terminated'
    provider.before_diagnosis=damage
    final=resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    assert final['finalization']['backend_session_state']=='terminated'
    assert len(final['diagnosis']['report_history'])==2
    assert len([c for c in provider.contexts if c['phase']=='diagnose'])==2


@pytest.mark.parametrize('failure',['unwritable','corrupt'])
def test_persistence_failure_never_dispatches_diagnosis(api,tmp_path,budget_config,monkeypatch,failure):
    from agent_poc.orchestration.persistence import CallJournal
    state,runtime,wire,client,provider,llm,root=started(api,tmp_path,budget_config)
    finish_worker(api)
    if failure=='unwritable':
        def reject(*a,**kw):raise sqlite3.OperationalError('fixture read only')
        monkeypatch.setattr(CallJournal,'freeze_diagnosis_input',reject)
    else:
        with sqlite3.connect(root/'calls.sqlite') as db:
            db.execute("INSERT INTO diagnosis_inputs_v1 VALUES(?,?,?,?,?,?)",('step9-graph','event-1',0,'a'*64,'{}','b'*64))
    final=resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm,wait=True)
    assert final['lifecycle']['status']=='needs_attention'
    assert not [c for c in provider.contexts if c['phase']=='diagnose']
    assert final['execution']['run_id']==state['execution']['run_id']



@pytest.mark.parametrize('protocol',['json_action','native_tools'])
@pytest.mark.parametrize('awareness',['on','off'])
@pytest.mark.parametrize('outcome',['success','failed'])
def test_capacity_receipt_crash_resumes_without_dispatch(api,tmp_path,budget_config,monkeypatch,protocol,awareness,outcome):
    from agent_poc.orchestration.persistence import CallJournal
    from agent_poc.orchestration.runtime import TaskInterrupted
    from backend.app.datasets.repository import DatasetRepository
    from scripts.export_diagnosis import export_diagnosis
    import numpy as np
    import pandas as pd
    frame=pd.DataFrame(np.random.default_rng(3).normal(size=(60,128)),columns=[str(i) for i in range(128)])
    frame.insert(0,'Name',['sample-'+str(i) for i in range(60)])
    frame.insert(0,'Sample_ID',['group-'+str(i//2) for i in range(60)])
    frame.insert(0,'Label',[str(i//30) for i in range(60)])
    frame.insert(0,'Index',list(range(60)))
    uploaded=api[0].post('/api/datasets/upload',files={'file':('capacity.csv',frame.to_csv(index=False).encode(),'text/csv')})
    assert uploaded.status_code==200,uploaded.text
    api=(api[0],api[1],uploaded.json()['dataset_id'])
    state,runtime,wire,client,provider,llm,root=started(api,tmp_path,{**budget_config,'context_window':65536},protocol=protocol,awareness=awareness,full_pool=True)
    assert len(state['recipes']['catalog']['recipes'])==96
    if outcome=='failed':
        source=DatasetRepository(api[1]/'datasets.sqlite3',storage_root=api[1]).resolve_system(api[2],legacy_path=None).path
        source.write_bytes(source.read_bytes()+b'\n')
    finish_worker(api)
    original_tokens=llm._prompt_tokens
    monkeypatch.setattr(llm,'_prompt_tokens',lambda request:1000000 if json.loads(request['messages'][-1]['content'])['phase']=='diagnose' else original_tokens(request))
    original_freeze=CallJournal.freeze_diagnosis_input
    def crash(self,snapshot):
        original_freeze(self,snapshot)
        raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr(CallJournal,'freeze_diagnosis_input',crash)
        with pytest.raises(TaskInterrupted):resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm,wait=True)
    final=resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    assert final['finalization']['backend_session_state']==('finalized' if outcome=='success' else 'terminated')
    assert final['diagnosis']['report']['reason_code']=='diagnosis_context_too_long'
    assert final['execution']['run_id']==state['execution']['run_id']
    assert not [c for c in provider.contexts if c['phase']=='diagnose']
    before=(len(wire.requests),len(provider.contexts))
    exported=export_diagnosis(storage=root,thread_id='step9-graph',output=tmp_path/'capacity-export')
    assert exported['status']=='unavailable'
    assert resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm)==final
    assert before==(len(wire.requests),len(provider.contexts))
    with sqlite3.connect(root/'calls.sqlite') as db:
        assert db.execute("SELECT count(*) FROM orchestration_calls_v1 WHERE name='diagnose'").fetchone()[0]==0
        assert db.execute('SELECT count(*) FROM diagnosis_inputs_v1').fetchone()[0]==1
        assert db.execute('SELECT count(*) FROM diagnosis_reports_v1').fetchone()[0]==1
    with sqlite3.connect(api[1]/'runs.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM runs').fetchone()[0]==1



@pytest.mark.parametrize('assessment',['no_issue_identified','insufficient_evidence'])
def test_assessment_proposal_report_export_and_terminal_replay(api,tmp_path,budget_config,monkeypatch,assessment):
    from backend.app.agent.diagnosis import digest
    from scripts.export_diagnosis import export_diagnosis
    from agent_poc.orchestration.runtime import _historical_snapshot
    from copy import deepcopy
    state,runtime,wire,client,provider,llm,root=started(api,tmp_path,budget_config)
    finish_worker(api)
    original=provider.request
    def response(method,url,*,headers,json:dict,timeout):
        import json as codec
        result=original(method,url,headers=headers,json=json,timeout=timeout)
        if codec.loads(json['messages'][-1]['content'])['phase']=='diagnose':
            payload=result.json();proposal=codec.loads(payload['choices'][0]['message']['content'])
            proposal['assessment']=assessment
            payload['choices'][0]['message']['content']=codec.dumps(proposal)
            return httpx.Response(200,json=payload)
        return result
    monkeypatch.setattr(provider,'request',response)
    final=resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm,wait=True)
    assert final['lifecycle']['status']=='completed',final['lifecycle']
    before=(len(wire.requests),len(provider.contexts))
    exported=export_diagnosis(storage=root,thread_id='step9-graph',output=tmp_path/'assessment-export')
    assert exported['report']['assessment']==assessment
    assert assessment in (tmp_path/'assessment-export/diagnosis.md').read_text(encoding='utf8')
    assert resume_task(runtime,storage=root,thread_id='step9-graph',client=client,llm=llm)==final
    assert before==(len(wire.requests),len(provider.contexts))
    # A genuine legacy report shape remains hash-identical in the database; the read projection labels it.
    with sqlite3.connect(root/'calls.sqlite') as db:
        raw=db.execute('SELECT payload_json FROM diagnosis_reports_v1').fetchone()[0]
        old=json.loads(raw);del old['assessment'];del old['proposal_digest']
        old_raw=json.dumps(old)
        db.execute('UPDATE diagnosis_reports_v1 SET payload_json=?,payload_digest=?',(old_raw,digest(old)))
    old_export=export_diagnosis(storage=root,thread_id='step9-graph',output=tmp_path/'historical-export')
    assert old_export['report']['assessment']=='historically_unrecorded'
    assert old_export['report']['facts']==exported['report']['facts']
    historical=deepcopy(final);historical['diagnosis']['report']=old
    assert _historical_snapshot(historical)['diagnosis']['report']['assessment']=='historically_unrecorded'
    with sqlite3.connect(root/'calls.sqlite') as db:
        assert db.execute('SELECT payload_json FROM diagnosis_reports_v1').fetchone()[0]==old_raw
