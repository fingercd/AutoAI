"""Failed search rows require settled local calls and a scoped terminal Run."""
import hashlib
import hmac
import json
import sqlite3

import httpx
import pytest

from backend.app.model_catalog import MODELS_BY_ID
from backend.app.model_config import model_policy
from backend.app.processing_policy import EXECUTABLE_MODELS, freeze_fixed_processing
from backend.tests.test_agent_model_sessions import api
from scripts.agent_ablation import (_confirm_failed_search_backend, collect_search_plan,
    digest, register_search_plan, save)

BACKEND='http://fixture.invalid'
SCOPE='fixture-scope'


def _fixture(tmp_path, monkeypatch, *, failure='llm', unsettled=False,
             wrong_run=False, wrong_principal=False, active_run=False,active_wait=False,
             failed_submission_call=False):
    monkeypatch.delenv('AUTOAI_API_TOKEN',raising=False)
    models=sorted(EXECUTABLE_MODELS)
    config=dict(tasks=[dict(task_id=f'task{i}',dataset_id=f'dataset{i}',
        dataset_sha256=f'{i:064x}') for i in range(10)],
        model_pools={family:[m for m in models if MODELS_BY_ID[m].execution_family==kind]
            for family,kind in (('ML','traditional_ml'),('DL','deep_learning'))},
        fixed_processing=freeze_fixed_processing(models,None),
        model_configs={m:{p['name']:p['default'] for p in model_policy(m,search_revision=True)['parameters']}
            for m in models},seed=42,repeat=0,bounded_max_trials=2,
        conditions={**{k:'a'*64 for k in ('llm_binding','source_binding',
            'tokenizer_binding','prompt_binding')},'backend_binding':digest(BACKEND),
            'scope_binding':digest(SCOPE)})
    plan=register_search_plan(config,'collection')
    storage=tmp_path/'rows'
    plan_path=storage/'collection'/'plan.json'
    save(plan_path,plan)
    failed=plan['rows'][0]
    states={}
    for row in plan['rows']:
        experiment_id=row['experiment_id']
        is_failed=row==failed
        run_id=None if is_failed and failure=='experiment' else 'run-'+experiment_id
        status=('needs_attention' if failure=='experiment' else 'failed') if is_failed else 'completed'
        session_id='session-'+experiment_id
        request_id='experiment-'+experiment_id
        principal=hmac.new(b'agent-local-scope-v1',
            '\0'.join(('agent-principal-binding-v1',BACKEND,SCOPE)).encode(),
            hashlib.sha256).hexdigest()
        record=dict(experiment_id=experiment_id,status=status,run_id=run_id,
            dataset_digest=row['dataset_sha256'],session_id=session_id,
            experiment_request_id=request_id,principal_fingerprint=principal,
            source={'source_digest':'a'*64},
            configuration=dict(plan_digest=plan['plan_digest'],dataset_id=row['dataset_id'],
                seed=42,processing_mode='fixed',search_mode=row['search_mode'],
                max_trials=row['max_trials'],allowed_models=sorted(row['allowed_models']),
                fixed_processing={m:plan['fixed_processing'][m] for m in row['allowed_models']},
                model_configs=config['model_configs'],
                **{k:plan['conditions'][k] for k in ('backend_binding','scope_binding','llm_binding')}))
        if not is_failed:
            record['offline_test']=dict(macro_f1=.5,balanced_accuracy=.5,accuracy=.5)
        save(storage/experiment_id/'record.json',record)
        pending=(dict(operation_id=('finalize-'+experiment_id if failure=='llm' else request_id),
            kind=failure,status='unknown',request_id=request_id if failure=='experiment' else None,
            content={'session_id':session_id} if failure=='experiment' else None) if is_failed else None)
        states[experiment_id]=dict(identity=dict(thread_id=experiment_id,session_id=session_id,
            experiment_request_id=request_id,principal_fingerprint=('c'*64 if wrong_principal and is_failed else principal),
            backend_fingerprint=hashlib.sha256(BACKEND.encode()).hexdigest()),
            task=dict(dataset_id=row['dataset_id'],dataset_fingerprint=row['dataset_sha256'],
                processing_mode='fixed',search_mode=row['search_mode']),
            lifecycle=dict(status=status,next_action=None,ended_at=1),
            recovery=dict(needs_human_review=is_failed,pending_operation=pending),
            execution=dict(run_id=run_id),
            finalization=dict(status='confirmed',selected_run_id=run_id,
                backend_session_state='finalized'))
    from agent_poc.orchestration import runtime
    monkeypatch.setattr(runtime,'read_status',lambda *,storage,thread_id:states[thread_id])
    journal=storage/failed['experiment_id']/'checkpoints'/'calls.sqlite'
    journal.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(journal) as db:
        db.execute('''CREATE TABLE orchestration_calls_v1 (id INTEGER PRIMARY KEY,thread_id TEXT,operation_id TEXT,
            kind TEXT,status TEXT,ended_at REAL,session_id TEXT,run_id TEXT)''')
        db.execute('''CREATE TABLE orchestration_monitor_waits_v1 (id INTEGER PRIMARY KEY,
            thread_id TEXT,status TEXT,ended_at_utc TEXT)''')
        db.execute('''INSERT INTO orchestration_calls_v1
            (thread_id,operation_id,kind,status,ended_at,session_id,run_id) VALUES (?,?,?,?,?,?,?)''',
            (failed['experiment_id'],states[failed['experiment_id']]['recovery']['pending_operation']['operation_id'],
             'llm' if failure=='llm' else 'api',
             'dispatched' if unsettled else 'failed' if failure=='llm' or failed_submission_call else 'confirmed',
             None if unsettled else 1,'session-'+failed['experiment_id'],None))
        if active_wait:
            db.execute('INSERT INTO orchestration_monitor_waits_v1 (thread_id,status,ended_at_utc) VALUES (?,?,?)',
                (failed['experiment_id'],'running',None))
    calls=[]
    class Response:
        def raise_for_status(self):pass
        def json(self):
            return dict(session_id='session-'+failed['experiment_id'],state='open',
                locked_config=dict(dataset_id=failed['dataset_id'],
                    dataset_sha256=failed['dataset_sha256'],max_runs=1),remaining_runs=0,
                experiments=[dict(run_id=('wrong-run' if wrong_run else 'run-'+failed['experiment_id']),
                    binding_state='bound',state='running' if active_run else 'succeeded')])
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def get(self,path,**kwargs):
            calls.append((path,kwargs))
            return Response()
    monkeypatch.setattr(httpx,'Client',Client)
    return plan_path,storage,failed,calls


@pytest.mark.parametrize('failure', ['llm','experiment'])
def test_failed_row_is_excluded_with_durable_idempotent_settlement(tmp_path,monkeypatch,failure):
    path,storage,row,calls=_fixture(tmp_path,monkeypatch,failure=failure)
    first=collect_search_plan(path,storage,BACKEND,SCOPE)
    evidence_path=storage/row['experiment_id']/'failure-settlement.json'
    evidence=json.loads(evidence_path.read_text(encoding='utf-8'))
    assert first['complete'] is False
    assert first['groups']['fixed']['succeeded']+first['groups']['bounded']['succeeded']==39
    assert evidence['terminal_status']==('failed' if failure=='llm' else 'needs_attention')
    assert evidence['settled_run_id']=='run-'+row['experiment_id']
    assert len(calls)==1 and '/sessions/' in calls[0][0]
    assert collect_search_plan(path,storage,BACKEND,SCOPE)==first
    assert json.loads(evidence_path.read_text(encoding='utf-8'))==evidence
    assert len(calls)==2


@pytest.mark.parametrize('condition', ['unsettled','wrong_run','wrong_principal','active_run','active_wait'])
def test_failed_row_rejects_unsettled_or_foreign_effects_before_test(tmp_path,monkeypatch,condition):
    path,storage,row,calls=_fixture(tmp_path,monkeypatch,failure='llm',**{condition:True})
    with pytest.raises(ValueError):
        collect_search_plan(path,storage,BACKEND,SCOPE)
    assert not (storage/row['experiment_id']/'failure-settlement.json').exists()
    assert all('/result' not in path for path,_ in calls)


def test_collection_rejects_different_operator_scope_before_http(tmp_path,monkeypatch):
    path,storage,row,calls=_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='scope differs'):
        collect_search_plan(path,storage,BACKEND,'another-scope')
    assert calls==[]


def test_unknown_submission_requires_confirmed_request(tmp_path,monkeypatch):
    path,storage,row,calls=_fixture(tmp_path,monkeypatch,
        failure='experiment',failed_submission_call=True)
    with pytest.raises(ValueError,match='settled call evidence'):
        collect_search_plan(path,storage,BACKEND,SCOPE)
    assert calls==[]


def test_failure_settlement_checks_a_real_scoped_backend_run(api):
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.worker import RunWorker
    from backend.app.runs.execution import execute_claimed_run
    from backend.tests.test_finite_search import HEADERS, search_session_request
    from datetime import datetime,timezone

    client,storage,dataset=api
    body=search_session_request(dataset,['logistic_regression'],
        client_request_id='failed-collection-engineering')
    created=client.post('/api/agent/v2/sessions',headers=HEADERS,json=body)
    assert created.status_code==201,created.text
    session_id=created.json()['session_id']
    catalog=created.json()['locked_config']['preparation']['catalog']
    recipe=catalog['recipes'][0]
    submitted=client.post(f'/api/agent/v2/sessions/{session_id}/experiments',
        headers=HEADERS,json=dict(recipe_id=recipe['recipe_id'],
        recipe_digest=recipe['recipe_digest'],catalog_digest=catalog['catalog_digest'],
        knowledge_refs=[],rationale='engineering settlement',client_request_id='experiment-engineering'))
    assert submitted.status_code==202,submitted.text
    run_id=submitted.json()['run_id']
    row=dict(dataset_id=dataset,dataset_sha256=created.json()['locked_config']['dataset_sha256'])
    evidence=dict(session_id=session_id)
    with pytest.raises(ValueError,match='active'):
        _confirm_failed_search_backend(client,row,dict(run_id=run_id),evidence.copy())
    repo=RunRepository(storage/'runs.sqlite3');repo.initialize()
    worker=RunWorker(repository=repo,worker_id='settlement-engineering',
        execute=lambda record:execute_claimed_run(record,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60)
    assert worker.run_once()
    assert repo.get(run_id).state=='succeeded'
    settled=_confirm_failed_search_backend(client,row,dict(run_id=run_id),evidence)
    assert settled['settled_run_id']==run_id and settled['run_state']=='succeeded'
    with pytest.raises(ValueError,match='another experiment'):
        _confirm_failed_search_backend(client,row,dict(run_id='wrong-run'),dict(session_id=session_id))
